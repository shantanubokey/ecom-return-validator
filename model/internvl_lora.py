"""
InternVL2.5-4B MPO + LoRA Inference Engine
Optimized with: model cache, image cache, result cache,
GPU memory management, 4-bit inference, latency tracking.
"""

import torch
import json
import re
import uuid
import time
from PIL import Image
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel

from utils.cache_manager import ModelCache, image_cache, result_cache, clear_gpu_memory
from utils.latency_tracker import tracker, LatencyRecord


MODEL_ID  = "OpenGVLab/InternVL2_5-4B-MPO"
LORA_PATH = "./lora_weights"

REQUIRED_FIELDS = [
    "product_match", "design_match", "color_match",
    "quantity_is_one", "is_damaged", "is_used", "accept_return"
]

VALIDATION_PROMPT = """Compare the 4 delivery images (customer return) with the 4 vendor images (original product) and metadata below.

Vendor Metadata:
- Product: {product}
- Brand: {brand}
- Color: {color}
- Design: {design}
- Quantity: {quantity}

Answer strictly yes/no for each field. Return ONLY this JSON:
{{
  "product_match": "yes/no",
  "design_match": "yes/no",
  "color_match": "yes/no",
  "quantity_is_one": "yes/no",
  "is_damaged": "yes/no",
  "is_used": "yes/no",
  "accept_return": "yes/no"
}}"""


class ReturnValidator:
    def __init__(self, use_lora: bool = False, device: str = "auto",
                 load_in_4bit: bool = True):
        self.use_lora     = use_lora
        self.device       = device
        self.load_in_4bit = load_in_4bit

    def load(self):
        if ModelCache.is_loaded():
            return

        print(f"[Model] Loading {MODEL_ID}...")
        t0 = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ) if self.load_in_4bit else None

        model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device,
            quantization_config=quant_cfg,
        )

        if self.use_lora:
            model = PeftModel.from_pretrained(model, LORA_PATH)
            model = model.merge_and_unload()

        model.eval()
        ModelCache.set(model, tokenizer)
        clear_gpu_memory()
        print(f"[Model] Loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

    def _preprocess_image(self, path: str):
        """Load and preprocess one image with caching."""
        cached = image_cache.get(path)
        if cached is not None:
            return cached

        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (448, 448), (128, 128, 128))

        tensor = transform(img)
        image_cache.set(path, tensor)
        return tensor

    def validate(
        self,
        delivery_images: List[str],
        vendor_images:   List[str],
        metadata:        Dict,
        request_id:      str = None,
    ) -> Dict:
        request_id = request_id or str(uuid.uuid4())[:8]
        record     = LatencyRecord(request_id=request_id)
        t_total    = time.perf_counter()

        # ── Check result cache ─────────────────────────────────────────────────
        cached_result = result_cache.get(delivery_images, vendor_images, metadata)
        if cached_result is not None:
            record.cached   = True
            record.total_ms = (time.perf_counter() - t_total) * 1000
            tracker.log(record)
            return cached_result

        self.load()
        model     = ModelCache.get_model()
        tokenizer = ModelCache.get_tokenizer()

        # ── Preprocessing ──────────────────────────────────────────────────────
        tracker.start(request_id, "pre")
        all_paths  = delivery_images[:4] + vendor_images[:4]
        tensors    = [self._preprocess_image(p) for p in all_paths]
        # Stack and move to GPU in one transfer (minimize transfers)
        pixel_values = torch.stack(tensors).to(
            next(model.parameters()).device, non_blocking=True
        )
        record.preprocessing_ms = tracker.stop(request_id, "pre")

        # ── Inference ──────────────────────────────────────────────────────────
        tracker.start(request_id, "inf")
        prompt = VALIDATION_PROMPT.format(**{
            k: metadata.get(k, "unknown") for k in
            ["product", "brand", "color", "design", "quantity"]
        })
        image_tokens = "".join([f"<image_{i+1}>\n" for i in range(len(all_paths))])
        full_prompt  = f"{image_tokens}\n{prompt}"

        with torch.inference_mode():          # faster than no_grad for inference
            response = model.chat(
                tokenizer,
                pixel_values=pixel_values,
                question=full_prompt,
                generation_config=dict(
                    max_new_tokens=200,
                    temperature=0.01,
                    do_sample=False,
                    use_cache=True,
                ),
            )
        record.inference_ms = tracker.stop(request_id, "inf")

        # ── Post-processing ────────────────────────────────────────────────────
        tracker.start(request_id, "post")
        result = self._parse_response(response)
        result = self.validate_accept_return(result)
        record.postprocessing_ms = tracker.stop(request_id, "post")

        # Free intermediate tensors immediately
        del pixel_values
        clear_gpu_memory()

        # Cache result
        result_cache.set(delivery_images, vendor_images, metadata, result)

        record.total_ms = (time.perf_counter() - t_total) * 1000
        tracker.log(record)
        result["_latency_ms"] = round(record.total_ms, 1)
        return result

    def _parse_response(self, response: str) -> Dict:
        try:
            start = response.find("{")
            end   = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                for f in REQUIRED_FIELDS:
                    v = str(result.get(f, "no")).lower().strip()
                    result[f] = v if v in ("yes", "no") else "no"
                return result
        except Exception:
            pass
        result = {}
        for f in REQUIRED_FIELDS:
            m = re.search(rf'"{f}"\s*:\s*"(yes|no)"', response, re.IGNORECASE)
            result[f] = m.group(1).lower() if m else "no"
        return result

    def validate_accept_return(self, result: Dict) -> Dict:
        should = (
            result.get("product_match")   == "yes" and
            result.get("design_match")    == "yes" and
            result.get("color_match")     == "yes" and
            result.get("quantity_is_one") == "yes" and
            result.get("is_damaged")      == "no"  and
            result.get("is_used")         == "no"
        )
        result["accept_return"] = "yes" if should else "no"
        return result
