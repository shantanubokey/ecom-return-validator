"""
InternVL2.5-4B MPO + LoRA Inference Engine
Fine-tuned on ~500K delivery/vendor image pairs for return validation.
"""

import torch
import json
import re
from PIL import Image
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


MODEL_ID   = "OpenGVLab/InternVL2_5-4B-MPO"
LORA_PATH  = "./lora_weights"   # path to your fine-tuned LoRA adapter

SYSTEM_PROMPT = """You are a strict return validation AI for an e-commerce platform.
You compare delivery images with vendor images and metadata to detect fraud.
You must respond ONLY with valid JSON — no explanation, no markdown, no extra text.
"""

VALIDATION_PROMPT = """Compare the 4 delivery images (what the customer is returning) 
with the 4 vendor images (original product) and the vendor metadata below.

Vendor Metadata:
- Product: {product}
- Brand: {brand}
- Color: {color}
- Design: {design}
- Quantity: {quantity}

Answer each field strictly as "yes" or "no":
- product_match: Does the returned item match the vendor product type?
- design_match: Does the design/pattern match the vendor images?
- color_match: Does the color match exactly?
- quantity_is_one: Is exactly 1 unit being returned?
- is_damaged: Is the product visibly damaged (torn, broken, scratched)?
- is_used: Does the product show signs of use (worn, dirty, missing tags)?
- accept_return: "yes" ONLY if product_match=yes AND design_match=yes AND color_match=yes AND quantity_is_one=yes AND is_damaged=no AND is_used=no

Return ONLY this JSON:
{{
  "product_match": "yes/no",
  "design_match": "yes/no",
  "color_match": "yes/no",
  "quantity_is_one": "yes/no",
  "is_damaged": "yes/no",
  "is_used": "yes/no",
  "accept_return": "yes/no"
}}"""


REQUIRED_FIELDS = [
    "product_match", "design_match", "color_match",
    "quantity_is_one", "is_damaged", "is_used", "accept_return"
]


class ReturnValidator:
    def __init__(self, use_lora: bool = False, device: str = "auto",
                 load_in_4bit: bool = True):
        self.device     = device
        self.use_lora   = use_lora
        self.model      = None
        self.tokenizer  = None
        self._loaded    = False
        self.load_in_4bit = load_in_4bit

    def load(self):
        if self._loaded:
            return
        print(f"[Model] Loading {MODEL_ID}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )

        quant_kwargs = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device,
            **quant_kwargs,
        )

        if self.use_lora:
            print(f"[Model] Loading LoRA weights from {LORA_PATH}...")
            self.model = PeftModel.from_pretrained(self.model, LORA_PATH)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self._loaded = True
        print("[Model] Ready")

    def _load_images(self, paths: List[str]) -> List[Image.Image]:
        images = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((448, 448))
                images.append(img)
            except Exception as e:
                print(f"[Warning] Could not load image {p}: {e}")
                images.append(Image.new("RGB", (448, 448), color=(128, 128, 128)))
        return images

    def validate(
        self,
        delivery_images: List[str],   # 4 paths
        vendor_images:   List[str],   # 4 paths
        metadata: Dict,
    ) -> Dict:
        """
        Run return validation.

        Args:
            delivery_images: list of 4 image paths (customer return photos)
            vendor_images:   list of 4 image paths (original product photos)
            metadata: dict with keys: product, brand, color, design, quantity

        Returns:
            dict with all 7 validation fields
        """
        self.load()

        delivery_imgs = self._load_images(delivery_images[:4])
        vendor_imgs   = self._load_images(vendor_images[:4])
        all_images    = delivery_imgs + vendor_imgs   # 8 total

        prompt = VALIDATION_PROMPT.format(
            product  = metadata.get("product",  "unknown"),
            brand    = metadata.get("brand",    "unknown"),
            color    = metadata.get("color",    "unknown"),
            design   = metadata.get("design",   "unknown"),
            quantity = metadata.get("quantity", "1"),
        )

        # Build InternVL2 conversation format
        pixel_values = self._preprocess_images(all_images)

        # Image tokens: <image> for each of 8 images
        image_tokens = "".join([f"<image_{i+1}>\n" for i in range(len(all_images))])
        full_prompt  = f"{image_tokens}\n{prompt}"

        response = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            question=full_prompt,
            generation_config=dict(
                max_new_tokens=256,
                temperature=0.01,
                do_sample=False,
            ),
        )

        return self._parse_response(response)

    def _preprocess_images(self, images: List[Image.Image]):
        """Convert PIL images to model input tensors."""
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        tensors = torch.stack([transform(img) for img in images])
        return tensors.to(next(self.model.parameters()).device)

    def _parse_response(self, response: str) -> Dict:
        """Parse JSON from model response with fallback."""
        # Try direct JSON parse
        try:
            start = response.find("{")
            end   = response.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(response[start:end])
                # Validate all fields present
                for field in REQUIRED_FIELDS:
                    if field not in result:
                        result[field] = "no"
                    result[field] = result[field].lower().strip()
                    if result[field] not in ("yes", "no"):
                        result[field] = "no"
                return result
        except Exception:
            pass

        # Fallback: regex extraction
        result = {}
        for field in REQUIRED_FIELDS:
            pattern = rf'"{field}"\s*:\s*"(yes|no)"'
            match   = re.search(pattern, response, re.IGNORECASE)
            result[field] = match.group(1).lower() if match else "no"

        return result

    def validate_accept_return(self, result: Dict) -> str:
        """
        Enforce the accept_return logic regardless of model output.
        accept = yes only if all match conditions met and not damaged/used.
        """
        should_accept = (
            result.get("product_match")   == "yes" and
            result.get("design_match")    == "yes" and
            result.get("color_match")     == "yes" and
            result.get("quantity_is_one") == "yes" and
            result.get("is_damaged")      == "no"  and
            result.get("is_used")         == "no"
        )
        result["accept_return"] = "yes" if should_accept else "no"
        return result
