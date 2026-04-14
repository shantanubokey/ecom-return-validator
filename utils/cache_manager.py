"""
Lightweight Cache Manager
- Model cache: singleton model instance (avoid reloading)
- Image cache: preprocessed tensors keyed by image hash
- Result cache: final JSON keyed by (image_hashes + metadata_hash)
- TTL-based invalidation
"""

import hashlib
import time
import torch
from typing import Dict, Optional, Any
from PIL import Image
import io


class ImageCache:
    """LRU-style cache for preprocessed image tensors."""

    def __init__(self, max_size: int = 64, ttl_seconds: int = 3600):
        self._cache:      Dict[str, Any]   = {}
        self._timestamps: Dict[str, float] = {}
        self.max_size    = max_size
        self.ttl         = ttl_seconds
        self.hits        = 0
        self.misses      = 0

    def _hash_image(self, image_bytes: bytes) -> str:
        return hashlib.md5(image_bytes).hexdigest()

    def _hash_path(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                return self._hash_image(f.read())
        except Exception:
            return hashlib.md5(path.encode()).hexdigest()

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, t in self._timestamps.items() if now - t > self.ttl]
        for k in expired:
            self._cache.pop(k, None)
            self._timestamps.pop(k, None)

    def _evict_lru(self):
        if len(self._cache) >= self.max_size:
            oldest = min(self._timestamps, key=self._timestamps.get)
            self._cache.pop(oldest, None)
            self._timestamps.pop(oldest, None)

    def get(self, path: str) -> Optional[Any]:
        self._evict_expired()
        key = self._hash_path(path)
        if key in self._cache:
            self.hits += 1
            self._timestamps[key] = time.time()   # refresh TTL
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, path: str, tensor: Any):
        self._evict_lru()
        key = self._hash_path(path)
        self._cache[key]      = tensor
        self._timestamps[key] = time.time()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict:
        return {
            "size":     len(self._cache),
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": round(self.hit_rate, 3),
        }

    def clear(self):
        self._cache.clear()
        self._timestamps.clear()


class ResultCache:
    """Cache final validation results keyed by image + metadata hash."""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 1800):
        self._cache:      Dict[str, Dict]  = {}
        self._timestamps: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl      = ttl_seconds
        self.hits     = 0
        self.misses   = 0

    def _make_key(self, delivery_paths, vendor_paths, metadata: Dict) -> str:
        parts = sorted(delivery_paths) + sorted(vendor_paths)
        content = "|".join(parts) + "|" + str(sorted(metadata.items()))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, delivery_paths, vendor_paths, metadata) -> Optional[Dict]:
        key = self._make_key(delivery_paths, vendor_paths, metadata)
        now = time.time()
        if key in self._cache:
            if now - self._timestamps[key] < self.ttl:
                self.hits += 1
                return self._cache[key]
            else:
                del self._cache[key]
                del self._timestamps[key]
        self.misses += 1
        return None

    def set(self, delivery_paths, vendor_paths, metadata, result: Dict):
        if len(self._cache) >= self.max_size:
            oldest = min(self._timestamps, key=self._timestamps.get)
            self._cache.pop(oldest, None)
            self._timestamps.pop(oldest, None)
        key = self._make_key(delivery_paths, vendor_paths, metadata)
        self._cache[key]      = result
        self._timestamps[key] = time.time()

    def invalidate_all(self):
        self._cache.clear()
        self._timestamps.clear()
        print("[Cache] Result cache cleared")

    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "size":     len(self._cache),
            "hits":     self.hits,
            "misses":   self.misses,
            "hit_rate": round(self.hits / total if total > 0 else 0, 3),
        }


class ModelCache:
    """Singleton model cache — load once, reuse forever."""

    _instance = None
    _model    = None
    _tokenizer = None

    @classmethod
    def get_model(cls):
        return cls._model

    @classmethod
    def get_tokenizer(cls):
        return cls._tokenizer

    @classmethod
    def set(cls, model, tokenizer):
        cls._model     = model
        cls._tokenizer = tokenizer
        print("[Cache] Model cached in memory")

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    @classmethod
    def clear(cls):
        if cls._model is not None:
            del cls._model
            del cls._tokenizer
            cls._model     = None
            cls._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print("[Cache] Model evicted from memory")


def clear_gpu_memory():
    """Aggressively free unused CUDA memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved  = torch.cuda.memory_reserved()  / 1024**2
        print(f"[GPU] After clear — allocated: {allocated:.0f}MB, reserved: {reserved:.0f}MB")


# Global singletons
image_cache  = ImageCache()
result_cache = ResultCache()
