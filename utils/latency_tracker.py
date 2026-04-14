"""
End-to-end latency tracker.
Tracks preprocessing, inference, post-processing per request.
"""

import time
import json
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict
from collections import deque


@dataclass
class LatencyRecord:
    request_id:        str
    preprocessing_ms:  float = 0.0
    inference_ms:      float = 0.0
    postprocessing_ms: float = 0.0
    total_ms:          float = 0.0
    cached:            bool  = False
    timestamp:         float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


class LatencyTracker:
    """Tracks per-request latency with rolling history."""

    def __init__(self, history_size: int = 100):
        self._history: deque = deque(maxlen=history_size)
        self._current: Dict  = {}

    def start(self, request_id: str, stage: str):
        self._current[f"{request_id}_{stage}"] = time.perf_counter()

    def stop(self, request_id: str, stage: str) -> float:
        key   = f"{request_id}_{stage}"
        start = self._current.pop(key, None)
        if start is None:
            return 0.0
        return (time.perf_counter() - start) * 1000   # ms

    def record(self, record: LatencyRecord):
        self._history.append(record)

    def get_stats(self) -> Dict:
        if not self._history:
            return {}
        records = list(self._history)
        non_cached = [r for r in records if not r.cached]

        def avg(vals): return sum(vals) / len(vals) if vals else 0

        return {
            "total_requests":      len(records),
            "cached_requests":     sum(1 for r in records if r.cached),
            "avg_total_ms":        round(avg([r.total_ms for r in non_cached]), 1),
            "avg_preprocessing_ms":round(avg([r.preprocessing_ms for r in non_cached]), 1),
            "avg_inference_ms":    round(avg([r.inference_ms for r in non_cached]), 1),
            "avg_postprocessing_ms":round(avg([r.postprocessing_ms for r in non_cached]), 1),
            "min_total_ms":        round(min((r.total_ms for r in non_cached), default=0), 1),
            "max_total_ms":        round(max((r.total_ms for r in non_cached), default=0), 1),
            "last_10": [r.to_dict() for r in list(self._history)[-10:]],
        }

    def log(self, record: LatencyRecord):
        self.record(record)
        status = "CACHED" if record.cached else f"{record.total_ms:.0f}ms"
        print(f"[Latency] {record.request_id} | {status} | "
              f"pre={record.preprocessing_ms:.0f}ms "
              f"inf={record.inference_ms:.0f}ms "
              f"post={record.postprocessing_ms:.0f}ms")


# Global singleton
tracker = LatencyTracker()
