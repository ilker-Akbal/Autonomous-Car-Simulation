from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_LOG_ROOT = "autonomous_driving/outputs/teknofest_sim_logs"


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


class RuntimeJsonlLogger:
    def __init__(
        self,
        *,
        node_name: str,
        file_name: str,
        log_root: str = DEFAULT_LOG_ROOT,
        session_id: Optional[str] = None,
        enabled: bool = True,
    ):
        self.node_name = node_name
        self.file_name = file_name
        self.enabled = bool(enabled)
        self.session_id = session_id or os.environ.get("TEKNOFEST_LOG_SESSION") or utc_timestamp()
        self.log_dir = Path(log_root).expanduser() / self.session_id
        self.file_path = self.log_dir / file_name

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_summary()

    def _ensure_summary(self):
        summary_path = self.log_dir / "summary.json"
        if summary_path.exists():
            return

        payload = {
            "session_id": self.session_id,
            "created_unix_s": round(time.time(), 6),
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "log_dir": str(self.log_dir),
            "expected_files": [
                "planning.jsonl",
                "lane.jsonl",
                "control.jsonl",
                "adapter.jsonl",
                "mission_runtime.jsonl",
            ],
        }
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def write(self, payload: dict):
        if not self.enabled:
            return

        record = {
            "timestamp": round(time.time(), 6),
            "node": self.node_name,
            **payload,
        }

        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def path(self) -> str:
        return str(self.file_path)
