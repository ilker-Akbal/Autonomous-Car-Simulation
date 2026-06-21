from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DEFAULT_LOG_ROOT = "autonomous_driving/outputs/teknofest_sim_logs"


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def default_session_id(prefix: str = "run") -> str:
    safe_prefix = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in prefix).strip("_")
    return f"{safe_prefix or 'run'}_{utc_timestamp()}"


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
        self.session_id = session_id or os.environ.get("TEKNOFEST_LOG_SESSION") or default_session_id(node_name)
        resolved_log_root = os.environ.get("TEKNOFEST_LOG_ROOT", log_root)
        self.log_dir = Path(resolved_log_root).expanduser() / self.session_id
        self.file_path = self.log_dir / file_name
        self.summary_path = self.log_dir / "summary.json"

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_summary()

    def _ensure_summary(self):
        payload = self._read_summary()
        if payload:
            return

        self.update_summary({
            "session_id": self.session_id,
            "created_unix_s": round(time.time(), 6),
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "log_dir": str(self.log_dir),
            "log_root": str(self.log_dir.parent),
            "log_files": [],
        })

    def _read_summary(self) -> dict[str, Any]:
        if not self.summary_path.exists():
            return {}
        try:
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def update_summary(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return

        summary = self._read_summary()
        summary.update(payload)
        log_files = list(summary.get("log_files", []))
        if self.file_name not in log_files:
            log_files.append(self.file_name)
        summary["log_files"] = sorted(log_files)
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

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
