"""Process memory and host description for metrics.json."""

from __future__ import annotations

import threading
import time

import psutil


class PeakMemory:
    """Sample this process and its children until stopped."""

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="peak-rss", daemon=True)

    def _rss(self) -> int:
        proc = psutil.Process()
        total = int(proc.memory_info().rss)
        for child in proc.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
            except psutil.Error:
                continue
        return total

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._rss())
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.peak_bytes = max(self.peak_bytes, self._rss())
        self._thread.start()

    def stop(self) -> float:
        self.peak_bytes = max(self.peak_bytes, self._rss())
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self.peak_bytes / (1024**3)


def host_summary() -> dict[str, object]:
    import platform

    vm = psutil.virtual_memory()
    cpu = platform.processor() or platform.machine()
    return {
        "cpu": cpu,
        "ram_gb": round(vm.total / (1024**3), 2),
        "gpu": None,
        "python": platform.python_version(),
        "os": platform.platform(),
    }
