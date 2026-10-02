"""Prometheus-compatible application metrics and telemetry exporter."""
from __future__ import annotations

from threading import Lock
import time
from typing import Any


class MetricsRegistry:
    """Thread-safe Prometheus-compatible metrics registry."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
        self._histograms: dict[str, list[float]] = {}

    def inc_counter(self, name: str, value: float = 1.0, **labels: str) -> None:
        sorted_labels = tuple(sorted(labels.items()))
        with self._lock:
            counter_map = self._counters.setdefault(name, {})
            counter_map[sorted_labels] = counter_map.get(sorted_labels, 0.0) + value

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        sorted_labels = tuple(sorted(labels.items()))
        with self._lock:
            gauge_map = self._gauges.setdefault(name, {})
            gauge_map[sorted_labels] = value

    def observe_histogram(self, name: str, value: float) -> None:
        with self._lock:
            hist_list = self._histograms.setdefault(name, [])
            hist_list.append(value)

    def render_prometheus_text(self) -> str:
        """Render registered metrics into Prometheus exposition text format."""
        lines: list[str] = []
        with self._lock:
            # Counters
            for name, entries in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                for labels, val in entries.items():
                    if labels:
                        lbl_str = ",".join(f'{k}="{v}"' for k, v in labels)
                        lines.append(f"{name}{{{lbl_str}}} {val}")
                    else:
                        lines.append(f"{name} {val}")

            # Gauges
            for name, entries in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                for labels, val in entries.items():
                    if labels:
                        lbl_str = ",".join(f'{k}="{v}"' for k, v in labels)
                        lines.append(f"{name}{{{lbl_str}}} {val}")
                    else:
                        lines.append(f"{name} {val}")

            # Histograms (count & sum)
            for name, values in self._histograms.items():
                lines.append(f"# TYPE {name} summary")
                count = len(values)
                total = sum(values)
                lines.append(f"{name}_count {count}")
                lines.append(f"{name}_sum {round(total, 4)}")

        return "\n".join(lines) + "\n"


GLOBAL_METRICS = MetricsRegistry()
