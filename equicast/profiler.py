"""Custom profilers for equicast."""

import re
from pathlib import Path

from pytorch_lightning.profilers import AdvancedProfiler


class LoggingProfiler(AdvancedProfiler):
    """Advanced profiler that uploads profiling results to logger."""

    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def summary(self) -> str:
        summary_text = super().summary()
        summary_text = filter_and_order_profiler_summary(summary_text)
        self._log_summary(summary_text)
        return ""

    #
    def _log_summary(self, summary_text: str):
        """Upload profiler summary to logger."""
        if self._output_file:
            self._log_file(self._output_file)
        else:
            self._log_text(summary_text)

    def _log_file(self, file_path: str):
        """Upload existing profiler file to logger."""
        self.logger.log_artifact(file_path, artifact_path="profiler")

    def _log_text(self, text: str):
        """Save text to temp file and upload to logger."""
        temp_file = Path("profiler_summary.txt")
        temp_file.write_text(text)
        self.logger.log_artifact(str(temp_file), artifact_path="profiler")


def filter_and_order_profiler_summary(text, thresh=1e-2):
    sections = re.split(r"\n(?=Profile stats for:)", text)
    kept = []

    for section in sections:
        m = re.search(r"in ([0-9.]+) seconds", section)
        if not m:
            continue

        total_time = float(m.group(1))
        if total_time >= thresh:
            kept.append((total_time, section))

    kept.sort(key=lambda x: x[0], reverse=True)

    return "\n".join(section for _, section in kept)
