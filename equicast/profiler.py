"""Custom profilers for equicast."""

from pathlib import Path

from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler


class LoggingProfiler(AdvancedProfiler):
    """Advanced profiler that uploads profiling results to logger."""

    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    @override
    def summary(self) -> str:
        recorded_stats = {}
        for action_name, pr in self.profiled_actions.items():
            if self.dump_stats:
                self._dump_stats(action_name, pr)
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
            ps.print_stats(self.line_count_restriction)
            recorded_stats[action_name] = s.getvalue()
        summary_text = self._stats_to_str(recorded_stats)
        self._log_summary(summary_text)
        return ""

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
