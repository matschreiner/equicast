"""Custom profilers for equicast."""

from pathlib import Path

from pytorch_lightning.profilers import PyTorchProfiler, SimpleProfiler


class LoggingProfiler(SimpleProfiler):
    """Advanced profiler that uploads profiling results to logger."""

    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def summary(self) -> str:
        summary_text = super().summary()
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


class CUDAProfiler(PyTorchProfiler):
    """PyTorch profiler with CUDA timing, uploads trace and summary to logger."""

    def __init__(self, logger, wait=5, warmup=5, active=20, **kwargs):
        import torch

        kwargs.setdefault("dirpath", ".")
        kwargs.setdefault("filename", "cuda_profile")
        super().__init__(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            record_module_names=False,
            with_stack=False,
            **kwargs,
        )
        self._logger = logger

    def __getstate__(self):
        # Exclude unpicklable torch profiler internals during checkpointing
        state = self.__dict__.copy()
        for key in ("profiler", "_parent_profiler", "_register", "_recording_map"):
            state.pop(key, None)
        return state

    def summary(self) -> str:
        summary_text = super().summary()
        if summary_text:
            summary_file = Path("cuda_profiler_summary.txt")
            summary_file.write_text(summary_text)
            self._logger.log_artifact(str(summary_file), artifact_path="profiler")
        return summary_text

    def teardown(self, stage=None):
        try:
            super().teardown(stage=stage)
        except (AssertionError, Exception):
            pass
        # Upload trace file to MLflow
        dirpath = Path(self.dirpath) if self.dirpath else Path(".")
        for trace_file in dirpath.glob(f"{self.filename}*.json"):
            self._logger.log_artifact(str(trace_file), artifact_path="profiler")

