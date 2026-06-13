from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


class CompassCSVLogger:
    """Append-only CSV logger for Safety Compass measurements."""

    def __init__(
        self,
        path: str,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = None
        self._file = open(self.path, "w", newline="")
        self._writer = None
        self._wandb_run = self._init_wandb(wandb_project, wandb_run_name)

    @staticmethod
    def _init_wandb(project: Optional[str], run_name: Optional[str]):
        if not project:
            return None
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb_project was provided, but wandb is not installed. "
                "Install safety-compass with the wandb extra or omit wandb_project."
            ) from exc
        return wandb.init(project=project, name=run_name)

    @staticmethod
    def _ordered_fields(row: dict) -> list[str]:
        priority = ["step", "epoch", "elapsed_seconds"]
        leading = [field for field in priority if field in row]
        rest = sorted(field for field in row.keys() if field not in leading)
        return leading + rest

    def log(self, row: dict):
        if self.fieldnames is None:
            self.fieldnames = self._ordered_fields(row)
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
        else:
            missing = set(self.fieldnames) - set(row.keys())
            extra = set(row.keys()) - set(self.fieldnames)
            if missing or extra:
                raise ValueError(
                    "CSV schema changed after first write. "
                    f"Missing={sorted(missing)}, extra={sorted(extra)}"
                )

        self._writer.writerow(row)
        self._file.flush()
        if self._wandb_run is not None:
            self._wandb_run.log(row, step=int(row["step"]))

    def close(self):
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
