import csv
from types import SimpleNamespace

from safety_compass.callback import SafetyCompassCallback


class FakeModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class FakeMonitor:
    def __init__(self):
        self.model = None
        self.is_setup = False
        self.setup_calls = 0
        self.measure_calls = []

    def set_model(self, model):
        self.model = model

    def setup(self):
        self.setup_calls += 1
        self.is_setup = True

    def measure(self, step, epoch=None):
        self.measure_calls.append((step, epoch))
        return {
            "step": step,
            "epoch": "" if epoch is None else epoch,
            "elapsed_seconds": 1.0 + step,
            "refusal_cosine_to_baseline": 1.0,
        }


def test_callback_logs_begin_interval_and_skips_duplicate_train_end(tmp_path):
    monitor = FakeMonitor()
    model = FakeModel()
    callback = SafetyCompassCallback(
        monitor=monitor,
        measure_every_n_steps=2,
        log_file=str(tmp_path / "drift.csv"),
    )
    args = SimpleNamespace()
    control = SimpleNamespace()

    callback.on_train_begin(
        args,
        SimpleNamespace(global_step=0, epoch=0.0),
        control,
        model=model,
    )
    callback.on_step_end(
        args,
        SimpleNamespace(global_step=1, epoch=0.1),
        control,
        model=model,
    )
    callback.on_step_end(
        args,
        SimpleNamespace(global_step=2, epoch=0.2),
        control,
        model=model,
    )
    callback.on_train_end(
        args,
        SimpleNamespace(global_step=2, epoch=0.2),
        control,
        model=model,
    )

    assert monitor.setup_calls == 1
    assert monitor.measure_calls == [(0, 0.0), (2, 0.2)]
    assert model.training is True

    with open(tmp_path / "drift.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    assert [row["step"] for row in rows] == ["0", "2"]
    assert rows[0]["refusal_cosine_to_baseline"] == "1.0"
