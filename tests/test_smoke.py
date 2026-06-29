from src.train import run_smoke_test


def test_smoke_single_task_runs():
    result = run_smoke_test(steps=1)
    assert "loss" in result
    assert "pixel_accuracy" in result
