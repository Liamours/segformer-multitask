from src.train import run_smoke_test


def test_smoke_single_task_runs():
    result = run_smoke_test(steps=1)
    assert "loss" in result
    assert "pixel_accuracy" in result


def test_smoke_dual_head_runs():
    result = run_smoke_test(task_mode="dual_head", steps=1)
    assert "loss" in result
    assert "pixel_accuracy" in result


def test_smoke_dual_decoder_runs():
    result = run_smoke_test(task_mode="dual_decoder", steps=1)
    assert "loss" in result
    assert "pixel_accuracy" in result
