from src.profiling import profile_model
from src.train import build_model


def test_profile_model_reports_params_and_flops():
    model = build_model(variant="mit_b0", task_mode="single_task", num_classes=3)
    profile = profile_model(model, input_size=(1, 3, 32, 32))

    assert profile.total_params > 0
    assert profile.trainable_params == profile.total_params
    assert profile.flops > 0
