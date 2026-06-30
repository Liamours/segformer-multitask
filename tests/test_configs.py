from src.configs import ConfigHandler


def test_config_handler_from_dict_overrides_sections():
    config = ConfigHandler.from_dict(
        {
            "model": {"variant": "mit_b3", "task_mode": "dual_head", "task_a_classes": 3, "task_b_classes": 6},
            "data": {"dataset_name": "dummy", "num_samples": 12, "image_size": [32, 48]},
            "run": {"epochs": 3, "max_train_batches": 2},
        }
    )

    assert config.model.variant == "mit_b3"
    assert config.model.task_mode == "dual_head"
    assert config.model.task_a_classes == 3
    assert config.model.task_b_classes == 6
    assert config.data.num_samples == 12
    assert config.data.image_size == (32, 48)
    assert config.run.epochs == 3
    assert config.run.max_train_batches == 2


def test_config_handler_json_roundtrip(tmp_path):
    config = ConfigHandler.from_dict({"logging": {"output_dir": str(tmp_path), "run_name": "roundtrip"}})
    path = tmp_path / "config.json"
    ConfigHandler.to_json(config, path)
    loaded = ConfigHandler.from_json(path)

    assert loaded.logging.output_dir == str(tmp_path)
    assert loaded.logging.run_name == "roundtrip"


def test_config_handler_requires_root_for_folder_dataset():
    try:
        ConfigHandler.from_dict({"data": {"dataset_name": "folder"}})
    except ValueError as error:
        assert "root_dir" in str(error)
    else:
        raise AssertionError("Expected ValueError for missing root_dir.")


def test_config_handler_rejects_invalid_checkpoint_metric():
    try:
        ConfigHandler.from_dict({"logging": {"checkpoint_metric": "dice"}})
    except ValueError as error:
        assert "checkpoint_metric" in str(error)
    else:
        raise AssertionError("Expected ValueError for invalid checkpoint_metric.")
