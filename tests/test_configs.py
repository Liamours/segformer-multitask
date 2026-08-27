from src.configs import ConfigHandler


def test_config_handler_from_dict_overrides_sections():
    config = ConfigHandler.from_dict(
        {
            "model": {"variant": "mit_b3", "task_mode": "dual_head", "task_a_classes": 3, "task_b_classes": 6},
            "data": {"dataset_name": "dummy", "num_samples": 12, "image_size": [32, 48]},
            "run": {"epochs": 3, "max_train_batches": 2, "allow_limited_batches": True},
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


def test_real_training_configs_do_not_inherit_smoke_batch_limits():
    config = ConfigHandler.from_json("configs/folder_single_task_100epochs.json")

    assert config.run.max_train_batches is None
    assert config.run.max_eval_batches is None


def test_b1_training_config_is_full_pretrained_run():
    config = ConfigHandler.from_json("configs/folder_single_task_b1_100epochs.json")

    # mit_b1 is an exploratory variant outside the three-model ablation; batch_size 1 is its own
    # setting, not a smoke-test leftover. This test guards the leftovers below, not the batch size.
    assert config.model.variant == "mit_b1"
    assert config.model.pretrained_hf_name == "nvidia/mit-b1"
    assert not config.model.freeze_backbone
    assert config.data.batch_size == 1
    assert config.data.image_size == (1024, 512)
    assert config.scheduler.warmup_iters == 1500
    assert config.run.epochs == 100
    assert config.run.max_train_batches is None
    assert config.run.max_eval_batches is None


def test_config_rejects_limited_batches_without_explicit_opt_in():
    try:
        ConfigHandler.from_dict({"run": {"max_train_batches": 1}})
    except ValueError as error:
        assert "allow_limited_batches" in str(error)
    else:
        raise AssertionError("Expected ValueError for implicit limited-batch run.")


def test_config_accepts_foreground_checkpoint_metric():
    config = ConfigHandler.from_dict({"logging": {"checkpoint_metric": "foreground_mean_dice"}})

    assert config.logging.checkpoint_metric == "foreground_mean_dice"


def test_config_accepts_huggingface_pretrained_single_task():
    config = ConfigHandler.from_dict({"model": {"pretrained_hf_name": "nvidia/mit-b2"}})

    assert config.model.pretrained_hf_name == "nvidia/mit-b2"


def test_config_accepts_huggingface_pretrained_for_multitask_modes():
    config = ConfigHandler.from_dict(
        {"model": {"task_mode": "dual_head", "task_a_classes": 3, "task_b_classes": 13, "pretrained_hf_name": "nvidia/mit-b2"}}
    )

    assert config.model.pretrained_hf_name == "nvidia/mit-b2"
    assert config.model.task_mode == "dual_head"


def test_config_accepts_dual_fuse_task_mode():
    config = ConfigHandler.from_dict(
        {"model": {"task_mode": "dual_fuse", "task_a_classes": 3, "task_b_classes": 13, "pretrained_hf_name": "nvidia/mit-b2"}}
    )

    assert config.model.task_mode == "dual_fuse"
    assert config.model.pretrained_hf_name == "nvidia/mit-b2"


def test_dual_fuse_training_config_loads_and_validates():
    config = ConfigHandler.from_json("configs/folder_dual_fuse.json")

    assert config.model.task_mode == "dual_fuse"
    assert config.model.variant == "mit_b2"
    assert config.model.task_a_classes == 3
    assert config.model.task_b_classes == 13
    assert config.loss.task_a_weight == 0.8
    assert config.loss.task_b_weight == 0.2
    assert config.run.max_train_batches is None
    assert config.run.max_eval_batches is None
