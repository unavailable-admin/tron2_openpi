import pytest

from openpi.training import config


@pytest.mark.parametrize(
    ("config_name", "repo_id"),
    [
        ("pi05_tron2_banana", "banana"),
        ("pi05_tron2_candy", "candy"),
        ("pi05_tron2_chess", "chess"),
        ("pi05_tron2_cloth", "cloth"),
        ("pi05_tron2_drawer", "drawer"),
        ("pi05_tron2_duck", "duck"),
        ("pi05_tron2_sort", "sort"),
    ],
)
def test_task_configs_use_lowercase_initial(config_name: str, repo_id: str):
    task_config = config.get_config(config_name)

    assert task_config.data.repo_id == repo_id


@pytest.mark.parametrize("task_name", ["Banana", "Candy", "Chess", "Cloth", "Drawer", "Duck", "SortFruit"])
def test_task_configs_do_not_register_uppercase_initial(task_name: str):
    config_name = f"pi05_tron2_{task_name}"

    with pytest.raises(ValueError, match=f"Config '{config_name}' not found"):
        config.get_config(config_name)
