from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from contextlib import contextmanager
from typing import Any, Dict


def dynamic_set_attr(obj: object, kwargs: dict, path: list):
    """Dynamically set attributes on an object from a nested dictionary."""
    if kwargs is None:
        return

    for k, v in kwargs.items():
        if hasattr(obj, k):
            attr = getattr(obj, k)
            if isinstance(v, dict) and hasattr(attr, "__dict__"):
                next_path = path.copy()
                next_path.append(k)
                dynamic_set_attr(attr, v, next_path)
            else:
                try:
                    current_val = getattr(obj, k)
                    if isinstance(
                        current_val, (int, float, bool, str)
                    ) and not isinstance(v, type(current_val)):
                        # Type conversion if needed
                        v = type(current_val)(v)
                    setattr(obj, k, v)
                    print(f"Set {'.'.join(path + [k])} from {getattr(obj, k)} to {v}")
                except Exception as e:
                    print(f"Error setting attribute {'.'.join(path + [k])}: {e}")
        else:
            print(f"Warning: Attribute {k} not found in {'.'.join(path)}")


@contextmanager
def hydra_context_base(**init_kwargs):
    """
    基础的 Hydra 上下文管理器，只负责状态管理

    Args:
        **init_kwargs: 传递给 hydra.initialize() 的参数

    Yields:
        None: 在此上下文中可以安全使用 Hydra 操作
    """
    gh = GlobalHydra.instance()
    prev_hydra = None

    # 保存当前 Hydra 状态
    if gh.is_initialized():
        print("[hydra_context_base] 检测到已存在的Hydra实例，将进行暂存和恢复。")
        prev_hydra = gh.hydra
        gh.clear()
        print("[hydra_context_base] 已清除现有的Hydra实例")

    try:
        yield  # 让调用者在这里执行 Hydra 操作
    finally:
        # 恢复原始 Hydra 状态
        print("[hydra_context_base] 清理临时上下文并恢复原始Hydra实例...")
        gh.clear()  # 清除临时状态
        if prev_hydra is not None:
            gh.initialize(prev_hydra)  # 恢复主程序的状态
        print("[hydra_context_base] 上下文恢复完毕。")


@contextmanager
def hydra_config_context(config_path: str, config_name: str | None = None, **kwargs):
    """
    完整的 Hydra 配置上下文管理器（使用基础版本）
    """
    with hydra_context_base():
        # 在安全的上下文中执行 Hydra 操作
        with initialize(config_path=config_path, **kwargs):
            if config_name is not None:
                cfg = compose(config_name=config_name)
                yield cfg
            else:
                yield None


def load_skill_config(
    config_path: str = "../conf", config_name: str = "config"
) -> Dict[str, Any]:
    """
    加载技能配置文件

    Args:
        config_path: 配置文件相对路径，默认为 "../conf"
        config_name: 配置文件名称，默认为 "config"

    Returns:
        配置字典
    """
    try:
        with hydra_config_context(config_path, config_name) as cfg:
            # 将 OmegaConf 转换为普通字典以避免后续依赖问题
            from omegaconf import OmegaConf

            return OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        print(f"[load_skill_config] 加载配置失败: {e}")
        raise
