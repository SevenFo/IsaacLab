from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, cast
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from omegaconf import DictConfig, OmegaConf


_RESOLVERS_REGISTERED = False


def ensure_default_resolvers() -> None:
    """确保注册默认的 OmegaConf 解析器。"""
    global _RESOLVERS_REGISTERED
    if _RESOLVERS_REGISTERED:
        return

    if not OmegaConf.has_resolver("now"):

        def _now_resolver(pattern: str = "%Y-%m-%d") -> str:
            if pattern.startswith(":"):
                pattern = pattern[1:]
            if not pattern:
                pattern = "%Y-%m-%d"
            return datetime.now().strftime(pattern)

        OmegaConf.register_new_resolver("now", _now_resolver)

    _RESOLVERS_REGISTERED = True


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


def load_config(
    config_path: Optional[str | Path] = None,
    config_name: Optional[str] = None,
    config: Optional[Dict[str, Any] | DictConfig] = None,
    overrides: Optional[Sequence[str]] = None,
    return_dict: bool = False,
    resolve: bool = True,
    allow_struct: bool = False,
) -> DictConfig | Dict[str, Any]:
    """
    统一的配置加载函数，支持多种配置源和输出格式。

    这是一个全功能配置加载器，整合了以下功能：
    - 从文件路径加载（替代 compose_config_from_path）
    - 从预加载配置创建（替代 normalize_config_to_dictconfig）
    - 使用 Hydra 上下文管理（替代 hydra_config_context）
    - 支持字典输出（替代 load_skill_config）

    Args:
        config_path: 配置文件路径或配置目录路径
        config_name: 配置文件名（不含扩展名），与 config_path 配合使用
        config: 预加载的配置字典或 DictConfig，如果提供则忽略文件加载
        overrides: Hydra 风格的覆盖参数列表，如 ["key=value", "nested.key=value"]
        return_dict: 如果为 True，返回普通字典；否则返回 DictConfig
        resolve: 是否解析配置中的插值引用（如 ${other.key}）
        allow_struct: 是否允许结构化配置（禁止动态添加字段）

    Returns:
        DictConfig 或普通字典，取决于 return_dict 参数

    Raises:
        ValueError: 如果没有提供任何配置源
        ImportError: 如果 Hydra 未安装但需要从文件加载

    Examples:
        # 从文件加载（自动检测是文件还是目录）
        cfg = load_config("/path/to/config.yaml")

        # 从目录 + 文件名加载（Hydra 风格）
        cfg = load_config(config_path="conf", config_name="config")

        # 从预加载配置创建
        cfg = load_config(config={"key": "value"})

        # 返回普通字典（用于技能配置）
        cfg_dict = load_config("conf/config.yaml", return_dict=True)

        # 使用覆盖参数
        cfg = load_config("config.yaml", overrides=["simulator.headless=true"])
    """
    ensure_default_resolvers()
    overrides = list(overrides or [])

    # 场景 1: 使用预加载的配置
    if config is not None:
        if isinstance(config, DictConfig):
            cfg = OmegaConf.create(config)
        else:
            cfg = OmegaConf.create(deepcopy(config))

        if not isinstance(cfg, DictConfig):
            raise TypeError(f"Expected a mapping style configuration, got {type(cfg)}")

        OmegaConf.set_struct(cfg, allow_struct)

    # 场景 2: 从文件加载
    elif config_path is not None:
        try:
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
        except ImportError as exc:
            raise ImportError(
                "Hydra is required to load configuration files. Install hydra-core."
            ) from exc

        config_path = Path(config_path)

        # 如果是相对路径，先不要 resolve，直接检查存在性
        # 如果路径不存在，尝试相对于当前工作目录解析
        if not config_path.exists():
            config_path = config_path.resolve()

        # 如果还是不存在，给出清晰的错误提示
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration path not found: {config_path}\n"
                f"Original input: {config_path}\n"
                f"Current working directory: {Path.cwd()}"
            )

        # 判断是完整文件路径还是目录路径
        if config_path.is_file():
            # 完整文件路径模式
            config_dir = str(config_path.parent.resolve())
            config_name = config_path.stem
        elif config_path.is_dir() and config_name:
            # 目录 + 文件名模式
            config_dir = str(config_path.resolve())
        else:
            raise ValueError(
                f"Invalid config_path: {config_path}. "
                "Provide either a file path or directory path with config_name."
            )

        # 使用 hydra_context_base 确保状态隔离
        with hydra_context_base():
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                cfg = compose(config_name=config_name, overrides=overrides)

        if not isinstance(cfg, DictConfig):
            raise TypeError(f"Hydra compose returned unexpected type: {type(cfg)}")

        OmegaConf.set_struct(cfg, allow_struct)

    else:
        raise ValueError(
            "Must provide either 'config' (preloaded config) or 'config_path' "
            "(file/directory path, optionally with config_name)."
        )

    # 解析插值引用（如 ${monitoring.log_dir}）
    if resolve:
        OmegaConf.resolve(cfg)

    # 返回普通字典或 DictConfig
    if return_dict:
        return OmegaConf.to_container(cfg, resolve=True)
    else:
        return cast(DictConfig, cfg)


def default_config_path() -> Path:
    """返回默认配置路径，便于 CLI 使用。"""
    return Path(__file__).resolve().parents[1] / "conf" / "config.yaml"


if __name__ == "__main__":
    # 测试新的统一 load_config 函数
    print("=== Test 1: Load from file path ===")
    cfg = load_config(
        "/home/ps/Projects/isaac-lab-workspace/IsaacLabLatest/IsaacLab/robot_brain_system/conf/config.yaml"
    )
    print(cfg)
    print()

    print("=== Test 2: Load from directory + name ===")
    cfg = load_config(
        config_path="/home/ps/Projects/isaac-lab-workspace/IsaacLabLatest/IsaacLab/robot_brain_system/conf",
        config_name="config",
    )
    print(cfg)
    print()

    print("=== Test 3: Return as dict ===")
    cfg_dict = load_config(
        config_path="/home/ps/Projects/isaac-lab-workspace/IsaacLabLatest/IsaacLab/robot_brain_system/conf",
        config_name="config",
        return_dict=True,
    )
    print(type(cfg_dict))
    print(cfg_dict["monitoring"])
    print()

    print("=== Test 4: Return as dict from relative path ===")
    # 相对路径需要基于当前文件位置计算
    cfg_dict = load_config(
        config_path=Path(__file__).parent.parent / "conf",
        config_name="config",
        return_dict=True,
    )
    print(type(cfg_dict))
    print(cfg_dict["monitoring"])
    print()

    print("=== Test 5: With overrides ===")
    cfg: DictConfig = load_config(  # type: ignore
        config_path="/home/ps/Projects/isaac-lab-workspace/IsaacLabLatest/IsaacLab/robot_brain_system/conf",
        config_name="config",
        overrides=["simulator.headless=true"],
        return_dict=False,  # 确保返回 DictConfig
    )
    try:
        # 使用字典访问方式更安全
        if "simulator" in cfg and "headless" in cfg.simulator:
            print(f"simulator.headless = {cfg.simulator.headless}")
        else:
            print("No simulator.headless config in this file")
    except Exception as e:
        print(f"Error accessing config: {e}")
