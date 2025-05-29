"""
Default configuration for the robot brain system.
"""

# System configuration
SYSTEM_CONFIG = {
    # Simulator configuration
    "simulator": {
        "env_name": "Isaac-Move-Box-Frank-IK-Rel",
        "device": "cuda:0",
        "num_envs": 1,
        "headless": True,
        "enable_cameras": True,
        "disable_fabric": False,
        "livestream": False,
        "sim_device": "cuda:0",
        "cpu": False,
        "physics_gpu": -1,
        "graphics_gpu": -1,
        "pipeline": "gpu",
        "fabric_gpu": -1,
        "verbosity": "info",
    },
    # Brain (Qwen VL) configuration
    "brain": {
        "qwen": {
            "model": "qwen-vl",
            "api_key": "",  # Set your API key here for OpenAI adapter
            "base_url": "",  # Set your API base URL here for OpenAI adapter
            "model_path": "/home/ps/models/model_qwen2.5_vl_7b_instruct",  # Set your local model path for Qwen VL adapter
            "adapter_type": "qwen_vl",  # "qwen_vl", "openai", or "mock"
            "max_tokens": 512,
            "device": "cuda:1"
        },
        "monitoring_interval": 1.0,  # seconds
        "max_retries": 3,
        "task_timeout": 300.0,  # 5 minutes default task timeout
    },
    # Skill system configuration
    "skills": {
        "auto_discover": True,
        "skills_directory": "../skills",
        "default_timeout": 30.0,
        "max_concurrent_skills": 1,
    },
    # System monitoring
    "monitoring": {
        "log_level": "INFO",
        "log_file": "logs/robot_brain_system.log",
        "performance_monitoring": True,
        "skill_execution_logging": True,
    },
    # Safety configuration
    "safety": {
        "emergency_stop_enabled": True,
        "max_task_duration": 600.0,  # 10 minutes
        "position_limits": {
            "x": [-1.0, 1.0],
            "y": [-1.0, 1.0],
            "z": [0.0, 2.0],
        },
        "velocity_limits": {
            "linear": 1.0,  # m/s
            "angular": 3.14,  # rad/s
        },
    },
}


# Development/testing configuration
DEVELOPMENT_CONFIG = {
    **SYSTEM_CONFIG,
    "simulator": {
        **SYSTEM_CONFIG["simulator"],
        "headless": False,  # Show GUI for development
        "num_envs": 1,
        "verbosity": "debug",
        "env_config_file": "configs/ur5_lunar_base.yaml",
    },
    "brain": {
        **SYSTEM_CONFIG["brain"],
        "monitoring_interval": 0.5,  # More frequent monitoring
    },
    "monitoring": {
        **SYSTEM_CONFIG["monitoring"],
        "log_level": "DEBUG",
        "performance_monitoring": True,
    },
}


# Production configuration
PRODUCTION_CONFIG = {
    **SYSTEM_CONFIG,
    "simulator": {
        **SYSTEM_CONFIG["simulator"],
        "headless": True,
        "verbosity": "warning",
    },
    "monitoring": {
        **SYSTEM_CONFIG["monitoring"],
        "log_level": "WARNING",
        "performance_monitoring": False,
    },
}
