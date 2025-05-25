"""
Example configurations for different model adapters.
"""

# Example configuration for Qwen VL local model
QWEN_VL_CONFIG = {
    "brain": {
        "qwen": {
            "adapter_type": "qwen_vl",
            "model_path": "/path/to/qwen2.5-vl-7b-instruct",  # Update with your model path
            "max_tokens": 512,
        },
        "monitoring_interval": 1.0,
        "max_retries": 3,
        "task_timeout": 300.0,
    },
    # ... other config sections
}

# Example configuration for OpenAI API
OPENAI_CONFIG = {
    "brain": {
        "qwen": {
            "adapter_type": "openai",
            "api_key": "sk-your-openai-api-key-here",  # Replace with your API key
            "base_url": "https://api.openai.com/v1",  # Optional: custom base URL
            "model": "gpt-4o",  # or "gpt-4-vision-preview", "gpt-3.5-turbo", etc.
            "max_tokens": 512,
        },
        "monitoring_interval": 1.0,
        "max_retries": 3,
        "task_timeout": 300.0,
    },
    # ... other config sections
}

# Example configuration for custom OpenAI-compatible API (e.g., local LLM server)
CUSTOM_API_CONFIG = {
    "brain": {
        "qwen": {
            "adapter_type": "openai",
            "api_key": "dummy-key",  # Some servers require any non-empty key
            "base_url": "http://localhost:8000/v1",  # Your local server URL
            "model": "qwen2.5-vl-7b-instruct",  # Model name on your server
            "max_tokens": 512,
        },
        "monitoring_interval": 1.0,
        "max_retries": 3,
        "task_timeout": 300.0,
    },
    # ... other config sections
}

# Mock configuration for testing/development
MOCK_CONFIG = {
    "brain": {
        "qwen": {
            "adapter_type": "mock",
            "max_tokens": 512,
        },
        "monitoring_interval": 0.5,  # More frequent for testing
        "max_retries": 3,
        "task_timeout": 300.0,
    },
    # ... other config sections
}

# Configuration with fallback strategy
FALLBACK_CONFIG = {
    "brain": {
        "qwen": {
            "adapter_type": "openai",  # Try OpenAI first
            "api_key": "sk-your-api-key",
            "model": "gpt-4o",
            "max_tokens": 512,
            # Fallback configuration
            "fallback_adapter_type": "mock",  # Fall back to mock if OpenAI fails
        },
        "monitoring_interval": 1.0,
        "max_retries": 3,
        "task_timeout": 300.0,
    },
    # ... other config sections
}
