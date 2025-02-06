from enum import Enum
from typing import Optional

class AIModel(str, Enum):
    """Supported AI models for research tasks"""
    O3_MINI = "o3-mini"
    GPT4O = "gpt-4o"

    @classmethod
    def get_default(cls) -> str:
        """Returns the default model if an invalid one is provided"""
        return cls.O3_MINI.value

    @classmethod
    def is_valid(cls, model: str) -> bool:
        """Check if the provided model is valid"""
        return model in [item.value for item in cls]

def validate_model(model: Optional[str] = None) -> str:
    """
    Validates the provided model and returns either the validated model
    or the default model if the provided one is invalid.

    Args:
        model: The model string to validate

    Returns:
        str: A valid model string

    Raises:
        ValueError: If strict validation is required instead of falling back to default
    """
    if not model or not AIModel.is_valid(model):
        # Option 1: Raise error
        # raise ValueError(f"Invalid model: {model}. Must be one of: {[m.value for m in AIModel]}")

        # Option 2: Fall back to default
        return AIModel.get_default()

    return model