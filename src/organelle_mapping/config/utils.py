from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationInfo


def get_base_dir(info: ValidationInfo) -> Path:
    """Get base_dir from validation context, defaulting to cwd."""
    if info.context and "base_dir" in info.context:
        return Path(info.context["base_dir"])
    return Path.cwd()


def load_subconfig(value, target_cls, info: ValidationInfo):
    """Load a subconfig from a file path or return as-is.

    Args:
        value: Either a file path (str) to load, or an already-parsed config object
        target_cls: The Pydantic model class to validate against
        info: ValidationInfo containing context with optional 'base_dir' for relative paths

    Returns:
        Validated config object of type target_cls
    """
    if isinstance(value, str):
        config_path = Path(value)

        # Resolve path relative to base_dir if provided
        if not config_path.is_absolute():
            config_path = get_base_dir(info) / config_path

        with open(config_path) as config:
            # Update base_dir to the loaded config's directory for nested relative paths
            new_context = {**(info.context or {}), "base_dir": str(config_path.parent)}
            return TypeAdapter(target_cls).validate_python(yaml.safe_load(config), context=new_context)

    return value


def resolve_path(value: str, info: ValidationInfo) -> str:
    """Resolve a path relative to base_dir from context.

    Args:
        value: Path string to resolve
        info: ValidationInfo containing context with optional 'base_dir'

    Returns:
        Resolved absolute path as string
    """
    path = Path(value)
    if not path.is_absolute():
        path = get_base_dir(info) / path
    return str(path)
