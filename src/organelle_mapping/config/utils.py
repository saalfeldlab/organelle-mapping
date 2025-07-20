from pydantic import TypeAdapter
from pathlib import Path
import yaml 

def load_subconfig(value, target_cls):
    if isinstance(value, str):
        config = open(Path(value))
        return TypeAdapter(target_cls).validate_python(yaml.safe_load(config))

    return value