import json
from typing import Dict, Any


def readConfig() -> Dict[str, Any]:
    """
    Reads the config file and returns a dictionary.
    """
    with open('config.json', encoding='utf-8') as f:
        return json.load(f)
