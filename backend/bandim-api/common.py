from typing import Union
from fastapi import HTTPException
import json


def read_locations_from_json(file_path: str):
    """
    Read geospatial locations from a JSON file.

    :param file_path: Path to the JSON file
    :return: List of dicts with 'latitude' and 'longitude' keys
    """
    with open(file_path, "r") as file:
        points = json.load(file)
    return points


def check_api_key(x_api_key: str, api_keys: Union[None, list[str]]) -> None:
    # Check the API key if it is present and thus required
    if api_keys:
        if not x_api_key in api_keys:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized. Missing or invalid API key.",
            )


def load_cors(PATH: Union[None, str] = None) -> dict:
    if PATH is None:
        PATH = "./cors_config.json"
    data = {}
    try:
        with open(PATH) as f:
            data = json.loads(f.read())
        return data
    except FileNotFoundError:
        pass
    return data


def str_to_bool_or_none(s: str) -> Union[None, bool]:
    """Convert a string to a boolean value.
    Args:
        s (str): A string.

    Returns:
        (bool or None): True if the string has boolean value True. False if the string
            has boolean value False. Otherwise None.
    """
    s = s.strip()  # Strip whitespace
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        return None
