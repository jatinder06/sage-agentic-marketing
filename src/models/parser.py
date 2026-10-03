"""Parse structured specialist responses."""
import json


def parse_json_response(response):
    if isinstance(response, dict):
        return response
    try:
        return json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Model response is not valid JSON") from exc