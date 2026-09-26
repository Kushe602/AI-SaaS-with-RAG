"""Shared Jinja2 template environment."""
import json
from pathlib import Path

from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _fromjson(value: str):
    try:
        return json.loads(value or "[]")
    except (ValueError, TypeError):
        return []


templates.env.filters["fromjson"] = _fromjson
