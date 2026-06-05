"""
Schema validation service.

Supports a custom `x-strict` extension on JSON Schema properties:

    {
      "properties": {
        "invoice_number": {
          "type": "string",
          "x-strict": true      ← missing/wrong type → hard error (HTTP 422)
        },
        "notes": {
          "type": "string",
          "x-strict": false     ← missing/wrong type → warning only
        }
      }
    }

Fields without `x-strict` default to strict=True (safe default).
"""

import logging
from typing import Any

import jsonschema
from jsonschema import Draft7Validator

logger = logging.getLogger(__name__)


class FieldViolation:
    def __init__(self, path: str, message: str, strict: bool):
        self.path = path
        self.message = message
        self.strict = strict

    def __repr__(self):
        mode = "STRICT" if self.strict else "LENIENT"
        return f"[{mode}] {self.path}: {self.message}"


def _get_strictness(schema: dict, path_parts: list[str]) -> bool:
    """
    Walk the schema following *path_parts* to find the leaf property
    definition and read its `x-strict` flag.
    Defaults to True when the flag is absent.
    """
    node = schema
    for part in path_parts:
        # try properties first, then items (arrays)
        if "properties" in node and part in node["properties"]:
            node = node["properties"][part]
        elif "items" in node:
            node = node["items"]
            if "properties" in node and part in node["properties"]:
                node = node["properties"][part]
        else:
            break
    return bool(node.get("x-strict", True))


def validate_output(
    data: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """
    Validate *data* against *schema* with per-field strictness.

    Returns:
        (cleaned_data, warnings)   on success (strict errors raise)

    Raises:
        ValueError  with a human-readable message listing all strict violations.
    """
    # Strip x-strict from a copy before passing to jsonschema
    # (jsonschema does not know this keyword and may complain)
    clean_schema = _strip_custom_keywords(schema)

    validator = Draft7Validator(clean_schema)
    errors = list(validator.iter_errors(data))

    if not errors:
        return data, []

    strict_errors: list[str] = []
    warnings: list[str] = []

    for error in errors:
        path_parts = [str(p) for p in error.absolute_path]
        path_str = ".".join(path_parts) if path_parts else error.json_path or "root"
        is_strict = _get_strictness(schema, path_parts)

        msg = f"'{path_str}': {error.message}"

        if is_strict:
            strict_errors.append(msg)
            logger.warning("STRICT violation — %s", msg)
        else:
            warnings.append(msg)
            logger.info("LENIENT violation (warning) — %s", msg)

    if strict_errors:
        bullet_list = "\n  • ".join(strict_errors)
        raise ValueError(
            f"Schema validation failed on {len(strict_errors)} strict field(s):\n"
            f"  • {bullet_list}"
        )

    return data, warnings


def _strip_custom_keywords(schema: dict) -> dict:
    """Recursively remove `x-*` keys so jsonschema doesn't warn about them."""
    if not isinstance(schema, dict):
        return schema
    return {
        k: _strip_custom_keywords(v)
        for k, v in schema.items()
        if not k.startswith("x-")
    }
