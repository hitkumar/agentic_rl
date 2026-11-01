"""Tools shared by every domain: the @tool marker, the schemas the agent sees, and running a tool call.

A domain's toolkit is a class whose tool methods are marked @tool and carry a Google-style docstring. The schema for
each tool is built from the method itself, as tau2-bench v1.0.1 does (src/tau2/environment/tool.py): the name from the
method, the description from the docstring's summary and body, and the parameters from the signature's type hints and
the docstring's Args section, as a pydantic model's JSON schema.
"""

import inspect
import json
from typing import Any, Callable

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model

TOOL_ATTR = "__is_tool__"


def tool(func: Callable) -> Callable:
    """Mark a toolkit method as a tool the agent can call."""
    setattr(func, TOOL_ATTR, True)
    return func


def tool_methods(toolkit: object) -> dict[str, Callable]:
    """The toolkit's tools as bound methods, in the order they are defined in the class."""
    return {
        name: getattr(toolkit, name) for name, attr in vars(type(toolkit)).items() if getattr(attr, TOOL_ATTR, False)
    }


def tool_schema(method: Callable) -> dict:
    """The OpenAI function-calling schema for one tool method."""
    doc = parse(method.__doc__ or "")
    arg_docs = {p.arg_name: p.description for p in doc.params}
    params = {}
    for name, param in inspect.signature(method).parameters.items():
        default = ... if param.default is param.empty else param.default
        if name in arg_docs:
            default = Field(default, description=arg_docs[name])
        params[name] = (Any if param.annotation is param.empty else param.annotation, default)
    description = doc.short_description or method.__name__
    if doc.long_description:
        description += "\n\n" + doc.long_description
    return {
        "type": "function",
        "function": {
            "name": method.__name__,
            "description": description,
            "parameters": create_model("parameters", **params).model_json_schema(),
        },
    }


def tool_schemas(toolkit: object) -> list[dict]:
    return [tool_schema(method) for method in tool_methods(toolkit).values()]


def to_json_str(result: Any) -> str:
    """Serialize a tool's return value the way tau2 does: models via model_dump, strings as they are."""

    def process(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, str) or value is None:
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [process(item) for item in value]
        if isinstance(value, dict):
            return {k: process(v) for k, v in value.items()}
        raise ValueError(f"Unsupported type: {type(value)}")

    if isinstance(result, str):
        return result
    return json.dumps(process(result), default=str)


def call_tool(toolkit: object, name: str, arguments: dict) -> tuple[str, bool]:
    """Run one tool call. Returns (content the agent sees, whether it was an error): any exception the tool raises
    becomes "Error: <message>"."""
    try:
        methods = tool_methods(toolkit)
        if name not in methods:
            raise ValueError(f"Tool '{name}' not found.")
        return to_json_str(methods[name](**arguments)), False
    except Exception as e:
        return f"Error: {e}", True
