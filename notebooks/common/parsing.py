"""Turn SDK objects into plain data, then group response items into a report.

These helpers do not print, consume streams, call models, or execute tools.
"""

import json
import math
from collections.abc import Mapping
from typing import Any


def _plain_data(value: Any) -> Any:
    """Normalize SDK models without modifying caller-owned values."""
    if callable(getattr(value, "model_dump", None)):
        return _plain_data(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {key: _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    return value


def _object_list(value: Any, path: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"{path} must be a list of objects.")
    return value


def _request_input_data(request_input: Any) -> Any:
    if request_input is not None and not isinstance(request_input, (str, list, tuple)):
        raise TypeError("request_input must be text or a list/tuple of input items.")
    data = _plain_data(request_input)
    if isinstance(data, list):
        _object_list(data, "request_input")
    return data


def _parse_response(response: Any, *, request_input: Any = None) -> dict[str, Any]:
    r"""Group response output by type without changing the original payload.

    Example function-call item inside response["output"]:
    ```json
    {"type": "function_call", "call_id": "call_demo", "name": "get_weather",
     "arguments": "{\"city\":\"Paris\"}"}
    ```
    The corresponding report["function_calls"][0] also has:
    ```json
    {"parsed_arguments": {"city": "Paris"}, "arguments_error": null}
    ```
    The raw arguments string stays present. Parsing does not execute the call.
    """
    data = _plain_data(response)
    if not isinstance(data, dict) or data.get("object") != "response":
        raise TypeError("Expected a Responses API object with object='response'; "
                        "use inspect_response_stream for streaming events.")
    items = _object_list(data.get("output"), "response.output")
    input_data = _request_input_data(request_input)
    report = dict(
        response=data, input=input_data, output_text="", messages=[], reasoning=[],
        function_calls=[], tool_calls=[], refusals=[], citations=[],
        other_items=[], usage=data.get("usage"), duration_seconds=None, issues=[],
        stream=None,
    )
    texts = []
    for index, item in enumerate(items):
        kind = item.get("type")
        if kind == "message":
            report["messages"].append(item)
            for content_index, part in enumerate(_object_list(item.get("content"), f"output[{index}].content")):
                if part.get("type") == "output_text":
                    if not isinstance(part.get("text"), str):
                        raise TypeError(f"output[{index}].content[{content_index}].text must be text.")
                    texts.append(part["text"])
                    for annotation in _object_list(part.get("annotations"), "annotations"):
                        report["citations"].append(dict(
                            output_index=index, content_index=content_index, annotation=annotation,
                        ))
                elif part.get("type") == "refusal":
                    report["refusals"].append(part.get("refusal", ""))
        elif kind == "reasoning":
            _object_list(item.get("summary"), f"output[{index}].summary")
            _object_list(item.get("content"), f"output[{index}].content")
            report["reasoning"].append(item)
        elif kind == "function_call":
            call = dict(item, parsed_arguments=None, arguments_error=None)
            try:
                call["parsed_arguments"] = json.loads(
                    item["arguments"], parse_constant=_reject_json_constant,
                )
            except (KeyError, TypeError, ValueError) as error:
                call["arguments_error"] = str(error)
                report["issues"].append(f"Function {item.get('name', '(unnamed)')}: invalid JSON arguments ({error}).")
            report["function_calls"].append(call)
            report["tool_calls"].append(item)
        elif kind in {"file_search_call", "web_search_call", "function_call_output", "mcp_list_tools"} or (
            isinstance(kind, str) and (kind.endswith("_call") or kind.endswith("_call_output"))
        ):
            report["tool_calls"].append(item)
        else:
            report["other_items"].append(item)
    # Match the SDK output_text property: concatenate only output_text parts.
    report["output_text"] = "".join(texts)
    created, completed = data.get("created_at"), data.get("completed_at")
    if created is not None and completed is not None:
        try:
            duration = float(completed) - float(created)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("timestamps must be finite and ordered")
            report["duration_seconds"] = duration
        except (ValueError, TypeError, OverflowError):
            report["issues"].append("Server duration unavailable: invalid response timestamps.")
    return report


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not valid JSON")
