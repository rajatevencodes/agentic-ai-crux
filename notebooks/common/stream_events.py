"""Rebuild output items from decoded streaming events.

Both streaming.py and conversation.py use the same event rules here.
"""

import copy
from typing import Any

from .parsing import _object_list, _parse_response

# Event families that carry incremental text. Final values replace deltas.
# Reference: https://developers.openai.com/api/reference/resources/responses/streaming-events
_STREAM_TEXT_FIELDS = {
    "response.output_text": ("message", "content", "content_index", "output_text", "text"),
    "response.refusal": ("message", "content", "content_index", "refusal", "refusal"),
    "response.reasoning_text": ("reasoning", "content", "content_index", "reasoning_text", "text"),
    "response.reasoning_summary_text": ("reasoning", "summary", "summary_index", "summary_text", "text"),
}
_STREAM_ARGUMENT_FIELDS = {
    "response.function_call_arguments": ("function_call", "arguments"),
    "response.custom_tool_call_input": ("custom_tool_call", "input"),
    "response.mcp_call_arguments": ("mcp_call", "arguments"),
    "response.code_interpreter_call_code": ("code_interpreter_call", "code"),
}
_TERMINAL_EVENTS = {"response.completed", "response.failed", "response.incomplete"}
_LIFECYCLE_EVENTS = _TERMINAL_EVENTS | {"response.created", "response.in_progress", "response.queued"}
_STREAM_TOOL_PHASES = {
    "response.file_search_call": {"in_progress", "searching", "completed"},
    "response.web_search_call": {"in_progress", "searching", "completed"},
    "response.code_interpreter_call": {"in_progress", "interpreting", "completed"},
    "response.image_generation_call": {"in_progress", "generating", "completed"},
    "response.mcp_call": {"in_progress", "completed", "failed"},
    "response.mcp_list_tools": {"in_progress", "completed", "failed"},
}
_STREAM_ITEM_EVENTS = {
    "response.output_item.added", "response.output_item.done",
    "response.content_part.added", "response.content_part.done",
    "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
    "response.output_text.annotation.added", "response.image_generation_call.partial_image",
    "response.shell_call_command.added", "response.shell_call_command.delta", "response.shell_call_command.done",
    "response.shell_call_output_content.delta", "response.shell_call_output_content.done",
} | {
    f"{family}.{phase}" for family in _STREAM_TEXT_FIELDS | _STREAM_ARGUMENT_FIELDS
    for phase in ("delta", "done")
} | {f"{family}.{phase}" for family, phases in _STREAM_TOOL_PHASES.items() for phase in phases}
_STREAM_AUDIO_EVENTS = {
    "response.audio.delta", "response.audio.done",
    "response.audio.transcript.delta", "response.audio.transcript.done",
}


def _event_index(event: dict[str, Any], key: str) -> int:
    index = event.get(key)
    if type(index) is not int or index < 0:
        raise ValueError(f'{event.get("type")}: {key} must be a nonnegative integer.')
    return index


def _stream_part(item: dict[str, Any], field: str, index: int) -> dict[str, Any]:
    parts = item.get(field)
    if parts is None:
        parts = []
        item[field] = parts
    elif not isinstance(parts, list):
        raise TypeError(f"Stream item field {field!r} must be a list or null.")
    while len(parts) <= index:
        parts.append({})
    return parts[index]


def _apply_stream_event(
    items: dict[int, dict[str, Any]], event: dict[str, Any],
    shell_outputs: dict[int, dict[int, list[dict[str, Any]]]],
) -> None:
    """Commit one valid item update; a malformed event cannot erase the prefix.

    Example event, applied to initially empty items and shell_outputs dicts:
    ```json
    {"type": "response.output_text.delta", "output_index": 0,
     "content_index": 0, "delta": "Hello"}
    ```
    items[0] becomes the object below. This function mutates items and returns None.
    ```json
    {"type": "message", "content": [
      {"type": "output_text", "annotations": [], "text": "Hello"}
    ]}
    ```
    A later delta appends text. A done event supplies the final value.
    """
    kind = event["type"]
    index = _event_index(event, "output_index")
    item = copy.deepcopy(items.get(index, {}))
    if event.get("item_id"):
        if item.get("id") and item["id"] != event["item_id"]:
            raise ValueError(f"{kind}: item_id changed at output[{index}].")
        item["id"] = event["item_id"]
    if kind in {"response.output_item.added", "response.output_item.done"}:
        if not isinstance(event.get("item"), dict):
            raise TypeError(f"{kind}: item must be an object.")
        _parse_response({"object": "response", "output": [event["item"]]})
        if item.get("id") and item["id"] != event["item"].get("id"):
            raise ValueError(f"{kind}: item ID changed at output[{index}].")
        items[index] = copy.deepcopy(event["item"])
        return
    if kind in {"response.content_part.added", "response.content_part.done",
                "response.reasoning_summary_part.added", "response.reasoning_summary_part.done"}:
        summary = "reasoning_summary_part" in kind
        field, index_key = ("summary", "summary_index") if summary else ("content", "content_index")
        if not isinstance(event.get("part"), dict):
            raise TypeError(f"{kind}: part must be an object.")
        part = _stream_part(item, field, _event_index(event, index_key))
        part.clear()
        part.update(copy.deepcopy(event["part"]))
        item.setdefault("type", "reasoning" if summary or part.get("type") == "reasoning_text" else "message")
        _parse_response({"object": "response", "output": [item]})
        items[index] = item
        return
    family, _, phase = kind.rpartition(".")
    if family in _STREAM_TEXT_FIELDS:
        item_type, field, index_key, part_type, text_key = _STREAM_TEXT_FIELDS[family]
        item.setdefault("type", item_type)
        part = _stream_part(item, field, _event_index(event, index_key))
        part.setdefault("type", part_type)
        if part_type == "output_text":
            part.setdefault("annotations", [])
        value = event.get("delta" if phase == "delta" else text_key)
        if not isinstance(value, str):
            raise TypeError(f"{kind}: expected a text value.")
        part[text_key] = part.get(text_key, "") + value if phase == "delta" else value
        if part_type == "output_text" and event.get("logprobs") is not None:
            logprobs = _object_list(event["logprobs"], f"{kind}.logprobs")
            part["logprobs"] = (part.get("logprobs") or []) + logprobs if phase == "delta" else logprobs
    elif family in _STREAM_ARGUMENT_FIELDS:
        item_type, field = _STREAM_ARGUMENT_FIELDS[family]
        item.setdefault("type", item_type)
        value = event.get("delta" if phase == "delta" else field)
        if not isinstance(value, str):
            raise TypeError(f"{kind}: expected a text value.")
        previous = item.get(field)
        item[field] = ("" if previous is None else previous) + value if phase == "delta" else value
        if event.get("name"):
            item["name"] = event["name"]
    elif kind == "response.output_text.annotation.added":
        annotation = event.get("annotation")
        if annotation is None:  # The reference explicitly permits null.
            return
        if not isinstance(annotation, dict):
            raise TypeError(f"{kind}: annotation must be an object or null.")
        item.setdefault("type", "message")
        part = _stream_part(item, "content", _event_index(event, "content_index"))
        part.setdefault("type", "output_text")
        part.setdefault("text", "")
        annotation_index = _event_index(event, "annotation_index")
        annotations = part.setdefault("annotations", [])
        while len(annotations) <= annotation_index:
            annotations.append({})
        annotations[annotation_index] = copy.deepcopy(event["annotation"])
    elif phase in _STREAM_TOOL_PHASES.get(family, set()):
        item.setdefault("type", family.removeprefix("response."))
        item["status"] = phase
    elif kind == "response.image_generation_call.partial_image":
        _event_index(event, "partial_image_index")
        if not isinstance(event.get("partial_image_b64"), str):
            raise TypeError(f"{kind}: partial_image_b64 must be text.")
        item.setdefault("type", "image_generation_call")
    elif family == "response.shell_call_command":
        item.setdefault("type", "shell_call")
        commands = item.setdefault("action", {}).setdefault("commands", [])
        command_index = _event_index(event, "command_index")
        value = event.get("delta" if phase == "delta" else "command")
        if not isinstance(value, str):
            raise TypeError(f"{kind}: command must be text.")
        while len(commands) <= command_index:
            commands.append("")
        commands[command_index] = commands[command_index] + value if phase == "delta" else value
    elif family == "response.shell_call_output_content":
        item.setdefault("type", "shell_call_output")
        command_index = _event_index(event, "command_index")
        groups = copy.deepcopy(shell_outputs.get(index, {}))
        if phase == "done":
            groups[command_index] = _object_list(event["output"], f"{kind}.output")
        else:
            delta = event.get("delta")
            if not isinstance(delta, dict):
                raise TypeError(f"{kind}: delta must be an object.")
            output = groups.setdefault(command_index, [{"stdout": "", "stderr": ""}])
            for field in ("stdout", "stderr"):
                if field in delta:
                    if not isinstance(delta[field], str):
                        raise TypeError(f"{kind}: {field} must be text.")
                    output[-1][field] += delta[field]
        item["output"] = [chunk for key in sorted(groups) for chunk in groups[key]]
        shell_outputs[index] = groups
    items[index] = item
