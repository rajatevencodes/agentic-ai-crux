"""Notebook setup and configurable Responses API inspectors."""

import copy
import json
import logging
import math
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_QUIET_LOGGERS = ("", "httpx", "openai")


def configure_notebook() -> Path:
    """Load the project environment and keep notebook output focused."""
    warnings.filterwarnings("ignore")

    for logger_name in _QUIET_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"No .env file found at {env_path}")

    load_dotenv(env_path)
    return env_path


RESPONSES_API_REFERENCE = (
    "https://developers.openai.com/api/reference/python/resources/responses/methods/create"
)
RESPONSES_STREAMING_REFERENCE = (
    "https://developers.openai.com/api/reference/resources/responses/streaming-events"
)


def inspect_response(response: Any, *, request_input: Any = None) -> dict[str, Any]:
    """Parse an SDK Response or response dict and print a configurable report.

    Edit display below; True means print. Hidden fields remain in the return value.
    Pass the exact input used for responses.create as request_input to inspect it.
    No API requests or tool execution happen here. Use inspect_response_stream for
    responses.create(stream=True). Reference: RESPONSES_API_REFERENCE.
    """
    # EDIT THESE SWITCHES. Add any response field (e.g. service_tier=True).
    display = dict(
        input=True, answer=True, reasoning=True, duration=True,
        tools=True, citations=True, usage=True, errors=True,
        input_tokens=True, cached_tokens=True, output_tokens=True,
        reasoning_tokens=True, total_tokens=True, cost=True, cost_details=False,
        usage_details=False,
        status=True, model=True,
        response_id=False, message_id=False, tool_id=False, call_id=False,
        role=False, item_status=False, tool_status=True, created_at=False, completed_at=False,
        top_p=False, temperature=False, instructions=False,
        reasoning_settings=False, tool_choice=False, tool_definitions=False,
        parallel_tool_calls=False, max_output_tokens=False, max_tool_calls=False,
        service_tier=False, previous_response_id=False, metadata=False,
        background=False, store=False, truncation=False, text=False,
        other_items=True, other_fields=False, raw_response=False, binary_payloads=False,
    )
    report = _parse_response(response, request_input=request_input)
    _render_response(report, display)
    return report


def inspect_response_stream(events: Any, *, request_input: Any = None) -> dict[str, Any]:
    """Consume one synchronous SDK event stream, print live text, return a report.

    Accepts responses.create(stream=True), an entered responses.stream() context,
    or an iterable of decoded SDK events/dicts. Owns and closes the stream.
    All received events are retained in report["stream"]["events"].
    Audio chunks/transcripts, image previews and shell outputs are also indexed
    under stream["audio"], stream["partial_images"] and stream["shell_outputs"].
    Unknown event types are retained and listed in stream["unknown_event_types"].
    A missing terminal event is reported as interrupted, never as completed.
    An API error event has stream status "error"; tool failures are item-local.
    Transport exceptions propagate as ResponseStreamError with a partial report.
    Async iterators and raw SSE bytes are intentionally not accepted.
    Reference: RESPONSES_STREAMING_REFERENCE.
    """
    # EDIT THESE SWITCHES independently of inspect_response's display settings.
    display = dict(
        input=True, answer=True, reasoning=True, duration=True,
        tools=True, citations=True, usage=True, errors=True,
        input_tokens=True, cached_tokens=True, output_tokens=True,
        reasoning_tokens=True, total_tokens=True, cost=True, cost_details=False,
        usage_details=False,
        status=True, model=True,
        response_id=False, message_id=False, tool_id=False, call_id=False,
        role=False, item_status=False, tool_status=True, created_at=False, completed_at=False,
        top_p=False, temperature=False, instructions=False,
        reasoning_settings=False, tool_choice=False, tool_definitions=False,
        parallel_tool_calls=False, max_output_tokens=False, max_tool_calls=False,
        service_tier=False, previous_response_id=False, metadata=False,
        background=False, store=False, truncation=False, text=False,
        other_items=True, other_fields=False, raw_response=False, binary_payloads=False,
    )
    live_text = True
    live_tool_progress = True
    event_details = False  # Full event payloads; IDs follow the switches above.
    report = _consume_response_stream(
        events, request_input=request_input, display=display,
        live_text=live_text, live_tool_progress=live_tool_progress,
        event_details=event_details,
    )
    _render_response(report, display)
    return report


class ResponseStreamError(RuntimeError):
    """Stream reading failed. The successfully parsed prefix is in .report."""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


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


def _section(title: str) -> None:
    print(f"\n{title}\n{'─' * 72}", flush=True)


def _field(label: str, value: Any) -> None:
    if value is None:
        value = "not provided"
    elif isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    elif isinstance(value, bool):
        value = "true" if value else "false"
    lines = str(value).splitlines() or [""]
    print(f"  {label:<25} {lines[0]}", flush=True)
    for line in lines[1:]:
        print(f"  {'':25} {line}", flush=True)


def _display_data(value: Any, display: dict[str, bool], *, item_type: str = "") -> Any:
    """Filter the printed view only; retain opaque payloads in structured data."""
    if isinstance(value, dict):
        kind = value.get("type", item_type)
        result = {}
        for key, child in value.items():
            switch = {"role": "role", "call_id": "call_id", "item_id": "tool_id"}.get(key)
            if key == "id":
                switch = "response_id" if value.get("object") == "response" else (
                    "message_id" if kind == "message" else "tool_id"
                )
            if key == "status" and kind:
                switch = "item_status" if kind in {"message", "reasoning"} else "tool_status"
            if switch and not display.get(switch, False):
                continue
            if key in {"encrypted_content", "file_data", "partial_image_b64"} or (
                kind == "image_generation_call" and key == "result"
            ) or (
                kind == "response.audio.delta" and key == "delta"
            ):
                if child and not display.get("binary_payloads", False):
                    result[key] = f"[opaque payload: {len(str(child))} characters]"
                    continue
            result[key] = _display_data(child, display)
        return result
    if isinstance(value, list):
        return [_display_data(item, display) for item in value]
    if isinstance(value, str) and value.startswith("data:") and not display.get("binary_payloads", False):
        return f"[{value.split(',', 1)[0]}; payload hidden]"
    return value


def _render_response(report: dict[str, Any], display: dict[str, bool]) -> None:
    data = report["response"]
    print(f"\n{'═' * 72}\n{'RESPONSES API · INSPECTION':^72}\n{'═' * 72}")
    _section("STATE")
    for key in ("status", "model"):
        if display.get(key):
            _field(key, data.get(key))
    if display.get("status") and report["stream"]:
        _field("stream outcome", report["stream"]["status"])
    if display.get("duration"):
        seconds = report["duration_seconds"]
        _field("server duration", f"{seconds:.3f} s" if seconds is not None else "not provided")
        if report["stream"]:
            _field("stream observation", f'{report["stream"]["duration_seconds"]:.3f} s (local consumption)')
            observations = report["stream"]["reasoning_seconds"]
            for item_label, elapsed in observations.items():
                _field("reasoning observation", f"{item_label}: {elapsed:.3f} s (added → done, local)")
    if display.get("input"):
        _section("INPUT")
        _field("request input", _display_data(report["input"], display) if report["input"] is not None
               else "not supplied; pass request_input=the_input_used_for_create")
    if display.get("answer"):
        _section("OUTPUT · ASSISTANT")
        print(report["output_text"] or "(No assistant text returned.)")
        for message in report["messages"]:
            if display.get("message_id"):
                _field("message id", message.get("id"))
            if display.get("role"):
                _field("role", message.get("role"))
            if display.get("item_status"):
                _field("message status", message.get("status"))
        for refusal in report["refusals"]:
            _field("refusal", refusal)
    if display.get("reasoning") and report["reasoning"]:
        _section("INTERMEDIATE RESULTS · REASONING")
        for item in report["reasoning"]:
            if display.get("tool_id"):
                _field("reasoning id", item.get("id"))
            if display.get("item_status"):
                _field("status", item.get("status"))
            for name, title in (("summary", "summary"), ("content", "returned reasoning")):
                for part in item.get(name) or []:
                    _field(title, part.get("text"))
            if not item.get("summary") and not item.get("content"):
                _field("reasoning", "No readable reasoning supplied by the API.")
            if item.get("encrypted_content"):
                _field("encrypted reasoning", _display_data(item, display)["encrypted_content"])
    if display.get("tools") and report["tool_calls"]:
        _section("INTERMEDIATE RESULTS · TOOLS")
        function_calls = iter(report["function_calls"])
        for index, item in enumerate(report["tool_calls"], 1):
            kind = item["type"]
            print(f"\n  [{index}] {kind}")
            visible = _display_data(item, display)
            if kind == "function_call":
                call = next(function_calls)
                visible["arguments"] = call["parsed_arguments"] if not call["arguments_error"] else item.get("arguments")
                visible["execution"] = "Requested by the model; this inspector does not execute functions."
            elif kind == "file_search_call" and item.get("results") is None:
                visible["results"] = 'Not included. Request include=["file_search_call.results"].'
            elif kind == "web_search_call" and (item.get("action") or {}).get("type") == "search":
                if (item.get("action") or {}).get("sources") is None:
                    visible["sources"] = 'Not included. Request include=["web_search_call.action.sources"].'
            for key, value in visible.items():
                if key != "type" and value is not None:
                    _field(key.replace("_", " "), value)
    if display.get("citations") and report["citations"]:
        _section("OUTPUT · CITATIONS")
        for citation in report["citations"]:
            _field(f'output[{citation["output_index"]}] part[{citation["content_index"]}]',
                   _display_data(citation["annotation"], display))
    if report["stream"] and display.get("other_items"):
        audio = report["stream"]["audio"]
        if audio["chunks"] or audio["transcript"] or audio["done"] or audio["transcript_done"]:
            _section("OUTPUT · AUDIO")
            _field("audio chunks", len(audio["chunks"]))
            _field("audio done", audio["done"])
            _field("transcript", audio["transcript"])
            _field("transcript done", audio["transcript_done"])
            if display.get("binary_payloads"):
                _field("base64 chunks", audio["chunks"])
        if report["stream"]["partial_images"]:
            _section("INTERMEDIATE RESULTS · IMAGE PREVIEWS")
            for preview in report["stream"]["partial_images"]:
                _field("preview", _display_data(preview, display))
    if display.get("usage"):
        _section("STATE · USAGE")
        usage = report["usage"] or {}
        values = {
            "input_tokens": usage.get("input_tokens"),
            "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost": usage.get("cost"),
            "cost_details": usage.get("cost_details"),
        }
        for key, value in values.items():
            if display.get(key) and (key not in {"cost", "cost_details"} or value is not None):
                _field(key.replace("_", " "), value)
        if display.get("usage_details"):
            _field("complete usage", usage)
    if display.get("errors") and (data.get("error") or data.get("incomplete_details") or report["issues"]):
        _section("STATE · ERRORS / INCOMPLETE")
        for key in ("error", "incomplete_details"):
            if data.get(key):
                _field(key.replace("_", " "), data[key])
        for issue in report["issues"]:
            _field("notice", issue)
    aliases = {"id": "response_id", "reasoning": "reasoning_settings", "tools": "tool_definitions"}
    handled = {"status", "model", "output", "usage", "error", "incomplete_details"}
    settings = {key: _display_data(value, display) for key, value in data.items()
                if key not in handled and display.get(aliases.get(key, key), False)}
    if settings:
        _section("STATE · RESPONSE FIELDS")
        for key, value in settings.items():
            _field(key.replace("_", " "), value)
    if display.get("other_fields"):
        _section("INTERMEDIATE RESULTS · ALL ITEMS / EXTRA FIELDS")
        _field("output items", _display_data(data.get("output"), display))
        _field("response fields", _display_data({k: v for k, v in data.items() if k != "output"}, display))
    elif display.get("other_items") and report["other_items"]:
        _section("INTERMEDIATE RESULTS · ADDITIONAL ITEM TYPES")
        for item in report["other_items"]:
            _field(str(item.get("type", "unknown")), _display_data(item, display))
    if display.get("raw_response"):
        _section("RAW RESPONSE · UNFILTERED")
        _field("response", data)

# Event families that carry incremental text. Final values replace deltas.
# Source of truth: RESPONSES_STREAMING_REFERENCE.
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
    """Commit one valid item update; a malformed event cannot erase the prefix."""
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


def _consume_response_stream(
    events: Any, *, request_input: Any, display: dict[str, bool],
    live_text: bool, live_tool_progress: bool, event_details: bool,
) -> dict[str, Any]:
    if isinstance(events, (Mapping, str, bytes)) or hasattr(events, "model_dump") or hasattr(events, "__aiter__"):
        raise TypeError("Expected a synchronous iterable of decoded Responses events.")
    request_input = _request_input_data(request_input)
    if display.get("input"):
        _section("INPUT · STREAM REQUEST")
        _field("request input", _display_data(request_input, display) if request_input is not None
               else "not supplied; pass request_input=the_input_used_for_create")
        display["input"] = False
    started = time.perf_counter()
    items: dict[int, dict[str, Any]] = {}
    response: dict[str, Any] = {"object": "response", "status": None, "output": []}
    received_events = []
    notices = []
    reasoning_started = {}
    reasoning_seconds = {}
    terminal_event = None
    last_sequence = None
    seen_sequences = set()
    stream_error = None
    active_lane = None
    printed_text: dict[tuple, str] = {}
    audio = dict(chunks=[], transcript="", done=False, transcript_done=False)
    partial_images = []
    shell_outputs = {}
    unknown_event_types = []
    api_error = False

    def finish_line() -> None:
        nonlocal active_lane
        if active_lane is not None:
            print(flush=True)
            active_lane = None

    try:
        for raw_event in events:
            event = _plain_data(raw_event)
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise TypeError("Each decoded event must be an object with a string type.")
            received_events.append(event)
            kind = event["type"]
            sequence = event.get("sequence_number")
            if sequence is not None:
                if type(sequence) is not int or sequence < 0:
                    raise ValueError("sequence_number must be a nonnegative integer.")
                if sequence in seen_sequences:
                    previous = next(e for e in received_events[:-1] if e.get("sequence_number") == sequence)
                    if previous != event:
                        raise ValueError(f"Conflicting events share sequence_number {sequence}.")
                    notices.append(f"Repeated event sequence {sequence} ignored during accumulation.")
                    continue
                if last_sequence is not None and sequence < last_sequence:
                    raise ValueError("Out-of-order stream sequence; cannot reliably accumulate deltas.")
                if last_sequence is not None and sequence > last_sequence + 1:
                    notices.append(f"Stream sequence gap: {last_sequence} → {sequence}.")
                seen_sequences.add(sequence)
                last_sequence = sequence
            if event.get("response_id") and response.get("id") and event["response_id"] != response["id"]:
                raise ValueError("Expected events for one response; received a different response ID.")
            if kind in _LIFECYCLE_EVENTS:
                snapshot = event.get("response")
                _parse_response(snapshot)
                if response.get("id") and snapshot.get("id") != response["id"]:
                    raise ValueError("Expected events for one response; received a different response ID.")
                if kind != "response.created" and snapshot.get("status") != kind.removeprefix("response."):
                    raise ValueError(f"{kind}: response status does not match the lifecycle event.")
                response = copy.deepcopy(snapshot)
                for output_index, output_item in enumerate(snapshot.get("output") or []):
                    items[output_index] = copy.deepcopy(output_item)
                if kind in _TERMINAL_EVENTS:
                    terminal_event = kind
            elif kind in _STREAM_ITEM_EVENTS:
                _apply_stream_event(items, event, shell_outputs)
                if kind == "response.image_generation_call.partial_image":
                    partial_images.append(event)
                if kind == "response.reasoning_summary_part.done" and event.get("status") == "incomplete":
                    notices.append(f"Reasoning summary output[{event['output_index']}] part[{event['summary_index']}] is incomplete.")
            elif kind in _STREAM_AUDIO_EVENTS:
                if kind.endswith(".delta"):
                    if not isinstance(event.get("delta"), str):
                        raise TypeError(f"{kind}: delta must be text.")
                    if kind == "response.audio.delta":
                        # Keep independently encoded chunks separate; concatenating
                        # padded Base64 strings is not equivalent to joining bytes.
                        audio["chunks"].append(event["delta"])
                    else:
                        audio["transcript"] += event["delta"]
                else:
                    audio["done" if kind == "response.audio.done" else "transcript_done"] = True
            elif kind != "error" and kind not in unknown_event_types:
                unknown_event_types.append(kind)
                notices.append(f"Unrecognized event {kind!r}; preserved without changing accumulated output.")
            item = event.get("item") or {}
            index = event.get("output_index")
            now = time.perf_counter()
            if kind == "response.output_item.added" and item.get("type") == "reasoning":
                reasoning_started[index] = now
            if kind == "response.output_item.done" and index in reasoning_started:
                reasoning_seconds[f"output[{index}]"] = now - reasoning_started.pop(index)
            if event_details:
                finish_line()
                _section(f"EVENT · {kind}")
                _field("payload", _display_data(event, display))
            family, _, phase = kind.rpartition(".")
            if live_text and family in _STREAM_TEXT_FIELDS and phase == "delta":
                is_reasoning = "reasoning" in family
                if display.get("reasoning" if is_reasoning else "answer"):
                    lane = (family, index, event.get("content_index", event.get("summary_index")))
                    if active_lane != lane:
                        finish_line()
                        label = "REASONING" if is_reasoning else ("REFUSAL" if "refusal" in family else "OUTPUT")
                        _section(f"LIVE {label} · output[{index}] part[{lane[2]}]")
                        active_lane = lane
                    delta = event["delta"]
                    print(delta, end="", flush=True)
                    printed_text[lane] = printed_text.get(lane, "") + delta
            elif live_tool_progress and display.get("tools") and (
                phase in _STREAM_TOOL_PHASES.get(family, set())
                or (kind == "response.output_item.done" and item.get("type", "").endswith("_call"))
            ):
                finish_line()
                _field("INTERMEDIATE RESULTS", f'{item.get("type", family)} · {item.get("status", phase)}')
            if kind == "error":
                api_error = True
                response["error"] = {key: value for key, value in event.items() if key != "type"}
                notices.append("The API emitted an error event.")
                break
            if terminal_event:
                break
    except Exception as error:
        stream_error = error
        notices.append(f"Stream interrupted: {type(error).__name__}: {error}")
    finally:
        finish_line()
        # A synchronous stream is owned by this function; always release it.
        close = getattr(events, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:
                stream_error = stream_error or error
                notices.append(f"Stream close failed: {error}")

    if terminal_event is None:
        # Preserve output_index for citations and live lanes in resumed streams.
        response["output"] = [items.get(index, {}) for index in range(max(items, default=-1) + 1)]
        notices.append("No terminal response event received; output is partial.")
    report = _parse_response(response, request_input=request_input)
    report["issues"].extend(notices)
    report["stream"] = dict(
        status="interrupted" if stream_error else ("error" if api_error else (
            "interrupted" if terminal_event is None else response.get("status")
        )),
        terminal_event=terminal_event, events=received_events,
        duration_seconds=time.perf_counter() - started,
        reasoning_seconds=reasoning_seconds, partial_snapshot=terminal_event is None,
        audio=audio, partial_images=partial_images, shell_outputs=shell_outputs,
        unknown_event_types=unknown_event_types,
    )
    if stream_error is not None:
        _render_response(report, display)
        raise ResponseStreamError(str(stream_error), report) from stream_error
    # Live deltas and final snapshots often repeat the same text. Suppress final
    # answer/reasoning only when every corresponding final part was printed fully.
    for switch, families in (
        ("answer", {"response.output_text", "response.refusal"}),
        ("reasoning", {"response.reasoning_text", "response.reasoning_summary_text"}),
    ):
        expected = {}
        for output_index, item in enumerate(response.get("output") or []):
            for family in families:
                item_type, field, _, part_type, text_key = _STREAM_TEXT_FIELDS[family]
                if item.get("type") == item_type:
                    for part_index, part in enumerate(item.get(field) or []):
                        if part.get("type") == part_type:
                            expected[(family, output_index, part_index)] = part.get(text_key, "")
        if expected and all(printed_text.get(key) == value for key, value in expected.items()):
            # Preserve metadata/opaque-reasoning visibility when requested.
            if switch == "answer" and not any(display.get(k) for k in ("message_id", "role", "item_status")):
                display["answer"] = False
            if switch == "reasoning" and not any(item.get("encrypted_content") for item in report["reasoning"]):
                display["reasoning"] = False
    return report
