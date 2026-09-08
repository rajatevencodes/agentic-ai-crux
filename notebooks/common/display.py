"""Display switches and printing for the two single-response inspectors.

Filtering affects printed output only. Full data stays in the returned report.
Conversation timeline printing lives with its events in conversation.py.
"""

import json
from typing import Any


def _default_display() -> dict[str, bool]:
    """Return fresh switches so streaming can hide repeated text for one call.

    Example excerpt of the returned settings:
    ```json
    {"answer": true, "usage": true, "call_id": false, "raw_response": false}
    ```
    Change these defaults for both response inspectors, or override a switch
    in response.py / streaming.py for only that inspector.
    """
    return dict(
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
    """Filter the printed view only; retain opaque payloads in structured data.

    Example input with default display switches:
    ```json
    {"type": "function_call", "call_id": "call_demo", "name": "get_weather"}
    ```
    Returned view, while the original object keeps its call_id:
    ```json
    {"type": "function_call", "name": "get_weather"}
    ```
    """
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
