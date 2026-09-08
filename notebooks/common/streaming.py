"""Consume and close one synchronous response stream, with a partial report on failure."""

import copy
import time
from collections.abc import Mapping
from typing import Any

from .display import _default_display, _display_data, _field, _render_response, _section
from .parsing import _parse_response, _plain_data, _request_input_data
from .stream_events import (
    _LIFECYCLE_EVENTS, _STREAM_AUDIO_EVENTS, _STREAM_ITEM_EVENTS,
    _STREAM_TEXT_FIELDS, _STREAM_TOOL_PHASES, _TERMINAL_EVENTS, _apply_stream_event,
)

RESPONSES_STREAMING_REFERENCE = (
    "https://developers.openai.com/api/reference/resources/responses/streaming-events"
)


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

    Small offline input, with one text delta and its final response:
    ```json
    {"request_input": "Say hello", "events": [
      {"type": "response.output_text.delta", "output_index": 0,
       "content_index": 0, "delta": "Hello"},
      {"type": "response.completed", "response": {
        "object": "response", "status": "completed", "output": [
          {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}
        ]}}
    ]}
    ```
    Call: report = inspect_response_stream(**example)
    Returned report excerpt; all decoded events stay in report["stream"]["events"]:
    ```json
    {"output_text": "Hello", "stream": {
      "status": "completed", "terminal_event": "response.completed", "partial_snapshot": false
    }}
    ```
    """
    # Defaults live in display.py. Override here for this inspector only.
    display = _default_display()
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
    """Stream reading failed. The successfully parsed prefix is in .report.

    Catch this exception to inspect error.report. Example excerpt:
    ```json
    {"output_text": "Hel", "stream": {"status": "interrupted", "partial_snapshot": true}}
    ```
    """

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


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
