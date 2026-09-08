"""Observe a conversation across requests and local tool executions.

Read __call__ first: it routes each event to one step, then publishes a copy.
The report holds data; the notebook still owns model requests and tool execution.
"""

import copy
import json
import math
import time
from collections.abc import Callable

from .parsing import _parse_response, _plain_data, _request_input_data
from .stream_events import _STREAM_ITEM_EVENTS, _STREAM_TEXT_FIELDS, _TERMINAL_EVENTS, _apply_stream_event


def inspect_conversation(*, live: bool = True, on_event: Callable | None = None,
                         show_reasoning: bool = False):
    """Create an observer for one conversation run. It never calls APIs or tools.

    debug = inspect_conversation()
    debug("conversation.started", input="What time is it?")
    debug("request.started", input=history, model=model, stream=True)
    # In YOUR stream loop: debug(event)
    # Around YOUR tool execution: debug("tool.started", call_id=..., ...)
    debug("conversation.finished", status="completed")
    report = debug.report

    Accepts decoded SDK streaming events, Response objects, and application
    events (request.started/failed/interrupted, tool.started/completed/failed,
    conversation.started/finished). Call once per observed event.
    on_event receives a JSON-ready copy: sequence, round, elapsed_seconds,
    type, data. SDK event names/payloads are retained for frontend replay.
    live=False disables printing, not collection or callbacks.
    show_reasoning=True also prints API-provided reasoning text live.

    Usage belongs to model requests. Cached/reasoning tokens are subsets.
    OpenRouter usage.cost is account credits, not a per-function price.
    Missing usage stays unknown; totals include coverage and known subtotals.
    Raw request inputs, responses, and events stay in memory in .report.
    Transport handling, stream closure, and execution remain with the caller.

    Example options, passed as inspect_conversation(**options):
    ```json
    {"live": false, "show_reasoning": false}
    ```
    Returns a callable observer, not JSON. Its initial debug.report is:
    ```json
    {"schema_version": 1, "status": "idle", "requests": [], "tool_calls": [],
     "events": [], "totals": {}, "issues": [], "output_text": ""}
    ```
    See __call__ for the event format; see _totals for usage accounting.
    """
    return _ConversationInspector(live=live, on_event=on_event, show_reasoning=show_reasoning)


class _ConversationInspector:
    def __init__(self, *, live, on_event, show_reasoning):
        self.live, self.on_event = live, on_event
        self.show_reasoning = show_reasoning
        self.report = dict(schema_version=1, status="idle", requests=[], tool_calls=[],
                           events=[], totals={}, issues=[], output_text="")
        self._started = time.perf_counter()
        self._request = None
        self._items, self._shell_outputs, self._seen = {}, {}, {}
        self._calls = {}
        self._lane = None
        self._printed = {}

    def __call__(self, event, **fields):
        """Observe one event and return the record also sent to on_event.

        Input dict (equivalent to debug("conversation.started", input="Hello")):
        ```json
        {"type": "conversation.started", "input": "Hello"}
        ```
        Returned record for this first event, with illustrative timing:
        ```json
        {"sequence": 1, "round": 0, "elapsed_seconds": 0.01,
         "type": "conversation.started", "data": {"input": "Hello"}}
        ```
        Identical repeated provider events with sequence_number return None.
        """
        data = _plain_data({"type": event, **fields} if isinstance(event, str) else event)
        if not isinstance(data, dict):
            raise TypeError("Pass a decoded SDK event, Response, or application event name.")
        if data.get("object") == "response":
            data = {"type": f"response.{data.get('status')}", "response": data}
        data = copy.deepcopy(data)
        kind = data.get("type")
        if not isinstance(kind, str):
            raise TypeError("Conversation events require a string type.")
        elapsed = time.perf_counter() - self._started
        request = self._request

        if kind == "conversation.started":
            self.report.update(status="running", input=data.get("input"))
            self._line(f"User: {data.get('input', '')}")
        elif kind == "request.started":
            request = self._start_request(data, elapsed)
        elif kind.startswith("response.") or kind == "error":
            if not self._observe_response(data, elapsed):
                return None  # An identical repeated event has already been recorded.
        elif kind in {"request.failed", "request.interrupted"}:
            self._interrupt_request(data, elapsed)
        elif kind in {"tool.started", "tool.completed", "tool.failed"}:
            self._record_tool_execution(data, elapsed)
        elif kind == "conversation.finished":
            self._finish_conversation(data, elapsed)

        record = dict(sequence=len(self.report["events"]) + 1, round=request["round"] if request else 0,
                      elapsed_seconds=time.perf_counter() - self._started, type=kind,
                      data={k: v for k, v in data.items() if k != "type"})
        self.report["events"].append(copy.deepcopy(record))
        if self.on_event is not None:
            try:
                self.on_event(copy.deepcopy(record))
            except Exception as error:
                self._notice(f"on_event callback failed: {type(error).__name__}: {error}")
        return record

    def _start_request(self, data, elapsed):
        """Snapshot the full request input before the caller sends it.

        Example event:
        ```json
        {"type": "request.started", "model": "example-model", "stream": true,
         "input": [{"role": "user", "content": "Hello"}]}
        ```
        report["requests"] gains the input and settings plus status/timing fields.
        """
        request = self._request
        if request and request["status"] == "in_progress":
            self("request.interrupted", error="A new request began before a terminal event.")
        inputs = _request_input_data(data.get("input"))
        request = dict(data, round=len(self.report["requests"]) + 1, status="in_progress",
                       started_seconds=elapsed, first_event_seconds=None, first_text_seconds=None,
                       output_text="", response_id=None, metrics={})
        self._request = request
        self.report["requests"].append(request)
        self._items, self._shell_outputs, self._seen, self._printed = {}, {}, {}, {}
        results = [i.get("call_id") for i in (inputs if isinstance(inputs, list) else [])
                   if i.get("type") == "function_call_output"]
        for call_id in results:
            if call_id in self._calls:
                self._calls[call_id].setdefault("sent_in_rounds", []).append(request["round"])
        size = len(inputs) if isinstance(inputs, list) else 1
        self._line(f"\nRequest {request['round']} -> {data.get('model')} | {size} history items | tool choice: {data.get('tool_choice', 'auto')}")
        if results:
            self._line(f"  Tool results in this input: {', '.join(str(c) for c in results)}")
        self._totals()
        return request

    def _observe_response(self, data, elapsed):
        """Apply one provider event. Return False only for an identical duplicate."""
        request, kind = self._request, data["type"]
        if request is None:
            raise ValueError("Observe request.started before its response or streaming events.")
        sequence = data.get("sequence_number")
        if sequence is not None:
            if type(sequence) is not int or sequence < 0:
                raise ValueError("sequence_number must be a nonnegative integer.")
            if sequence in self._seen:
                if self._seen[sequence] != data:
                    raise ValueError(f"Conflicting stream events at sequence {sequence}.")
                return False
            if self._seen and sequence < max(self._seen):
                raise ValueError("Out-of-order stream event; cannot reliably inspect deltas.")
            if self._seen and sequence > max(self._seen) + 1:
                self._notice(f"Stream sequence gap: {max(self._seen)} to {sequence}. Some events may be missing.")
            self._seen[sequence] = copy.deepcopy(data)
        if request["first_event_seconds"] is None:
            request["first_event_seconds"] = elapsed - request["started_seconds"]
        snapshot = data.get("response") or {}
        if snapshot.get("id"):
            if request["response_id"] and request["response_id"] != snapshot["id"]:
                raise ValueError("Received another response ID within one request.")
            request["response_id"] = snapshot["id"]
        if data.get("response_id") and request["response_id"] and data["response_id"] != request["response_id"]:
            raise ValueError("Received another response ID within one request.")
        if kind in _STREAM_ITEM_EVENTS:
            _apply_stream_event(self._items, data, self._shell_outputs)
            item = self._items[data["output_index"]]
            if item.get("type") == "function_call":
                ready = kind in {"response.output_item.done", "response.function_call_arguments.done"}
                self._tool(item, "requested" if ready else "preparing")
                data["call_id"] = item.get("call_id")
                data["name"] = item.get("name")
            if kind == "response.output_item.added" and item.get("type") == "reasoning":
                self._line("  Model producing reasoning (only API-provided text can be shown)")
        family, _, phase = kind.rpartition(".")
        if family in _STREAM_TEXT_FIELDS and phase == "delta":
            lane = (family, data.get("output_index"), data.get("content_index", data.get("summary_index")))
            delta = data["delta"]
            self._printed[lane] = self._printed.get(lane, "") + delta
            if family == "response.output_text":
                request["output_text"] += delta
                if request["first_text_seconds"] is None:
                    request["first_text_seconds"] = elapsed - request["started_seconds"]
            if self.live and ("reasoning" not in family or self.show_reasoning):
                if self._lane != lane:
                    label = "API reasoning" if "reasoning" in family else ("Refusal" if "refusal" in family else "Assistant")
                    self._line(f"  {label}:")
                    self._lane = lane
                print(delta, end="", flush=True)
        if kind in _TERMINAL_EVENTS:
            if snapshot.get("status") != kind.removeprefix("response."):
                raise ValueError("Terminal event and response status disagree.")
            self._finish_request(snapshot, elapsed)
        elif kind == "error":
            self("request.failed", error=data)
        return True

    def _finish_request(self, response, elapsed):
        request = self._request
        if request["status"] != "in_progress":
            return  # A repeated final snapshot must not charge the request twice.
        parsed = _parse_response(response)
        usage = parsed["usage"] or {}
        request.update(response=response, response_id=response.get("id"),
                       status=response.get("status"), output_text=parsed["output_text"],
                       duration_seconds=elapsed - request["started_seconds"], usage=usage,
                       error=response.get("error"), incomplete_details=response.get("incomplete_details"))
        request["metrics"] = dict(
            input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"), cost=usage.get("cost"),
            cached_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens"),
            reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
        )
        for item in parsed["function_calls"]:
            self._tool(item, "requested")
        # Final snapshots fill missing text without printing already streamed text twice.
        for index, item in enumerate(response.get("output") or []):
            for family, (kind, field, _, part_type, text_key) in _STREAM_TEXT_FIELDS.items():
                if "reasoning" in family and not self.show_reasoning:
                    continue
                if item.get("type") == kind:
                    for part_index, part in enumerate(item.get(field) or []):
                        value = part.get(text_key, "")
                        lane = (family, index, part_index)
                        if part.get("type") == part_type and value and self._printed.get(lane) != value:
                            label = "API reasoning" if "reasoning" in family else "Assistant"
                            self._line(f"  {label} (final): {value}")
        for issue in parsed["issues"]:
            self._notice(issue)
        for field in ("error", "incomplete_details"):
            if response.get(field):
                self._line(f"  {field}: {json.dumps(response[field], ensure_ascii=False)}")
        m = request["metrics"]
        show = lambda value: "unknown" if value is None else str(value)
        self._line(f"  Request {request['round']}: {request['status']} in {request['duration_seconds']:.2f}s")
        self._line(f"    Response ID: {request['response_id']}")
        if request["first_text_seconds"] is not None:
            self._line(f"    First assistant text: {request['first_text_seconds']:.2f}s after request start")
        self._line(f"    Tokens: input {show(m['input_tokens'])}, output {show(m['output_tokens'])}, total {show(m['total_tokens'])}")
        self._line(f"    Included above: cached {show(m['cached_tokens'])}, reasoning {show(m['reasoning_tokens'])}")
        self._line(f"    Model cost: {show(m['cost'])} credits (this request)")
        self._totals()

    def _interrupt_request(self, data, elapsed):
        """Keep the successfully observed prefix when a request stops early."""
        request, kind = self._request, data["type"]
        if request:
            request.update(status=kind.split(".")[1], error=data.get("error"),
                           duration_seconds=elapsed - request["started_seconds"],
                           partial_output=[self._items[i] for i in sorted(self._items)])
        self._line(f"  {kind}: {data.get('error')}")

    def _tool(self, item, status):
        call_id = item.get("call_id")
        if not call_id:
            return
        if call_id not in self._calls:
            call = dict(call_id=call_id, round=self._request["round"], status="preparing",
                        duration_seconds=None, output=None, error=None)
            self._calls[call_id] = call
            self.report["tool_calls"].append(call)
            self._line(f"  Tool {item.get('name')} [{call_id}]: model preparing arguments")
        call = self._calls[call_id]
        call.update({k: item[k] for k in ("name", "arguments", "id") if k in item})
        if call["status"] == "preparing" and status == "requested":
            call["status"] = status
            try:
                call["parsed_arguments"] = json.loads(call.get("arguments", ""))
            except (TypeError, ValueError):
                call["parsed_arguments"] = None
            self._line(f"  Tool {call.get('name')} [{call_id}]: requested, awaiting local execution")
            self._line(f"    Arguments: {call.get('arguments', '')}")

    def _record_tool_execution(self, data, elapsed):
        """Record what the caller executed, linked to the model call by call_id.

        After observing the model's function call, the caller sends:
        ```json
        {"type": "tool.started", "call_id": "call_demo", "arguments": {"city": "Paris"}}
        ```
        Then, with an illustrative tool result:
        ```json
        {"type": "tool.completed", "call_id": "call_demo", "output": "Sunny"}
        ```
        The matching report["tool_calls"] entry gains status, output and duration.
        This event records execution; the caller still sends function_call_output.
        """
        kind = data["type"]
        call = self._calls.get(data.get("call_id"))
        if call is None:
            raise ValueError("Observe the model's function call before local tool execution.")
        if kind == "tool.started":
            call.update(status="running", started_seconds=elapsed, execution_arguments=data.get("arguments"))
            self._line(f"  Tool {call['name']} [{call['call_id']}]: running locally")
            if call.get("parsed_arguments") != data.get("arguments"):
                self._line(f"    Execution arguments: {json.dumps(data.get('arguments'), ensure_ascii=False)}")
        else:
            call.update(status=kind.split(".")[1], output=data.get("output"), error=data.get("error"))
            if "started_seconds" in call:
                call["duration_seconds"] = elapsed - call["started_seconds"]
            duration = call["duration_seconds"]
            timing = f" in {duration:.3f}s" if duration is not None else " (not executed)"
            self._line(f"  Tool {call['name']} [{call['call_id']}]: {call['status']}{timing}")
            value = data.get("error") if kind == "tool.failed" else data.get("output")
            self._line(f"    {'Error' if kind == 'tool.failed' else 'Result'}: {value}")
        data["tool"] = copy.deepcopy(call)

    def _finish_conversation(self, data, elapsed):
        """Resolve the final status and sum only usage actually reported."""
        request = self._request
        if request and request["status"] == "in_progress":
            self("request.interrupted", error="Stream ended without a terminal response event.")
        status = data.get("status", "interrupted")
        if data.get("error"):
            self.report["error"] = data["error"]
            self._line(f"  Conversation error: {data['error']}")
        if request and request["status"] != "completed" and status == "completed":
            status = request["status"]
        pending = [c for c in self.report["tool_calls"] if c["status"] in {"preparing", "requested", "running"}]
        if pending and status == "completed":
            status = "interrupted"
            self._notice("Conversation ended with unresolved tool calls.")
        self.report.update(status=status, duration_seconds=elapsed,
                           output_text=data.get("output_text", request["output_text"] if request else ""))
        totals = self._totals()
        data.update(status=status, totals=copy.deepcopy(totals), output_text=self.report["output_text"])
        self._line(f"\nConversation: {status} | {len(self.report['requests'])} requests | {len(self.report['tool_calls'])} tool calls | {elapsed:.2f}s")
        for metric, label in (("total_tokens", "Model tokens"), ("cost", "Model cost (credits)")):
            count, total = totals["reported_requests"][metric], totals[metric]
            value = "unknown" if total is None else f"{total:.8g}"
            scope = "total" if count == totals["requests"] else "known subtotal"
            self._line(f"  {label}: {value} ({scope}, reported for {count}/{totals['requests']} requests)")
        self._line("  Function execution fees: not reported. Model cost is not a price per function.")

    def _totals(self):
        """Sum known request metrics, keeping unknown values as None (JSON null).

        Example totals excerpt after one request that reports no usage:
        ```json
        {"requests": 1, "total_tokens": null, "cost": null,
         "cost_unit": "OpenRouter credits", "reported_requests": {"total_tokens": 0, "cost": 0}}
        ```
        Coverage counts tell whether a number is a complete total or a subtotal.
        Cached and reasoning tokens are already included in input/output tokens.
        """
        requests = self.report["requests"]
        metrics = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "cost")
        totals = dict(requests=len(requests), reported_requests={}, cost_unit="OpenRouter credits")
        for metric in metrics:
            values = [r.get("metrics", {}).get(metric) for r in requests]
            known = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
            totals[metric] = sum(known) if known else None
            totals["reported_requests"][metric] = len(known)
        self.report["totals"] = totals
        return totals

    def _line(self, text):
        if self.live:
            if self._lane is not None:
                print(flush=True)
                self._lane = None
            print(text, flush=True)

    def _notice(self, message):
        self.report["issues"].append(message)
        self._line(f"  Notice: {message}")
