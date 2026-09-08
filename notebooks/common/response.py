"""Inspect one completed or partial Responses object."""

from typing import Any

from .display import _default_display, _render_response
from .parsing import _parse_response

RESPONSES_API_REFERENCE = (
    "https://developers.openai.com/api/reference/python/resources/responses/methods/create"
)


def inspect_response(response: Any, *, request_input: Any = None) -> dict[str, Any]:
    """Parse an SDK Response or response dict and print a configurable report.

    Edit defaults in display.py or override display below; True means print.
    Hidden fields remain in the return value.
    Pass the exact input used for responses.create as request_input to inspect it.
    No API requests or tool execution happen here. Use inspect_response_stream for
    responses.create(stream=True). Reference: RESPONSES_API_REFERENCE.

    Small input dict, accepted by this inspector (not a model request):
    ```json
    {"object": "response", "status": "completed", "output": [
      {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}
    ]}
    ```
    Call: report = inspect_response(response_dict, request_input="Say hello")
    Returned report excerpt, with full response data in report["response"]:
    ```json
    {"input": "Say hello", "output_text": "Hello", "usage": null, "stream": null}
    ```
    """
    # Defaults live in display.py. Override here, e.g. display["call_id"] = True.
    display = _default_display()
    report = _parse_response(response, request_input=request_input)
    _render_response(report, display)
    return report
