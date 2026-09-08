Start with [__init__.py](__init__.py). It lists the functions available to notebooks.
Existing imports still work:

```python
from common import configure_notebook, inspect_response, inspect_response_stream, inspect_conversation
```

| File | Open it when you want to understand |
| --- | --- |
| [setup.py](setup.py) | Loading `.env` and quieting warnings and library logs |
| [response.py](response.py) | Inspecting one Response object and returning its report |
| [streaming.py](streaming.py) | Reading and closing one stream, including partial results on failure |
| [conversation.py](conversation.py) | Tracking requests, local tools, events, tokens, cost, and timing |
| [parsing.py](parsing.py) | Converting SDK objects to plain data and grouping output items |
| [stream_events.py](stream_events.py) | Building text and tool arguments from streaming deltas |
| [display.py](display.py) | Choosing fields and printing single-response reports |

Each public function has a small JSON example in its docstring. Examples are
illustrative. Report excerpts show selected fields; the actual report retains
the full payload. `configure_notebook()` returns a Path. `inspect_conversation()`
returns a callable observer. The two response inspectors return dictionaries.

For the conversation flow, read `_ConversationInspector.__call__` first. It routes
an event to a named step, records it, and gives `on_event` a copy. Follow
`_start_request`, `_observe_response`, `_record_tool_execution`, then
`_finish_conversation`. Both inspectors use `stream_events.py` to rebuild streamed
output, so text and tool arguments follow the same rules.

`inspect_response_stream(events)` consumes and closes a stream.
`inspect_conversation()` observes events you pass to it. Your notebook still owns
the stream, sends model requests, runs tools, and adds their outputs to history.

For a conversation observer named `debug`, these are the main report entries:

| Entry | Contents |
| --- | --- |
| `debug.report["requests"]` | Each request's exact inputs, settings, response, timing, and usage |
| `debug.report["tool_calls"]` | Arguments, results, status, timing, and matching `call_id` |
| `debug.report["events"]` | Ordered records with `sequence`, `round`, `elapsed_seconds`, `type`, and `data` |
| `debug.report["totals"]` | Reported token and model cost sums, plus coverage counts |

Missing usage becomes JSON `null`. A cost with incomplete coverage is a known
subtotal. Model cost belongs to the model request, not to an individual local
function. Cached and reasoning tokens are subsets, so they are not added again.

Use `print(json.dumps(debug.report["totals"], indent=2))` for a focused JSON view.
Pass `on_event=frontend_events.append` to retain event copies for a frontend.
The default timeline stays readable text. The report stays structured data.

The two single-response inspectors share fresh display defaults from
`display.py`. Override a switch inside `response.py` or `streaming.py` when only
one inspector should print it. Conversation printing stays in `conversation.py`.

Run [Common-Checks.ipynb](../Common-Checks.ipynb) for offline checks. After replacing
the old `common.py` with this package, restart an already running notebook kernel
once so Python imports the new package.
