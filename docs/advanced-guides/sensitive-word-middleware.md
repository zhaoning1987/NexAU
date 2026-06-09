# Sensitive Word Middleware

`SensitiveWordMiddleware` blocks agent runs when configured words appear in model input,
tool results, or model output. A hit stops the current run, returns a refusal response, and
emits a structured `ContentBlockedEvent` when the event middleware is wired.

The middleware does not ship with a default lexicon. You must configure at least one
lexicon source explicitly:

- `lexicon_dir`: directory of `.txt` files; each file name becomes the category.
- `lexicon_file`: a single `.txt` file; the file name becomes the category.
- `lexicon_words`: an inline iterable of words; words use the `explicit` category.

## YAML Configuration

```yaml
middlewares:
  - import: nexau.archs.main_sub.execution.middleware.sensitive_word:SensitiveWordMiddleware
    params:
      lexicon_dir: /opt/nexau/sensitive_lexicon
      case_sensitive: false
      block_input: true
      block_output: true
      raise_on_block: false
```

Each lexicon file should contain one word per line. Empty lines and lines starting with
`#` are ignored.

```text
# /opt/nexau/sensitive_lexicon/security.txt
internal-code-name
restricted phrase
```

## Python Configuration

```python
from nexau import AgentConfig
from nexau.archs.main_sub.execution.middleware.sensitive_word import SensitiveWordMiddleware

config = AgentConfig(
    name="safe_agent",
    middlewares=[
        SensitiveWordMiddleware(
            lexicon_dir="/opt/nexau/sensitive_lexicon",
            case_sensitive=False,
            block_input=True,
            block_output=True,
        )
    ],
)
```

## Behavior

- Input scanning runs in `before_model` for user, system, framework, and tool-result messages.
- Output scanning runs in `after_model` for the complete model response.
- Tool results are scanned before the next model call, after the tool result has been added
  to conversation history.
- If `raise_on_block` is `false`, the run stops with a refusal response. If it is `true`,
  the middleware raises `SensitiveContentBlockedError`.

For a runnable example and a tiny sample lexicon, see `examples/sensitive_word/`.
