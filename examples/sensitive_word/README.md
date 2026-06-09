# 敏感词中间件使用指南（RFC-0027）

`SensitiveWordMiddleware`：模型**输入 / 输出 / 工具结果**命中敏感词即拦截、终止本次
run，返回统一拒绝文案，并发 `ContentBlockedEvent`。

## 文件

| 文件 | 说明 |
|---|---|
| `sensitive_word_agent.yaml` | 示例 agent 配置（装载敏感词中间件 + 可选 Langfuse tracer） |
| `sensitive_lexicon/*.txt` | 示例词库（3 个词，3 个类别） |
| `quickstart.py` | 加载上面的 YAML，演示命中拦截 / 干净放行 |

## 一键运行

```bash
export LLM_MODEL=nex-agi/Nex-N2-Pro
export LLM_BASE_URL=https://your-gateway/v1      # 注意带 /v1
export LLM_API_KEY=sk-...
export LLM_API_TYPE=openai_chat_completion
# 可选：trace 上报 Langfuse
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://your-langfuse

python examples/sensitive_word/quickstart.py
```

预期：「为什么有人喜欢打人」被拦截，「介绍杭州西湖」正常作答。

## 快速开始（Python）

```python
from nexau import Agent, AgentConfig
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.middleware.sensitive_word import SensitiveWordMiddleware

config = AgentConfig(
    name="my_agent",
    llm_config=LLMConfig(model="...", base_url="https://your-gateway/v1", api_key="sk-..."),
    middlewares=[SensitiveWordMiddleware(lexicon_dir="/opt/nexau/sensitive_lexicon")],
)
print(Agent(config=config).run(message="..."))   # 命中敏感词时返回拒绝文案
```

## YAML 配置

```yaml
middlewares:
  - import: nexau.archs.main_sub.execution.middleware.sensitive_word:SensitiveWordMiddleware
    params:
      lexicon_dir: /opt/nexau/sensitive_lexicon   # 生产：指向完整词库（绝对路径最稳）
      case_sensitive: false
      block_input: true       # 拦输入（含工具结果）
      block_output: true      # 拦模型输出
      raise_on_block: false   # false=返回拒绝文案；true=抛 SensitiveContentBlockedError
      extra_words: ["内部代号A", "项目X"]   # 在词库基础上追加
```

## 词库来源（三选一或叠加）

```python
SensitiveWordMiddleware(lexicon_dir="/opt/full_lexicon")    # 目录：每个 .txt 文件名=类别
SensitiveWordMiddleware(lexicon_file="/opt/words.txt")      # 单文件：文件名=类别
SensitiveWordMiddleware(lexicon_words=["打人", "腐败"])      # 直接传词
```

- `lexicon_dir` / `lexicon_file` / `lexicon_words` 必须至少配置一个；不再内置默认词库。
- 示例目录 `examples/sensitive_word/sensitive_lexicon/` 只带 **3 词示例**（民生:打人 /
  涉枪涉爆:出售雷管 / 贪腐:腐败），避免塞入大量敏感词；**生产环境用 `lexicon_dir` 指向你自己的完整词库**。
- `lexicon_dir` 建议用绝对路径（相对路径按进程 CWD 解析）。

## 拦截范围与时机

| 来源 | 钩子 | 时机 |
|---|---|---|
| 用户 / 系统 / 框架输入 | `before_model` | 调模型**前**短路（不发请求） |
| 工具结果（`Role.TOOL`） | `before_model` | 工具执行后的**下一轮** before_model |
| 模型输出 | `after_model` | 整段响应聚合后，跑工具前 |

> 流式输出下，输出侧是"整段流完才拦"（违规 chunk 可能已发给前端）；输入侧不受影响。

## 命中信息怎么拿

1. **拒绝文案**（最终回复 / `RunErrorEvent.message`）：含类别 + 命中词，给人看。
2. **`ContentBlockedEvent`**（结构化，给机器用）：`source` / `categories` / `words` / `message`。
   需链路有事件消费方——走流式传输层时框架会自动装 `AgentEventsMiddleware`，事件随 SSE 流出。
3. **日志**：`[SensitiveWordMiddleware] BLOCKED source=... categories=... hits=...`。

## 工具调用 + 拦截（注意）

要让"工具读取的敏感内容"被拦，模型得能真正调用工具：
- 若网关支持 OpenAI 原生 function calling，用 `tool_call_mode="openai"`。
- 工具返回含敏感词的内容后，会在**下一轮 `before_model`** 被扫到并拦截，
  事件序列为 `... → TOOL_CALL_RESULT → ContentBlockedEvent(source=input) → RUN_ERROR`。

## 可观测性

- `tracers=[LangfuseTracer()]`（读 `LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST` 环境变量）可把 run
  trace 上报 Langfuse。注意：拦截详情在 `ContentBlockedEvent` / 日志里，Langfuse trace 本身
  暂无专属拦截 span。
