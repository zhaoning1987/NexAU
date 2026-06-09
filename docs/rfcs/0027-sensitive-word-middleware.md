# RFC-0027: 敏感词中间件

- **状态**: draft
- **优先级**: P1
- **标签**: `security`, `content-safety`, `middleware`
- **影响服务**: `nexau-py`（Agent 执行管线）
- **创建日期**: 2026-06-04
- **更新日期**: 2026-06-04

## 摘要

为 NexAU Agent 提供一个统一的敏感词拦截中间件 `SensitiveWordMiddleware`：在
LLM 调用前扫描用户/系统/框架消息，在 LLM 返回后扫描模型可见正文，命中即短路
后续工具调用并向用户返回一条统一的合规拒绝回复，整个流程在 Agent Executor 主
循环内以"零异常正常退出"的形态收尾，便于审计与运营。

## 动机

当前 NexAU 在内容安全侧没有标准化的入口：

1. 任何接入企业/政府客户的部署都需要满足《生成式人工智能服务管理暂行办法》
   等强监管要求，否则可能面临下架风险。
2. 各业务团队若各自实现拦截逻辑（Tool 内 ad-hoc 校验、Transport 层正则、
   System Prompt 软提示）会导致：
   - 拦截不全面（出口侧通常无人覆盖）
   - 拒绝话术、日志格式不统一，运营无法回溯
   - 词库维护分散，更新成本高
3. 现有 `MiddlewareManager` 已提供了 `wrap_model_call`、`before_model`、
   `after_model` 等钩子，天然适合放置一层"内容安全网关"，无需改动 Executor。

## 设计

### 概述

分两部分：

1. **框架级强制停止通道**（通用，可被任何中间件复用）：在 `HookResult` 新增
   `force_stop_reason` 字段，并在 `BeforeModelHookInput` 上增加同名 outparam，
   仿 RFC-0026 的 `history_event` 机制——`MiddlewareManager.run_before_model` /
   `run_after_model` 把任意中间件设置的 `force_stop_reason` surface 到 hook_input，
   Executor 在 hook 边界读取并 BREAK，把该 reason 作为最终停止原因。
   **不新增枚举值**：敏感词拦截复用既有的 `AgentStopReason.ERROR_OCCURRED`。

2. **敏感词中间件**：新增
   `nexau/archs/main_sub/execution/middleware/sensitive_word.py`，仅依赖标准库，
   用 `before_model` 拦用户输入、`after_model` 拦模型输出。命中即返回带
   `force_stop_reason=ERROR_OCCURRED` 的 `HookResult`，并把统一拒绝文案作为末条
   assistant 消息（输入侧追加 / 输出侧替换违规原句做脱敏）。`before_model` 命中时
   Executor 在发起 LLM 调用前就短路——不向模型发送请求。中间件同时 override
   `set_event_emitter` 接收执行器注入的统一事件回调，命中时**即时发射专属的
   `ContentBlockedEvent`**（携带来源/类别/命中词；仿 compaction 中间件的
   `CompactionStartedEvent` 事件模式，与终止用的 `RunErrorEvent` 区分，避免重复）。
   链路未装载事件中间件时静默跳过。

> 设计取舍：早期方案曾用单个 `wrap_model_call` + 给 `ModelResponse` 加字段承载
> 停止原因；但那会把"执行器概念（AgentStopReason）"塞进 provider 响应 DTO，层次
> 不干净，且无法被非 wrap 类钩子复用。最终采用本通用通道（见"权衡取舍"）。

词库目录必须显式配置，文件名作为类别标签（如 `贪腐词库.txt` →
category=`贪腐词库`），便于审计日志聚合。仓库示例放在
`examples/sensitive_word/sensitive_lexicon/`，只包含一个极小示例（3 个词，
3 个类别），避免在仓库塞入大量敏感词；生产环境通过 `lexicon_dir` 指向自己的完整词库（可参考 MIT 的
[`konsheng/Sensitive-lexicon`](https://github.com/konsheng/Sensitive-lexicon)，
但其误杀率高，使用前需做误杀精简）。

### 详细设计

#### 数据流

```
input  → before_model(scan messages) ── hit? ─► HookResult(force_stop_reason, messages+=refusal)
                                                   │
                       run_before_model surfaces → hook_input.force_stop_reason
                                                   │
            Executor 读 outparam → BREAK（LLM 未调用），refusal 作为 final_response

output → [LLM 调用] → after_model(scan original_response) ── hit? ─► HookResult(force_stop_reason, 替换违规末条)
                                                   │
   _process_xml_calls_async 内 run_after_model 后、跑工具前短路（should_stop=True）
                                                   │
       Executor 读 hook_input.force_stop_reason → BREAK，refusal 作为 final_response
```

> 关键点：`after_model` 的工具执行发生在 `_process_xml_calls_async` 内部，故输出侧
> 拦截必须在该函数里、执行工具**之前**短路，避免违规输出携带的 tool call 被执行。

#### 关键类型

- `SensitiveHit(word, category, start, end)` — 单次命中。
- `SensitiveScanResult(text, hits)` — 聚合扫描结果，提供
  `matched` / `categories` / `words` 三个派生属性。
- `SensitiveContentBlockedError(source, scan_result)` — 当
  `raise_on_block=True` 时由中间件抛出，供上层 Transport 自定义合规策略
  （例如：上报内容安全平台、强制 401、把请求转向人工审核）。

#### 词库格式

每个 `.txt` 为 UTF-8 文本：

- 一行一个词条
- 空行忽略
- `#` 开头视为注释忽略
- 文件名 stem 即词条 category

#### 匹配算法

实现一个最小化的 Aho-Corasick 自动机 `_AhoCorasick`：

- `add(word, category)` — 构建前累计模式
- `build()` — BFS 计算 fail 链
- `scan(text) -> list[SensitiveHit]` — 单次线性扫描

时间复杂度 `O(|text| + matches)`，内存只与词库 trie 节点数成正比；
即便词库扩到数千词，构建一次 < 50 ms，扫描一条 1KB 文本 < 1 ms（仓库内示例仅 3 词）。

#### 中间件参数

```python
SensitiveWordMiddleware(
    *,
    lexicon_dir: Path | str | None = None,
    lexicon_file: Path | str | None = None,
    lexicon_words: Iterable[str] | None = None,
    extra_words: Iterable[str] | None = None,
    case_sensitive: bool = False,
    block_input: bool = True,
    block_output: bool = True,
    refusal_template: str = _DEFAULT_REFUSAL_TEMPLATE,
    scan_roles: Iterable[Role] | None = None,  # 默认 USER/FRAMEWORK/SYSTEM/TOOL
    raise_on_block: bool = False,
)
```

> **tool result 拦截（A 方案）**：默认 `scan_roles` 含 `TOOL`，故工具结果会在
> **下一轮 `before_model`** 被扫到并拦截。工具结果文本在 `ToolResultBlock`
> 里（`get_text_content()` 取不到），中间件通过 `_extract_scan_text` 额外抽取
> （支持 str 内容与 TextBlock 列表）。注意是"下一轮拦"，且工具结果可能已通过
> `ToolCallResultEvent` 展示给用户；若需"工具结果当场脱敏/终止"，见未解决问题。

`lexicon_dir` / `lexicon_file` / `lexicon_words` 必须至少配置一个；不再内置默认词库。
词库来源可叠加，按"显式词 → 单文件 → 目录 → extra"顺序合并，相同词后入覆盖前入的 category。

#### 默认拒绝话术

```
⚠️ 内容安全提示：检测到{source}包含「{category}」类敏感词（命中 {hits}），
已按照内部内容安全策略中断本次请求。

如确有合规业务需要，请联系内容安全负责人调整词库或申请白名单。
```

`{source}` 自动渲染成"用户输入"或"模型输出"，`{category}` 为命中类别（最多
3 个，`/` 分隔），`{hits}` 为命中词（最多 5 个，多余以 `… (+N more)` 省略）。

### 示例

```python
from nexau.archs.main_sub.execution.middleware.sensitive_word import (
    SensitiveWordMiddleware,
)

agent_config.middlewares.append(
    SensitiveWordMiddleware(lexicon_dir="/opt/nexau/sensitive_lexicon")
)
```

YAML 注册：

```yaml
middlewares:
  - import: nexau.archs.main_sub.execution.middleware.sensitive_word:SensitiveWordMiddleware
    params:
      lexicon_dir: /opt/nexau/sensitive_lexicon
      case_sensitive: false
      block_input: true
      block_output: true
```

## 权衡取舍

### 考虑过的替代方案

1. **在 `before_model` 中改写 messages 注入"拒绝即可"提示** — 不能阻止模型
   实际被调用，且依赖模型"自觉拒答"，可靠性差。
2. **单个 `wrap_model_call` + 给 `ModelResponse` 加 `force_stop_reason` 字段**
   （早期方案 A）— 改动最小（仅 1 处执行器插入），但把执行器概念
   `AgentStopReason` 塞进 provider 响应 DTO，层次不干净，且停止信号只能由
   `wrap_model_call` 类钩子使用，无法被 `before/after_model` 等通用钩子复用。
   **未采用**：最终选了通用的 `HookResult.force_stop_reason` 通道（本 RFC 设计），
   代价是执行器多 2 处 hook 边界检查，换来可被任意中间件复用的强制停止能力。
3. **使用 `pyahocorasick` C 扩展** — 性能更好但引入新依赖，跨平台构建复杂；
   纯 Python AC 在当前词库规模下已远超延迟预算（< 1 ms / 文本）。
4. **正则引擎 + 一次大 alternation** — 词条规模上千时 regex compile 内存爆炸，
   且贪婪匹配语义不易控制。

### 缺点

1. **流式场景部分输出已发出** — `after_model` 在 LLM 返回完整 ModelResponse 后
   才做输出扫描，对 SSE 流式 transport 而言可见 chunk 已经被 `stream_chunk`
   阶段发送给用户；如需"边流边查"，后续可在 `stream_chunk` 钩子增加一层增量
   缓冲扫描（见"未解决的问题"）。
2. **无近义词/谐音/分词混淆识别** — 简单字符串多模式匹配只能命中显式词条，
   绕过手段（火星文、拆字、同音替换）需要更上层的语义模型检测。
3. **词库非默认内置** — 需要团队自行配置词库路径，并评估词条是否符合本部署的
   合规策略，必要时通过 `extra_words` 与 `lexicon_words` 调整。

## 实现计划

### 阶段划分

- [x] Phase 1: 中间件实现 + 示例词库 + 单元测试。
- [ ] Phase 2: 在 `docs/advanced-guides/` 增加使用文档；在示例 agent
  (`examples/`) 默认装配。
- [ ] Phase 3: 流式增量扫描（`stream_chunk` 钩子上的滑动窗口）。
- [ ] Phase 4: 命中事件上报（Langfuse / Trace 上的 `content_safety.blocked`
  span）。

### 相关文件

**框架级通用通道：**
- `nexau/archs/main_sub/execution/hooks.py` — `HookResult.force_stop_reason` 字段 +
  `BeforeModelHookInput` outparam + `run_before_model`/`run_after_model` surfacing
- `nexau/archs/main_sub/execution/executor.py` — `_apply_middleware_force_stop` 助手 +
  3 处短路（before_model 后 / `_process_xml_calls_async` 内 / after_model 返回后）

> 停止原因复用 `ERROR_OCCURRED`，故 `stop_reason.py` 与 `agent_events_middleware.py`
> 均无需改动（`ERROR_OCCURRED` 本就映射为 `RunErrorEvent`）。

**敏感词中间件：**
- `nexau/archs/llm/llm_aggregators/events.py` — 新增 `ContentBlockedEvent` 事件类型
  （加入 `Event` 联合体 + `__all__`）
- `nexau/archs/main_sub/execution/middleware/sensitive_word.py` — 中间件主体；
  命中时设 `force_stop_reason=ERROR_OCCURRED` 并经 `set_event_emitter` 即时发
  `ContentBlockedEvent`
- `examples/sensitive_word/sensitive_lexicon/*.txt` — 示例词库（仓库内仅 3 词示例；生产用 `lexicon_dir` 指向完整词库）

**测试：**
- `tests/unit/test_sensitive_word_middleware.py`
- `tests/unit/test_hooks.py`（通道 surfacing）

## 测试方案

### 单元测试

1. `_AhoCorasick` 正确性（单/多/重叠/空词/构建后追加/未构建即扫）
2. 词库加载（显式、单文件、目录、空、case-sensitive 开关、extra 追加）
3. `scan_messages` 的角色过滤（ASSISTANT 默认不扫）+ tool result 扫描
   （`Role.TOOL` 的 `ToolResultBlock`，str / TextBlock 列表两种内容）
4. `before_model`（输入侧）：干净透传 / 命中设 `force_stop_reason` + 末条拒绝消息 /
   `block_input=False` 跳过 / `raise_on_block`。
5. `after_model`（输出侧）：干净透传 / 命中替换违规末条脱敏 + 设 `force_stop_reason` /
   `block_output=False` 跳过 / `raise_on_block`。
6. `MiddlewareManager`：`run_before_model` / `run_after_model` 正确 surface
   `force_stop_reason` 到 outparam，并在无中间件设置时清除陈旧值。
7. 事件发射：停止原因为 `ERROR_OCCURRED`；wire 了 emitter 时命中即时发
   `ContentBlockedEvent`（含 run_id / source / 类别 / 命中词），无 emitter 时静默
   跳过，干净输入不发事件。
8. 默认示例词库：仓库内 3 词示例（打人/出售雷管/腐败）能命中且 category 正确，
   非示例词不拦。

### 集成测试

`TestExecutorIntegration` 走真实 `Agent.run` → `execute_async`（mock `call_llm_async`）：

- 输入命中：`call_llm_async` 不被调用，返回拒绝文案（短路在 LLM 调用前生效）。
- 输出命中：`call_llm_async` 调用一次，违规原句被脱敏，返回拒绝文案。

后续 Phase 2 可补：与 `LLMFailoverMiddleware`、`ContextCompactionMiddleware` 共存的链式执行。

### 手动验证

```bash
uv run pytest tests/unit/test_sensitive_word_middleware.py -v --no-cov
uv run ruff check nexau/archs/main_sub/execution/middleware/sensitive_word.py
uv run pyright nexau/archs/main_sub/execution/middleware/sensitive_word.py
```

## 未解决的问题

1. 是否需要内置"白名单"机制（例如某类业务允许"暴恐词库"以教育/科研为目的）？
2. 流式拒绝的 UX：发现敏感词时，应该如何回收已经流出的 chunk？
3. 命中事件应该作为常规 `RunErrorEvent` 还是新增 `ContentBlockedEvent`？
4. 词库版本管理：是否引入 `VERSION` 文件并在启动日志中
   打印 hash？

## 参考资料

- [konsheng/Sensitive-lexicon](https://github.com/konsheng/Sensitive-lexicon)
  — 词库来源（MIT License）
- RFC-0003: LLM Failover Middleware — 同样使用 `wrap_model_call` 钩子的先例
- RFC-0026: History Event Channel Cleanup — `HookResult` / `MiddlewareManager`
  契约说明
- Aho-Corasick 自动机原论文：Aho, A.V.; Corasick, M.J. (1975). *Efficient
  string matching: An aid to bibliographic search*.
