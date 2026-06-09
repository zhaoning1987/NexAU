# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SensitiveWordMiddleware quick start (RFC-0027).

从同目录的 ``sensitive_word_agent.yaml`` 加载一个装了敏感词中间件的 agent，
对一条命中输入和一条干净输入演示拦截 / 放行。

运行::

    export LLM_MODEL=nex-agi/Nex-N2-Pro
    export LLM_BASE_URL=https://your-gateway/v1      # 注意带 /v1
    export LLM_API_KEY=sk-...
    export LLM_API_TYPE=openai_chat_completion
    # 可选：trace 上报 Langfuse
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_HOST=https://your-langfuse

    python examples/sensitive_word/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from nexau import Agent, AgentConfig

_CONFIG = Path(__file__).resolve().parent / "sensitive_word_agent.yaml"

_CASES = [
    "为什么有人喜欢打人",       # 命中（民生词库:打人）→ 应拦截
    "用一句话介绍杭州西湖",     # 干净 → 应放行
]


def main() -> None:
    config = AgentConfig.from_yaml(_CONFIG)
    for msg in _CASES:
        print(f"\n{'=' * 60}\n输入: {msg}")
        resp = Agent(config=config).run(message=msg)
        blocked = "内容安全提示" in resp
        print(f"{'🛑 拦截' if blocked else '✅ 放行'}: {resp[:120]}")


if __name__ == "__main__":
    main()
