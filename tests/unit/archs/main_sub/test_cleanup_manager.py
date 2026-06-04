# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for CleanupManager platform signal integration.

RFC-0020: Windows validation keeps cleanup behavior tied to the platform
compatibility layer instead of registering POSIX-only signals directly.
"""

from __future__ import annotations

import importlib
import signal
from unittest.mock import Mock

import pytest

from nexau.archs.main_sub.utils.cleanup_manager import CleanupManager

cleanup_manager_module = importlib.import_module("nexau.archs.main_sub.utils.cleanup_manager")


@pytest.fixture(autouse=True)
def restore_cleanup_manager_state():
    manager = CleanupManager()
    active_agents = list(manager._active_agents)
    sandbox_manager = manager._sandbox_manager
    cleanup_registered = manager._cleanup_registered
    yield
    manager._active_agents.clear()
    for agent in active_agents:
        manager._active_agents.add(agent)
    manager._sandbox_manager = sandbox_manager
    manager._cleanup_registered = cleanup_registered


def _fresh_manager() -> CleanupManager:
    manager = CleanupManager()
    manager._active_agents.clear()
    manager._sandbox_manager = None
    manager._cleanup_registered = False
    return manager


def test_register_cleanup_handlers_uses_platform_supported_signals(monkeypatch) -> None:
    """RFC-0020: Windows cleanup registration must not assume POSIX signals."""
    manager = _fresh_manager()
    signal_mock = Mock()
    monkeypatch.setattr(cleanup_manager_module, "supported_cleanup_signals", lambda: (signal.SIGINT,))
    monkeypatch.setattr(cleanup_manager_module.signal, "signal", signal_mock)

    manager._register_cleanup_handlers()

    signal_mock.assert_called_once_with(signal.SIGINT, manager._signal_handler)
    assert manager._cleanup_registered is True


def test_signal_handler_cleans_resources_before_platform_reemit(monkeypatch) -> None:
    """RFC-0020: signal cleanup delegates final termination to process_compat."""

    class Agent:
        def __init__(self) -> None:
            self.sync_cleanup = Mock()

    manager = _fresh_manager()
    agent = Agent()
    sandbox_manager = Mock()
    manager._active_agents.add(agent)
    manager._sandbox_manager = sandbox_manager
    reemit_mock = Mock(side_effect=SystemExit(128 + signal.SIGINT))
    monkeypatch.setattr(cleanup_manager_module, "reemit_termination_signal", reemit_mock)

    try:
        manager._signal_handler(signal.SIGINT, None)
    except SystemExit as exc:
        assert exc.code == 128 + signal.SIGINT

    sandbox_manager.stop.assert_called_once_with()
    agent.sync_cleanup.assert_called_once_with()
    reemit_mock.assert_called_once_with(signal.SIGINT)


@pytest.mark.parametrize(
    ("action_env", "expect_stop", "expect_pause"),
    [
        ("none", False, False),  # 不触碰 sandbox（NAC 用此值，#932 修复）
        ("pause", False, True),  # best-effort 暂停
        ("stop", True, False),  # 显式 stop
        ("  STOP  ", True, False),  # strip + lower 归一
        ("kill", True, False),  # 非法值回落 stop
    ],
)
def test_cleanup_sandbox_atexit_action(monkeypatch, action_env, expect_stop, expect_pause) -> None:
    """RFC-0140: 退出清理按 NEXAU_SANDBOX_ATEXIT_ACTION 三态分发（非法值回落 stop）。

    仅验证分发（调对方法），不验证 pause 真正完成——pause_no_wait 非阻塞、
    退出路径下 best-effort（解释器可能在 pause 线程完成前退出）。
    """
    manager = _fresh_manager()
    sandbox_manager = Mock()
    manager._sandbox_manager = sandbox_manager
    monkeypatch.setenv("NEXAU_SANDBOX_ATEXIT_ACTION", action_env)

    manager._cleanup_sandbox()

    assert sandbox_manager.stop.called is expect_stop
    assert sandbox_manager.pause_no_wait.called is expect_pause


def test_cleanup_sandbox_default_is_stop(monkeypatch) -> None:
    """RFC-0140: env 缺省时默认 stop（= 历史行为，单机用户零影响）。"""
    manager = _fresh_manager()
    sandbox_manager = Mock()
    manager._sandbox_manager = sandbox_manager
    monkeypatch.delenv("NEXAU_SANDBOX_ATEXIT_ACTION", raising=False)

    manager._cleanup_sandbox()

    sandbox_manager.stop.assert_called_once_with()
    sandbox_manager.pause_no_wait.assert_not_called()


def test_cleanup_sandbox_none_skips_when_no_manager(monkeypatch) -> None:
    """RFC-0140: 无 sandbox_manager 时直接 return，不读 env、不报错。"""
    manager = _fresh_manager()
    manager._sandbox_manager = None
    monkeypatch.setenv("NEXAU_SANDBOX_ATEXIT_ACTION", "stop")

    # 不应抛异常
    manager._cleanup_sandbox()
