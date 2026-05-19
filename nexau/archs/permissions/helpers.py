# Permission matching helpers (reference implementations).
#
# RFC-0019: 工具权限管理
#
# 框架附带的内置 tool 匹配 helper 函数。封装了"匹配规则 + raise 异常"的
# 常见模式。开发者可以直接使用，也可以参考其实现编写自己的判断逻辑。

from __future__ import annotations

import fnmatch
import re
import shlex
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import pathspec

from .types import AskPermission, PermissionDenied

if TYPE_CHECKING:
    from nexau.archs.main_sub.framework_context import FrameworkContext

# RFC-0019: "**" 通配符约定
_WILDCARD = "**"

# CC 对齐: Bash 只读命令白名单
# CC 文档明确列出: ls, cat, head, tail, grep, find, wc, diff, stat, du, cd
# 以下扩展命令为纯信息查询/文本处理，不修改文件系统，等同只读语义。
# 注意: sed、awk 可通过 -i / 重定向修改文件，CC 不视为只读，故不纳入。
_READONLY_COMMANDS: frozenset[str] = frozenset(
    {
        # CC 文档: 核心只读
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "wc",
        "diff",
        "stat",
        "du",
        "cd",
        # 扩展: 纯信息查询
        "file",
        "which",
        "whereis",
        "whoami",
        "pwd",
        "echo",
        "printf",
        "env",
        "printenv",
        "date",
        "uname",
        "hostname",
        "id",
        "uptime",
        # 扩展: 路径工具
        "basename",
        "dirname",
        "realpath",
        "readlink",
        # 扩展: 校验/哈希
        "md5sum",
        "sha256sum",
        # 扩展: 纯 stdout 文本处理（不含 sed/awk）
        "sort",
        "uniq",
        "tr",
        "cut",
        "tac",
        "rev",
        "nl",
        "fmt",
        "fold",
        "paste",
        "join",
        "comm",
        "column",
        "seq",
        "strings",
        "xxd",
        # 扩展: 搜索/过滤
        "egrep",
        "fgrep",
        "rg",
        "ag",
        # 扩展: 分页/导航
        "less",
        "more",
        "tree",
        # 扩展: 帮助/类型
        "type",
        "man",
        "help",
        # 扩展: shell 内建
        "test",
        "true",
        "false",
        "[",
    }
)

# CC 对齐: git 只读子命令白名单
_READONLY_GIT_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "log",
        "status",
        "diff",
        "show",
        "branch",
        "tag",
        "remote",
        "config",
        "describe",
        "rev-parse",
        "rev-list",
        "shortlog",
        "blame",
        "ls-files",
        "ls-tree",
        "ls-remote",
        "cat-file",
        "name-rev",
        "reflog",
        "grep",
        "cherry",
        "merge-base",
        "count-objects",
        "verify-commit",
        "verify-tag",
        "whatchanged",
    }
)

# CC 对齐: 进程包装器 — 权限判定前自动剥离，让规则匹配内部实际命令
# 例: timeout 30 git push → 按 "git push" 判定
_PROCESS_WRAPPERS: frozenset[str] = frozenset(
    {
        "timeout",
        "time",
        "nice",
        "nohup",
        "stdbuf",
    }
)

# CC 对齐: 写文件保护路径 — 即使目录已 allow 也强制 ask
#
# 暂时清空：原"保护路径强制 ask"机制与 RFC-0019 的"无 permissions 字段 = 行为与
# 当前一致（`"**"` 无条件放行）"承诺冲突——未声明 permissions 的 tool 走到
# check_path_permission 时，本应被 `"**"` 短路，却因保护路径检查再被拦下 ask。
# 在更细的策略（按 tool opt-in 启用保护、仅 workspace root 生效等）落地前，
# 先把两个集合清空以恢复向后兼容承诺。
#
# 原内容保留为注释以便日后恢复：
# _PROTECTED_DIRS = {".git", ".vscode", ".idea", ".husky", ".claude"}
# _PROTECTED_FILES = {
#     ".gitconfig", ".gitmodules",
#     ".bashrc", ".bash_profile", ".zshrc", ".zprofile", ".profile",
#     ".ripgreprc", ".mcp.json", ".claude.json",
# }
_PROTECTED_DIRS: frozenset[str] = frozenset()
_PROTECTED_FILES: frozenset[str] = frozenset()


def check_permission(
    ctx: FrameworkContext,
    permission_key: str,
    prompt: str,
) -> None:
    """通用三态检查（参考实现）。

    RFC-0019: 内置 tool 的匹配 helper

    对 permission_key 与 allow/deny rules 做精确匹配：
    命中 allow → 返回、命中 deny → raise PermissionDenied、无命中 → raise AskPermission。
    """
    # 1. "**" 通配符 = 无条件放行
    if _WILDCARD in ctx.allow_rules:
        return

    # 2. deny 优先
    if permission_key in ctx.deny_rules:
        raise PermissionDenied(
            reason=f"{permission_key} 被禁止",
            permission_key=permission_key,
        )

    # 3. allow 精确匹配
    if permission_key in ctx.allow_rules:
        return

    # 4. 无命中 → ask
    raise AskPermission(prompt=prompt, permission_key=permission_key)


def _path_to_dir_glob(path: str) -> str:
    """将文件路径转为目录级 glob 规则。

    CC 对齐: allow 一个文件后，同目录下所有文件都自动放行。
    例如 /Users/pcj/project/foo.py → /Users/pcj/project/**
    """
    from pathlib import PurePosixPath

    parent = str(PurePosixPath(path).parent)
    if parent == "/" or parent == ".":
        return "/**"
    return parent + "/**"


def _is_protected_path(path: str) -> bool:
    """判断是否为 CC 保护路径。

    CC 对齐: 这些路径即使目录已 allow 也强制 ask。
    """
    from pathlib import PurePosixPath

    parts = PurePosixPath(path).parts
    filename = PurePosixPath(path).name

    for part in parts:
        if part in _PROTECTED_DIRS:
            return True

    if filename in _PROTECTED_FILES:
        return True

    return False


def check_path_permission(ctx: FrameworkContext, path: str) -> None:
    """路径专用三态检查。

    RFC-0019: 内置 filesystem helper

    使用 pathspec 库（gitignore 语义）做模式匹配。
    CC 对齐: permission_key 为目录级 glob，allow 后同目录文件自动放行。
    CC 对齐: 保护路径（.git, .bashrc 等）即使目录已 allow 也强制 ask。
    供 write_file / replace / apply_patch / multiedit_tool 使用。
    """
    # 1. "**" 通配符 = 无条件放行（但保护路径仍然 ask）
    if _WILDCARD in ctx.allow_rules and not _is_protected_path(path):
        return

    # 2. deny 匹配（gitignore 语义）
    if ctx.deny_rules:
        deny_spec = pathspec.PathSpec.from_lines("gitwildmatch", ctx.deny_rules)
        if deny_spec.match_file(path):
            raise PermissionDenied(
                reason=f"路径 {path} 被禁止",
                permission_key=path,
            )

    # 3. allow 匹配（gitignore 语义）
    if ctx.allow_rules:
        allow_spec = pathspec.PathSpec.from_lines("gitwildmatch", ctx.allow_rules)
        if allow_spec.match_file(path):
            # CC 对齐: 保护路径即使 allow 也强制 ask
            if _is_protected_path(path):
                raise AskPermission(
                    prompt=f"允许访问受保护路径 {path} 吗?",
                    permission_key=path,
                )
            return

    # 4. 无命中 → ask（CC 对齐: permission_key 为目录级 glob）
    dir_glob = _path_to_dir_glob(path)
    raise AskPermission(
        prompt=f"允许访问 {path} 吗?",
        permission_key=dir_glob,
    )


def _split_shell_commands(command: str) -> list[str]:
    """按管道/链式操作符拆分命令，尊重引号。

    CC 对齐: 引号内的 ``|``, ``&&``, ``||``, ``;`` 不作为操作符。
    """
    parts: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    i = 0

    while i < len(command):
        c = command[i]

        if c == "\\" and not in_single and i + 1 < len(command):
            current.append(c)
            current.append(command[i + 1])
            i += 2
            continue

        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if i + 1 < len(command) and command[i : i + 2] in ("||", "&&"):
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 2
                continue
            elif c in ("|", ";"):
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 1
                continue

        current.append(c)
        i += 1

    part = "".join(current).strip()
    if part:
        parts.append(part)

    return parts


# CC 对齐: 输出重定向检测 — 即使命令头只读，重定向意味着文件写入
_OUTPUT_REDIRECT_RE = re.compile(r"^[0-9]*>{1,2}")

# CC 对齐: shell 解释器 — 检测 shell -c 模式并递归检查内部命令
_SHELL_INTERPRETERS: frozenset[str] = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "dash",
        "ksh",
        "fish",
    }
)


def _is_numeric_arg(s: str) -> bool:
    """判断是否为数字参数（如 timeout 的持续时间）。"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _strip_process_wrappers(tokens: list[str]) -> list[str]:
    """剥离进程包装器前缀，返回实际命令的 tokens。

    CC 对齐: ``timeout 30 git push`` → 按 ``git push`` 判定权限。
    支持: timeout, time, nice, nohup, stdbuf, 裸 xargs。
    """
    i = 0
    while i < len(tokens):
        if tokens[i] in _PROCESS_WRAPPERS:
            i += 1
            while i < len(tokens) and (tokens[i].startswith("-") or _is_numeric_arg(tokens[i])):
                i += 1
        elif tokens[i] == "env":
            # CC 对齐: env VAR=val command → 剥离 env 和赋值，检查内部命令
            j = i + 1
            while j < len(tokens) and tokens[j].startswith("-"):
                j += 1
            while j < len(tokens) and "=" in tokens[j] and not tokens[j].startswith("-"):
                j += 1
            if j < len(tokens):
                i = j
            else:
                break  # standalone env → 保留（只读）
        elif tokens[i] == "xargs" and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            i += 1
        else:
            break
    return tokens[i:] if i < len(tokens) else tokens


# CC 对齐: 有子命令结构的命令集合
# 这些命令的 permission_key 为 "command subcommand"（如 "npm install"），
# 其他命令的 key 为命令头（如 "python"）。
_COMMANDS_WITH_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        # VCS
        "git",
        "svn",
        "hg",
        # JS/TS
        "npm",
        "npx",
        "yarn",
        "pnpm",
        "bun",
        "deno",
        # Python
        "pip",
        "pip3",
        "uv",
        "poetry",
        "pdm",
        "rye",
        "conda",
        # Rust / Go / Java
        "cargo",
        "go",
        "mvn",
        "gradle",
        # Container / K8s
        "docker",
        "docker-compose",
        "podman",
        "kubectl",
        "helm",
        # System package managers
        "brew",
        "apt",
        "apt-get",
        "yum",
        "dnf",
        "pacman",
        "apk",
        # System services
        "systemctl",
        "journalctl",
        "launchctl",
        "service",
        # Build
        "make",
        "cmake",
        "ninja",
    }
)


def _command_permission_key(tokens: list[str]) -> str:
    """提取命令的 permission_key。

    CC 对齐: 有子命令的工具用 "command subcommand" 粒度（如 "npm install"），
    其他命令用命令头粒度（如 "python"）。
    """
    head = tokens[0]
    if head in _COMMANDS_WITH_SUBCOMMANDS and len(tokens) > 1 and not tokens[1].startswith("-"):
        return f"{head} {tokens[1]}"
    return head


def _has_output_redirect(tokens: list[str]) -> bool:
    """检测 tokens 中是否包含输出重定向操作符。

    CC 对齐: 即使命令头是只读的，输出重定向意味着文件写入，需要 ask。
    检测: ``>``, ``>>``, ``2>``, ``&>``, ``>&`` 等模式。
    """
    for token in tokens[1:]:
        if token == "&>" or token.startswith(">&"):
            return True
        if _OUTPUT_REDIRECT_RE.match(token):
            return True
    return False


def _check_shell_c_inner(
    ctx: FrameworkContext,
    inner_cmd: str,
) -> tuple[str | None, str]:
    """递归检查 shell -c 内部命令。

    CC 对齐: ``bash -c "git push"`` → 按内部 ``git push`` 判定权限。
    """
    sub_commands = _split_shell_commands(inner_cmd)
    first_ask_key = ""

    for sub in sub_commands:
        sub = sub.strip()
        if not sub:
            continue
        try:
            inner_tokens = shlex.split(sub)
        except ValueError:
            inner_tokens = sub.split()
        if not inner_tokens:
            continue
        result, perm_key = _check_single_command(ctx, inner_tokens)
        if result == "ask" and not first_ask_key:
            first_ask_key = perm_key

    if first_ask_key:
        return "ask", first_ask_key
    return None, ""


def _check_single_command(
    ctx: FrameworkContext,
    tokens: list[str],
) -> tuple[str | None, str]:
    """Check one sub-command. Return (None, key)=allow, ("ask", key)=ask, raises on deny."""
    if not tokens or not tokens[0]:
        return None, ""

    # CC 对齐: 剥离进程包装器
    tokens = _strip_process_wrappers(tokens)
    if not tokens:
        return None, ""

    head = tokens[0]

    # CC 对齐: shell -c 递归 — bash -c "inner" → 按内部命令判定
    if head in _SHELL_INTERPRETERS:
        try:
            c_idx = tokens.index("-c")
            if c_idx + 1 < len(tokens):
                return _check_shell_c_inner(ctx, tokens[c_idx + 1])
        except ValueError:
            pass  # 无 -c，按普通命令判定

    perm_key = _command_permission_key(tokens)

    # deny 优先（检查命令头和完整 key）
    if head in ctx.deny_rules or perm_key in ctx.deny_rules:
        raise PermissionDenied(
            reason=f"命令 {perm_key} 被禁止",
            permission_key=perm_key,
        )
    # 只读白名单（CC 对齐: 有输出重定向则不视为只读）
    if head in _READONLY_COMMANDS and not _has_output_redirect(tokens):
        return None, perm_key
    if head == "git" and len(tokens) > 1 and tokens[1] in _READONLY_GIT_SUBCOMMANDS and not _has_output_redirect(tokens):
        return None, perm_key
    # allow 规则（检查命令头和完整 key）
    if head in ctx.allow_rules or perm_key in ctx.allow_rules:
        return None, perm_key
    # 无命中 → ask
    return "ask", perm_key


def check_shell_permission(ctx: FrameworkContext, command: str) -> None:
    """命令专用三态检查。

    RFC-0019: 内置 shell helper

    CC 对齐: 按 ``|``, ``&&``, ``||``, ``;`` 分割命令链，对每个子命令
    分别做 deny → 只读白名单 → allow → ask 检查。任何一个子命令触发
    deny 则整条拒绝，任何一个触发 ask 则整条 ask。
    供 run_shell_command 使用。
    """
    # "**" 通配符 = 无条件放行
    if _WILDCARD in ctx.allow_rules:
        return

    # 按管道/链式操作符拆分子命令
    sub_commands = _split_shell_commands(command)

    need_ask = False
    first_ask_key = ""

    for sub in sub_commands:
        sub = sub.strip()
        if not sub:
            continue
        try:
            tokens = shlex.split(sub)
        except ValueError:
            tokens = sub.split()
        if not tokens:
            continue

        # _check_single_command 内部会 raise PermissionDenied
        result, perm_key = _check_single_command(ctx, tokens)
        if result == "ask" and not need_ask:
            need_ask = True
            first_ask_key = perm_key

    if need_ask:
        raise AskPermission(
            prompt=f"允许执行 {command} 吗?",
            permission_key=first_ask_key,
        )


def check_url_permission(ctx: FrameworkContext, url: str) -> None:
    """域名级三态检查。

    CC 对齐: WebFetch 按域名控制

    从 URL 中提取 hostname，与 allow/deny 规则做匹配。
    deny/allow 规则支持 fnmatch 通配（如 ``*.github.com``）。
    供 web_fetch 使用。
    """
    # 1. "**" 通配符 = 无条件放行
    if _WILDCARD in ctx.allow_rules:
        return

    # 2. 提取域名
    hostname = urlparse(url).hostname or url

    # 3. deny 匹配（支持 *.example.com 通配）
    for pattern in ctx.deny_rules:
        if fnmatch.fnmatch(hostname, pattern):
            raise PermissionDenied(
                reason=f"域名 {hostname} 被禁止",
                permission_key=hostname,
            )

    # 4. allow 匹配
    for pattern in ctx.allow_rules:
        if fnmatch.fnmatch(hostname, pattern):
            return

    # 5. 无命中 → ask
    raise AskPermission(
        prompt=f"允许访问 {url} 吗?",
        permission_key=hostname,
    )


def check_mcp_permission(ctx: FrameworkContext, server_name: str, tool_name: str) -> None:
    """MCP 工具三态权限检查。

    RFC-0019: 内置 MCP helper

    CC 对齐: MCP 工具默认 always-ask，权限键为 ``mcp__{server}__{tool}``。
    支持 server 级通配——allow/deny ``mcp__{server}`` 匹配该 server 下所有工具。
    与 shell 的 head/subcommand 双层匹配模式相同。
    供 MCPTool 使用。
    """
    # 1. "**" 通配符 = 无条件放行
    if _WILDCARD in ctx.allow_rules:
        return

    server_key = f"mcp__{server_name}"
    tool_key = f"mcp__{server_name}__{tool_name}"

    # 2. deny 优先（server 级 + tool 级）
    if server_key in ctx.deny_rules or tool_key in ctx.deny_rules:
        raise PermissionDenied(
            reason=f"MCP 工具 {tool_key} 被禁止",
            permission_key=tool_key,
        )

    # 3. allow（server 级 + tool 级）
    if server_key in ctx.allow_rules or tool_key in ctx.allow_rules:
        return

    # 4. 无命中 → ask
    raise AskPermission(
        prompt=f"允许调用 MCP 工具 {tool_key} 吗?",
        permission_key=tool_key,
    )
