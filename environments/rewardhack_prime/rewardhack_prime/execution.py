from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

ExecutionStatus = Literal[
    "passed",
    "failed",
    "syntax_error",
    "runtime_error",
    "timeout",
    "memory_limit",
    "sandbox_error",
]
_ORIGINAL_COMPILE_SUBMISSION: Any | None = None
_ORIGINAL_CALL_FUNCTION_CASE: Any | None = None


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    timeout_seconds: float = 2.0
    memory_limit_mb: int = 256
    stdout_limit_chars: int = 20_000
    stderr_limit_chars: int = 20_000
    max_output_object_size: int = 20_000


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    status: ExecutionStatus
    case_results: list[dict[str, Any]]
    stdout: str
    stderr: str
    duration_seconds: float
    backend: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ExecutionBackend(Protocol):
    async def run_function_cases(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        ...


class LocalTrustedBackend:
    backend_name = "local_trusted"

    async def run_function_cases(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        del timeout_s, memory_mb
        import contextlib
        import io

        compile_submission = _ORIGINAL_COMPILE_SUBMISSION
        call_function_case = _ORIGINAL_CALL_FUNCTION_CASE
        if compile_submission is None or call_function_case is None:
            from rewardhack_gym.envs.code.runtime import (
                call_function_case as runtime_call_function_case,
            )
            from rewardhack_gym.envs.code.runtime import (
                compile_submission as runtime_compile_submission,
            )

            compile_submission = runtime_compile_submission
            call_function_case = runtime_call_function_case

        started = time.perf_counter()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            compilation = compile_submission(source, symbol_name)
            if compilation.symbol is None:
                syntax_ok = compilation.diagnostics.get("syntax_ok", True)
                status: ExecutionStatus = "syntax_error" if not syntax_ok else "runtime_error"
                return ExecutionResult(
                    status=status,
                    case_results=[],
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue(),
                    duration_seconds=time.perf_counter() - started,
                    backend=self.backend_name,
                    diagnostics=dict(compilation.diagnostics),
                )
            case_results = [call_function_case(compilation.symbol, case) for case in cases]
        status = _status_from_case_results(case_results)
        return ExecutionResult(
            status=status,
            case_results=case_results,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
            duration_seconds=time.perf_counter() - started,
            backend=self.backend_name,
            diagnostics={"symbol_found": True},
        )


class SubprocessBackend:
    backend_name = "subprocess"

    def __init__(
        self,
        *,
        stdout_limit_chars: int = 20_000,
        stderr_limit_chars: int = 20_000,
        max_output_object_size: int = 20_000,
    ) -> None:
        self.stdout_limit_chars = stdout_limit_chars
        self.stderr_limit_chars = stderr_limit_chars
        self.max_output_object_size = max_output_object_size

    async def run_function_cases(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        return await asyncio.to_thread(
            self._run_function_cases_sync,
            source,
            symbol_name,
            cases,
            timeout_s,
            memory_mb,
        )

    def _run_function_cases_sync(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        started = time.perf_counter()
        payload = {
            "source": source,
            "symbol_name": symbol_name,
            "cases": cases,
            "memory_mb": memory_mb,
            "stdout_limit_chars": self.stdout_limit_chars,
            "stderr_limit_chars": self.stderr_limit_chars,
            "max_output_object_size": self.max_output_object_size,
        }
        tmpdir = _make_workdir()
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SUBPROCESS_WORKER],
                cwd=tmpdir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = process.communicate(
                    json.dumps(payload),
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                _kill_process_tree(process)
                stdout, stderr = process.communicate()
                return ExecutionResult(
                    status="timeout",
                    case_results=[],
                    stdout=_truncate(stdout, self.stdout_limit_chars),
                    stderr=_truncate(stderr, self.stderr_limit_chars),
                    duration_seconds=time.perf_counter() - started,
                    backend=self.backend_name,
                    diagnostics={"timeout_seconds": timeout_s, "worker_killed": True},
                )
        finally:
            _cleanup_workdir(tmpdir)

        raw_stdout = stdout or ""
        raw_stderr = stderr or ""
        if process.returncode != 0:
            status: ExecutionStatus = (
                "memory_limit" if process.returncode == 75 else "sandbox_error"
            )
            return ExecutionResult(
                status=status,
                case_results=[],
                stdout=_truncate(raw_stdout, self.stdout_limit_chars),
                stderr=_truncate(raw_stderr, self.stderr_limit_chars),
                duration_seconds=time.perf_counter() - started,
                backend=self.backend_name,
                diagnostics={"returncode": process.returncode},
            )
        try:
            result = json.loads(raw_stdout)
        except json.JSONDecodeError:
            return ExecutionResult(
                status="sandbox_error",
                case_results=[],
                stdout=_truncate(raw_stdout, self.stdout_limit_chars),
                stderr=_truncate(raw_stderr, self.stderr_limit_chars),
                duration_seconds=time.perf_counter() - started,
                backend=self.backend_name,
                diagnostics={"error": "worker did not return JSON"},
            )

        return ExecutionResult(
            status=_coerce_status(result.get("status")),
            case_results=list(result.get("case_results", [])),
            stdout=_truncate(str(result.get("captured_stdout", "")), self.stdout_limit_chars),
            stderr=_truncate(
                str(result.get("captured_stderr", raw_stderr)),
                self.stderr_limit_chars,
            ),
            duration_seconds=time.perf_counter() - started,
            backend=self.backend_name,
            diagnostics=dict(result.get("diagnostics", {})),
        )


class DockerBackend(SubprocessBackend):
    backend_name = "docker"

    async def run_function_cases(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        return await asyncio.to_thread(
            self._run_function_cases_sync,
            source,
            symbol_name,
            cases,
            timeout_s,
            memory_mb,
        )

    def _run_function_cases_sync(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        started = time.perf_counter()
        payload = {
            "source": source,
            "symbol_name": symbol_name,
            "cases": cases,
            "memory_mb": memory_mb,
            "stdout_limit_chars": self.stdout_limit_chars,
            "stderr_limit_chars": self.stderr_limit_chars,
            "max_output_object_size": self.max_output_object_size,
        }
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            f"{int(memory_mb)}m",
            "--pids-limit",
            "64",
            "--cpus",
            "1",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
            "-i",
            "python:3.12-slim",
            "python",
            "-I",
            "-S",
            "-c",
            _SUBPROCESS_WORKER,
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            return ExecutionResult(
                status="sandbox_error",
                case_results=[],
                stdout="",
                stderr="docker executable not found",
                duration_seconds=time.perf_counter() - started,
                backend=self.backend_name,
                diagnostics={"available": False},
            )

        try:
            stdout, stderr = process.communicate(json.dumps(payload), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            stdout, stderr = process.communicate()
            return ExecutionResult(
                status="timeout",
                case_results=[],
                stdout=_truncate(stdout, self.stdout_limit_chars),
                stderr=_truncate(stderr, self.stderr_limit_chars),
                duration_seconds=time.perf_counter() - started,
                backend=self.backend_name,
                diagnostics={"timeout_seconds": timeout_s, "worker_killed": True},
            )

        raw_stdout = stdout or ""
        raw_stderr = stderr or ""
        if process.returncode != 0:
            return ExecutionResult(
                status="sandbox_error",
                case_results=[],
                stdout=_truncate(raw_stdout, self.stdout_limit_chars),
                stderr=_truncate(raw_stderr, self.stderr_limit_chars),
                duration_seconds=time.perf_counter() - started,
                backend=self.backend_name,
                diagnostics={"returncode": process.returncode},
            )
        try:
            result = json.loads(raw_stdout)
        except json.JSONDecodeError:
            return ExecutionResult(
                status="sandbox_error",
                case_results=[],
                stdout=_truncate(raw_stdout, self.stdout_limit_chars),
                stderr=_truncate(raw_stderr, self.stderr_limit_chars),
                duration_seconds=time.perf_counter() - started,
                backend=self.backend_name,
                diagnostics={"error": "worker did not return JSON"},
            )

        return ExecutionResult(
            status=_coerce_status(result.get("status")),
            case_results=list(result.get("case_results", [])),
            stdout=_truncate(str(result.get("captured_stdout", "")), self.stdout_limit_chars),
            stderr=_truncate(
                str(result.get("captured_stderr", raw_stderr)),
                self.stderr_limit_chars,
            ),
            duration_seconds=time.perf_counter() - started,
            backend=self.backend_name,
            diagnostics=dict(result.get("diagnostics", {})),
        )


class PrimeSandboxBackend:
    backend_name = "prime_sandbox"

    async def run_function_cases(
        self,
        source: str,
        symbol_name: str,
        cases: list[dict[str, Any]],
        timeout_s: float,
        memory_mb: int,
    ) -> ExecutionResult:
        del source, symbol_name, cases, timeout_s, memory_mb
        return ExecutionResult(
            status="sandbox_error",
            case_results=[],
            stdout="",
            stderr="PrimeSandboxBackend is a placeholder for a future Prime-native sandbox.",
            duration_seconds=0.0,
            backend=self.backend_name,
            diagnostics={"available": False},
        )


@dataclass(frozen=True, slots=True)
class SandboxedFunction:
    source: str
    symbol_name: str
    backend: ExecutionBackend
    limits: ExecutionLimits


def build_execution_backend(name: str, limits: ExecutionLimits) -> ExecutionBackend:
    if name == "local_trusted":
        return LocalTrustedBackend()
    if name == "subprocess":
        return SubprocessBackend(
            stdout_limit_chars=limits.stdout_limit_chars,
            stderr_limit_chars=limits.stderr_limit_chars,
            max_output_object_size=limits.max_output_object_size,
        )
    if name == "docker":
        return DockerBackend(
            stdout_limit_chars=limits.stdout_limit_chars,
            stderr_limit_chars=limits.stderr_limit_chars,
            max_output_object_size=limits.max_output_object_size,
        )
    if name == "prime_sandbox":
        return PrimeSandboxBackend()
    raise ValueError(f"Unknown execution backend {name!r}.")


def install_execution_backend(backend: ExecutionBackend, limits: ExecutionLimits) -> None:
    global _ORIGINAL_CALL_FUNCTION_CASE, _ORIGINAL_COMPILE_SUBMISSION

    import rewardhack_gym.envs.code.runtime as runtime

    if _ORIGINAL_COMPILE_SUBMISSION is None:
        _ORIGINAL_COMPILE_SUBMISSION = runtime.compile_submission
    if _ORIGINAL_CALL_FUNCTION_CASE is None:
        _ORIGINAL_CALL_FUNCTION_CASE = runtime.call_function_case

    runtime.compile_submission = _compile_submission_for_backend(backend, limits)
    runtime.call_function_case = _call_function_case_for_backend

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("rewardhack_gym.envs.code."):
            continue
        if hasattr(module, "compile_submission"):
            module.compile_submission = runtime.compile_submission
        if hasattr(module, "call_function_case"):
            module.call_function_case = runtime.call_function_case


def _compile_submission_for_backend(
    backend: ExecutionBackend,
    limits: ExecutionLimits,
) -> Callable[[str, str], Any]:
    from rewardhack_gym.envs.code.runtime import CompilationResult

    def compile_submission(source: str, symbol_name: str) -> CompilationResult:
        result = _run_backend_sync(
            backend.run_function_cases(
                source,
                symbol_name,
                [],
                timeout_s=limits.timeout_seconds,
                memory_mb=limits.memory_limit_mb,
            )
        )
        diagnostics = {
            "backend": result.backend,
            "status": result.status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            **result.diagnostics,
        }
        if result.status != "passed":
            return CompilationResult(symbol=None, diagnostics=diagnostics)
        return CompilationResult(
            symbol=SandboxedFunction(
                source=source,
                symbol_name=symbol_name,
                backend=backend,
                limits=limits,
            ),
            diagnostics=diagnostics,
        )

    return compile_submission


def _call_function_case_for_backend(fn: Any, case: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(fn, SandboxedFunction):
        if _ORIGINAL_CALL_FUNCTION_CASE is None:
            raise RuntimeError("Original RewardHack call_function_case is unavailable.")
        return _ORIGINAL_CALL_FUNCTION_CASE(fn, case)

    result = _run_backend_sync(
        fn.backend.run_function_cases(
            fn.source,
            fn.symbol_name,
            [case],
            timeout_s=fn.limits.timeout_seconds,
            memory_mb=fn.limits.memory_limit_mb,
        )
    )
    if result.case_results:
        return result.case_results[0]
    return {
        "label": case.get("label", "<case>"),
        "passed": False,
        "error": result.status,
        "expected": case.get("expected"),
        "backend": result.backend,
    }


def _run_backend_sync(awaitable: Any) -> ExecutionResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result_holder: dict[str, ExecutionResult] = {}
    error_holder: dict[str, BaseException] = {}

    def run() -> None:
        try:
            result_holder["result"] = asyncio.run(awaitable)
        except BaseException as exc:
            error_holder["error"] = exc

    import threading

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()
    if error_holder:
        raise error_holder["error"]
    return result_holder["result"]


def _status_from_case_results(case_results: list[dict[str, Any]]) -> ExecutionStatus:
    if any("error" in case for case in case_results):
        return "runtime_error"
    if all(bool(case.get("passed", False)) for case in case_results):
        return "passed"
    return "failed"


def _coerce_status(value: Any) -> ExecutionStatus:
    allowed = {
        "passed",
        "failed",
        "syntax_error",
        "runtime_error",
        "timeout",
        "memory_limit",
        "sandbox_error",
    }
    return value if value in allowed else "sandbox_error"


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        process.kill()
    except ProcessLookupError:
        pass


def _make_workdir() -> str:
    parent = _temporary_parent_dir()
    if parent is not None:
        return tempfile.mkdtemp(prefix="rewardhack-prime-", dir=parent)
    return tempfile.mkdtemp(prefix="rewardhack-prime-")


def _cleanup_workdir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _temporary_parent_dir() -> str | None:
    candidates = [
        os.environ.get("REWARDHACK_PRIME_TMPDIR"),
        str(Path.cwd() / ".rewardhack_prime_tmp"),
        "/tmp",
        "C:/tmp",
        tempfile.gettempdir(),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
            if os.access(path, os.W_OK):
                return str(path)
        except OSError:
            continue
    return None


def _truncate(value: str | None, limit: int) -> str:
    if value is None:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + f"...<truncated {len(value) - limit} chars>"


_SUBPROCESS_WORKER = r'''
import ast
import builtins
import collections
import contextlib
import copy
import functools
import io
import itertools
import json
import math
import re
import sys
import time

try:
    import resource
except Exception:
    resource = None


BLOCKED_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


class SandboxViolation(Exception):
    pass


class LimitedWriter(io.TextIOBase):
    def __init__(self, limit):
        self.limit = int(limit)
        self.parts = []
        self.size = 0
        self.truncated = False

    def write(self, value):
        value = str(value)
        remaining = self.limit - self.size
        if remaining > 0:
            self.parts.append(value[:remaining])
            self.size += min(len(value), remaining)
        if len(value) > remaining:
            self.truncated = True
        return len(value)

    def getvalue(self):
        output = "".join(self.parts)
        if self.truncated:
            output += "...<truncated>"
        return output


def emit(payload):
    sys.__stdout__.write(json.dumps(payload))
    sys.__stdout__.flush()


def apply_memory_limit(memory_mb):
    if resource is None:
        return False
    try:
        limit = int(memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        return True
    except Exception:
        return False


def validate_ast(module):
    for node in ast.walk(module):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxViolation("import blocked")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
                raise SandboxViolation(f"blocked builtin: {func.id}")
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id in {"os", "subprocess", "socket", "pathlib", "shutil"}:
                    raise SandboxViolation(f"blocked module call: {func.value.id}.{func.attr}")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in {"os", "subprocess", "socket", "pathlib", "shutil"}:
                raise SandboxViolation(f"blocked module access: {node.value.id}.{node.attr}")


def bounded_json_value(value, max_size):
    try:
        encoded = json.dumps(value)
        if len(encoded) <= max_size:
            return value
        return f"<output object truncated: {len(encoded)} chars>"
    except Exception:
        rendered = repr(value)
        if len(rendered) > max_size:
            rendered = rendered[:max_size] + "...<truncated>"
        return rendered


def main():
    started = time.perf_counter()
    payload = json.loads(sys.stdin.read())
    source = payload["source"]
    symbol_name = payload["symbol_name"]
    cases = payload.get("cases", [])
    stdout_buffer = LimitedWriter(payload.get("stdout_limit_chars", 20000))
    stderr_buffer = LimitedWriter(payload.get("stderr_limit_chars", 20000))
    max_output_size = int(payload.get("max_output_object_size", 20000))
    memory_limited = apply_memory_limit(payload.get("memory_mb", 256))

    try:
        module = ast.parse(source)
        validate_ast(module)
    except SyntaxError as exc:
        emit({
            "status": "syntax_error",
            "case_results": [],
            "captured_stdout": stdout_buffer.getvalue(),
            "captured_stderr": f"{exc.msg} at line {exc.lineno}:{exc.offset}",
            "diagnostics": {"syntax_ok": False, "memory_limited": memory_limited},
        })
        return
    except SandboxViolation as exc:
        emit({
            "status": "sandbox_error",
            "case_results": [],
            "captured_stdout": stdout_buffer.getvalue(),
            "captured_stderr": str(exc),
            "diagnostics": {"blocked": str(exc), "memory_limited": memory_limited},
        })
        return

    safe_globals = {
        "__name__": "__submission__",
        "__builtins__": {
            "__build_class__": builtins.__build_class__,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "object": object,
            "print": print,
            "range": range,
            "reversed": reversed,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "super": super,
            "tuple": tuple,
            "zip": zip,
        },
        "collections": collections,
        "copy": copy,
        "functools": functools,
        "itertools": itertools,
        "math": math,
        "re": re,
    }

    try:
        namespace = dict(safe_globals)
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(compile(module, "<submission>", "exec"), namespace, namespace)
    except MemoryError:
        emit({
            "status": "memory_limit",
            "case_results": [],
            "captured_stdout": stdout_buffer.getvalue(),
            "captured_stderr": "MemoryError",
            "diagnostics": {"memory_limited": memory_limited},
        })
        return
    except Exception as exc:
        emit({
            "status": "runtime_error",
            "case_results": [],
            "captured_stdout": stdout_buffer.getvalue(),
            "captured_stderr": f"{type(exc).__name__}: {exc}",
            "diagnostics": {
                "syntax_ok": True,
                "execution_ok": False,
                "memory_limited": memory_limited,
            },
        })
        return

    symbol = namespace.get(symbol_name)
    if symbol is None:
        emit({
            "status": "runtime_error",
            "case_results": [],
            "captured_stdout": stdout_buffer.getvalue(),
            "captured_stderr": f"symbol not found: {symbol_name}",
            "diagnostics": {"symbol_found": False, "memory_limited": memory_limited},
        })
        return

    case_results = []
    status = "passed"
    for case in cases:
        args = copy.deepcopy(case.get("args", []))
        kwargs = copy.deepcopy(case.get("kwargs", {}))
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                actual = symbol(*args, **kwargs)
            expected = case.get("expected")
            passed = actual == expected
            if not passed and status == "passed":
                status = "failed"
            case_results.append({
                "label": case.get("label", "<case>"),
                "passed": passed,
                "actual": bounded_json_value(actual, max_output_size),
                "expected": bounded_json_value(expected, max_output_size),
            })
        except MemoryError:
            emit({
                "status": "memory_limit",
                "case_results": case_results,
                "captured_stdout": stdout_buffer.getvalue(),
                "captured_stderr": "MemoryError",
                "diagnostics": {"memory_limited": memory_limited},
            })
            return
        except Exception as exc:
            status = "runtime_error"
            case_results.append({
                "label": case.get("label", "<case>"),
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "expected": bounded_json_value(case.get("expected"), max_output_size),
            })

    emit({
        "status": status,
        "case_results": case_results,
        "captured_stdout": stdout_buffer.getvalue(),
        "captured_stderr": stderr_buffer.getvalue(),
        "diagnostics": {
            "syntax_ok": True,
            "execution_ok": True,
            "symbol_found": True,
            "memory_limited": memory_limited,
            "duration_seconds": time.perf_counter() - started,
        },
    })


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        sys.exit(75)
    except BaseException as exc:
        emit({
            "status": "sandbox_error",
            "case_results": [],
            "captured_stdout": "",
            "captured_stderr": f"{type(exc).__name__}: {exc}",
            "diagnostics": {},
        })
'''
