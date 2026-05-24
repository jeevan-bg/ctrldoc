"""§11 MCP server core — JSON-RPC 2.0 dispatch over the §6.10 tool surface.

This module implements only the protocol layer. The L4 dispatch surface
is owned by `ctrldoc.orch.tools.ToolDispatcher`; this server hands raw
arguments to the dispatcher, lifts the typed response into an MCP
`CallToolResult`, and returns the JSON-RPC 2.0 envelope.

Why not use the `mcp` Python SDK? The MCP stdio transport is
JSON-RPC 2.0, line-framed, with three required methods for a
read-only tool host (`initialize`, `tools/list`, `tools/call`). The
upstream SDK adds async streaming, prompts, resources, sampling, and
roots — none of which the §11 surface uses. Implementing the line
protocol in-house (a) keeps the dependency graph small, (b) keeps the
integration test deterministic (no async event loop, no third-party
release cadence), and (c) leaves a clean seam for swapping in the SDK
later if the v2 surface grows. ADR-0007 records this decision.

The error model:

* **Transport-level errors** (malformed envelope, unknown method,
  invalid params) → JSON-RPC `error` envelope with a code in the
  reserved band -32768..-32000 (`MCPErrorCode`).
* **Tool-level errors** (unwired handler, validation failure, runtime
  exception inside a handler) → `result.isError = true` per the MCP
  spec, so the host still sees the response correlated to its request
  id. This matches the upstream MCP Python SDK's convention and is the
  contract a stock client expects.

SPEC-REF: §11 (MCP Server), §6.10 (tool surface), §13 (non-negotiable 14)
"""

from __future__ import annotations

import enum
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import IO, Any

from pydantic import BaseModel

from ctrldoc.orch.tools import (
    TOOL_SURFACE_VERSION,
    ToolDispatcher,
    ToolNotImplementedError,
    ToolValidationError,
    UnknownToolError,
    default_dispatcher,
)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------


MCP_PROTOCOL_VERSION: str = "2025-03-26"
"""The MCP wire-protocol date-version this server supports.

The MCP spec versions its wire by ISO date; pinning here means a host
can detect an incompatible server at the `initialize` handshake without
trying every method. Bump when the wire changes (rare); the §6.10 tool
surface is versioned independently by `TOOL_SURFACE_VERSION` so schema
evolution on the L4 surface does not cascade into a wire bump.
"""

_JSONRPC_VERSION = "2.0"


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------


class MCPErrorCode(enum.IntEnum):
    """JSON-RPC 2.0 reserved error codes used by the MCP transport layer.

    Range -32768..-32000 is reserved for protocol-level errors per
    JSON-RPC 2.0; MCP layers on top of that without inventing new
    codes for the transport surface this server speaks.
    """

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


class MCPError(Exception):
    """A transport-level error raised inside the dispatch path.

    Tool-level errors should NOT raise this — they should be returned as
    a `CallToolResult` with `isError=true` so the host sees the response
    correlated to its request id.
    """

    def __init__(self, code: MCPErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Server core
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ToolEntry:
    """One row in the published tool catalogue."""

    name: str
    description: str
    input_schema: dict[str, Any]


class MCPServer:
    """JSON-RPC 2.0 MCP server over an injected `ToolDispatcher`.

    The server is stateless across requests. State (workspaces, the
    ledger, indexes) lives in the engines plugged into the dispatcher;
    this object only owns the protocol parsing and the tool catalogue
    derived from `TOOL_SURFACE`.
    """

    def __init__(self, dispatcher: ToolDispatcher | None = None) -> None:
        self._dispatcher = dispatcher if dispatcher is not None else default_dispatcher()
        # Pre-materialise the catalogue once so `tools/list` is
        # byte-deterministic across calls.
        self._catalogue: tuple[_ToolEntry, ...] = tuple(
            _ToolEntry(
                name=name,
                description=spec.description,
                input_schema=spec.input_model.model_json_schema(),
            )
            for name, spec in self._dispatcher_surface().items()
        )

    def _dispatcher_surface(self) -> dict[str, Any]:
        """Iterate the dispatcher's registered surface in declaration order."""
        return {name: self._dispatcher.spec(name) for name in self._dispatcher.tool_names()}

    # ------------------------------------------------------------------
    # Public entry point — one envelope in, one envelope (or None) out.
    # ------------------------------------------------------------------

    def handle_request(self, envelope: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC 2.0 envelope.

        Returns the response envelope, or `None` for a JSON-RPC
        notification (no `id`).
        """
        request_id = envelope.get("id")

        if envelope.get("jsonrpc") != _JSONRPC_VERSION:
            return _error_envelope(
                request_id, MCPErrorCode.INVALID_REQUEST, "missing or invalid jsonrpc version"
            )

        method = envelope.get("method")
        if not isinstance(method, str) or not method:
            return _error_envelope(request_id, MCPErrorCode.INVALID_REQUEST, "missing method")

        # Notifications (no id) — never produce a response.
        is_notification = "id" not in envelope
        if is_notification:
            # Silently drop — the MCP host sends `notifications/initialized`
            # after the handshake; nothing for us to do.
            return None

        params = envelope.get("params") or {}
        if not isinstance(params, dict):
            return _error_envelope(
                request_id, MCPErrorCode.INVALID_PARAMS, "params must be an object"
            )

        try:
            result = self._dispatch_method(method, params)
        except MCPError as exc:
            return _error_envelope(request_id, exc.code, exc.message)

        return {"jsonrpc": _JSONRPC_VERSION, "id": request_id, "result": result}

    # ------------------------------------------------------------------
    # Method handlers
    # ------------------------------------------------------------------

    def _dispatch_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return self._initialize(params)
        if method == "tools/list":
            return self._tools_list(params)
        if method == "tools/call":
            return self._tools_call(params)
        raise MCPError(MCPErrorCode.METHOD_NOT_FOUND, f"unknown method: {method!r}")

    def _initialize(self, _params: dict[str, Any]) -> dict[str, Any]:
        """Return the MCP handshake payload.

        We do not version-negotiate downwards — clients that send a
        different `protocolVersion` still receive ours; per the MCP
        spec the client decides whether to proceed.
        """
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                # We expose tools only; no prompts, resources, sampling,
                # or roots. `listChanged: false` tells hosts the catalogue
                # is stable across the session.
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": "ctrldoc",
                "version": TOOL_SURFACE_VERSION,
            },
        }

    def _tools_list(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": entry.name,
                    "description": entry.description,
                    "inputSchema": entry.input_schema,
                }
                for entry in self._catalogue
            ]
        }

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise MCPError(MCPErrorCode.INVALID_PARAMS, "tools/call requires a non-empty `name`")

        raw_arguments = params.get("arguments") or {}
        if not isinstance(raw_arguments, dict):
            raise MCPError(MCPErrorCode.INVALID_PARAMS, "tools/call `arguments` must be an object")

        try:
            output = self._dispatcher.dispatch(tool_name=name, raw_input=raw_arguments)
        except UnknownToolError as exc:
            return _tool_error_result(f"unknown_tool: {exc}")
        except ToolNotImplementedError as exc:
            return _tool_error_result(f"not_implemented: {exc}")
        except ToolValidationError as exc:
            return _tool_error_result(f"validation_error: {exc}")
        except Exception as exc:
            return _tool_error_result(f"handler_error: {exc!r}")

        payload = _model_to_jsonable(output)
        text_block = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return {
            "content": [{"type": "text", "text": text_block}],
            "structuredContent": payload,
            "isError": False,
        }


# ---------------------------------------------------------------------------
# Stdio driver
# ---------------------------------------------------------------------------


def serve_stdio(
    *,
    server: MCPServer | None = None,
    instream: IO[str] | None = None,
    outstream: IO[str] | None = None,
) -> None:
    """Read JSON-RPC 2.0 requests from `instream`, write responses to `outstream`.

    One JSON envelope per line. Blank lines are skipped silently. A line
    that fails to parse produces a parse-error envelope (null id) but
    the loop continues — a stock MCP host treats parse errors as the
    server's signal to soldier on, not to abort the session.

    The loop terminates on EOF (the client closed its half of the pipe).

    When called without an explicit `server`, this function wires the
    pure-Python handler floor (`subsumes` / `optimal_transport` /
    `calibration` — see `ctrldoc.mcp.handlers.register_default_handlers`)
    into a fresh dispatcher before serving. Storage-backed, OT-backed,
    and LLM-backed handler waves attach in later slices; until those
    land, their tools remain unwired and surface as `isError=true`.
    """
    import sys as _sys

    if server is None:
        from ctrldoc.mcp.handlers import MCPHandlerDeps, register_default_handlers

        dispatcher = ToolDispatcher()
        # No claim lookup and no calibration data on the bare CLI entry —
        # that wiring belongs to slices that own those substrates
        # (S-158 for the SQLiteStore-backed `claim_lookup`, the
        # calibration sweep slice for per-backend labelled batches).
        # Until then the factory wires `optimal_transport` and
        # `calibration` unconditionally; `subsumes` stays unwired.
        register_default_handlers(dispatcher=dispatcher, deps=MCPHandlerDeps())
        srv = MCPServer(dispatcher=dispatcher)
    else:
        srv = server
    src = instream if instream is not None else _sys.stdin
    dst = outstream if outstream is not None else _sys.stdout

    for line in _iter_lines(src):
        if not line.strip():
            continue

        try:
            envelope = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_envelope(dst, _error_envelope(None, MCPErrorCode.PARSE_ERROR, str(exc)))
            continue

        if not isinstance(envelope, dict):
            _write_envelope(
                dst,
                _error_envelope(None, MCPErrorCode.INVALID_REQUEST, "envelope must be an object"),
            )
            continue

        response = srv.handle_request(envelope)
        if response is not None:
            _write_envelope(dst, response)


def _iter_lines(src: IO[str]) -> Iterable[str]:
    """Yield lines from `src` until EOF."""
    while True:
        line = src.readline()
        if not line:
            return
        yield line


def _write_envelope(dst: IO[str], envelope: dict[str, Any]) -> None:
    """Line-frame one JSON envelope to `dst` and flush so hosts read promptly."""
    dst.write(json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n")
    dst.flush()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_envelope(request_id: Any, code: MCPErrorCode, message: str) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error envelope."""
    return {
        "jsonrpc": _JSONRPC_VERSION,
        "id": request_id,
        "error": {"code": code.value, "message": message},
    }


def _tool_error_result(message: str) -> dict[str, Any]:
    """Build an MCP `CallToolResult` with `isError=true`.

    Note: this is a *result*, not a JSON-RPC error envelope — the host
    still receives a response correlated to its request id.
    """
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


def _model_to_jsonable(output: Any) -> Any:
    """Coerce a dispatcher return into a JSON-serialisable value."""
    if isinstance(output, BaseModel):
        return output.model_dump(mode="json")
    return output


__all__ = [
    "MCP_PROTOCOL_VERSION",
    "MCPError",
    "MCPErrorCode",
    "MCPServer",
    "serve_stdio",
]
