"""§11 MCP server — stdio JSON-RPC 2.0 surface over the §6.10 tool dispatcher.

The L4 tool surface (`ctrldoc.orch.tools`) and its `ToolDispatcher`
are reused verbatim — this package is purely a protocol adapter that
speaks the Model Context Protocol over stdin/stdout so any
MCP-compatible host (Claude Desktop, Claude CLI, third-party) can
discover and invoke the 13-tool surface.

The wire format is the standard MCP stdio transport: one JSON-RPC 2.0
envelope per line of stdin, one response envelope per line of stdout.
Only the three methods a stock host needs are implemented:

    initialize         — handshake; pin the protocol version.
    tools/list         — enumerate the tool catalogue with JSON Schemas.
    tools/call         — dispatch a tool invocation through the L4
                         dispatcher and return a `CallToolResult`.

SPEC-REF: §11 (MCP Server), §6.10 (tool surface)
"""

from __future__ import annotations

from ctrldoc.mcp.server import (
    MCP_PROTOCOL_VERSION,
    MCPError,
    MCPErrorCode,
    MCPServer,
    serve_stdio,
)

__all__ = [
    "MCP_PROTOCOL_VERSION",
    "MCPError",
    "MCPErrorCode",
    "MCPServer",
    "serve_stdio",
]
