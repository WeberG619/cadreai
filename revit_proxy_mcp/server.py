#!/usr/bin/env python3
"""
Revit MCP Proxy Server
======================
A Model Context Protocol (MCP) server that proxies tool calls to the
RevitMCPBridge2026 Windows named pipe, giving AI agents (Google ADK / Gemini)
live access to an open Revit model.

Architecture:
  Gemini Agent (Google ADK)
      ↓  MCP tool call (stdio)
  This server  (WSL2 Python)
      ↓  subprocess → powershell.exe → NamedPipeClientStream
  RevitMCPBridge2026  (Windows / Revit 2026)
      ↓  JSON-RPC response
  ← result returned to agent

Protocol:
  - Transport: stdio (MCP standard)
  - Pipe:      \\\\.\\pipe\\RevitMCPBridge2026
  - Framing:   UTF-8 JSON, newline-delimited
  - Format:    JSON-RPC  {"method": "...", "params": {...}}
               Response  {"success": true/false, "result": ...}

Author:  Weber Gouin — BIM Ops Studio
Project: CADRE AI Hackathon 2026
"""

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIPE_NAMES = ["RevitMCPBridge2026", "RevitMCPBridge2025"]
PIPE_TIMEOUT_MS = 10_000  # 10 seconds — matches Revit transaction timeout


def detect_pipe() -> str:
    """Auto-detect which Revit bridge pipe is available."""
    for name in PIPE_NAMES:
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 f"Test-Path \\\\.\\pipe\\{name}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip() == "True":
                return name
        except Exception:
            continue
    # Default to 2026 if neither found (will fail gracefully on call)
    return PIPE_NAMES[0]

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------
server = Server("revit-proxy-mcp")


# ---------------------------------------------------------------------------
# Core: named-pipe relay via PowerShell
# ---------------------------------------------------------------------------

def call_revit_pipe(method: str, params: dict) -> dict:
    """
    Send a JSON-RPC request to RevitMCPBridge2026 via PowerShell and return
    the parsed response dict.

    Because this server runs in WSL2, the Windows named pipe is not directly
    accessible from the Linux filesystem. We shell out to powershell.exe which
    runs natively on the Windows side and has full access to the pipe.

    Args:
        method: Revit MCP method name (e.g. "getLevels")
        params: Dict of parameters for the method

    Returns:
        Parsed JSON response dict from Revit, e.g. {"success": true, "result": ...}

    Raises:
        RuntimeError: If PowerShell fails, pipe is unreachable, or timeout fires
    """
    pipe_name = detect_pipe()
    payload = json.dumps({"method": method, "params": params})

    # Escape the JSON payload for safe embedding in a PowerShell string.
    # Single-quote the payload and escape any existing single quotes.
    escaped_payload = payload.replace("'", "''")

    ps_script = textwrap.dedent(f"""
        try {{
            $pipe = [System.IO.Pipes.NamedPipeClientStream]::new(
                '.', '{pipe_name}',
                [System.IO.Pipes.PipeDirection]::InOut
            )
            $pipe.Connect({PIPE_TIMEOUT_MS})
            $writer = [System.IO.StreamWriter]::new($pipe)
            $reader = [System.IO.StreamReader]::new($pipe)
            $writer.AutoFlush = $true
            $writer.WriteLine('{escaped_payload}')
            $response = $reader.ReadLine()
            $pipe.Close()
            Write-Output $response
        }} catch {{
            Write-Output (ConvertTo-Json @{{ success = $false; error = $_.Exception.Message }})
        }}
    """).strip()

    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=15,  # outer Python timeout (slightly above pipe timeout)
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Timed out waiting for RevitMCPBridge2026. "
            "Is Revit open with the bridge plugin loaded?"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "powershell.exe not found. "
            "This server must run in WSL2 with Windows PowerShell accessible."
        )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if not stdout:
        detail = f" PowerShell stderr: {stderr}" if stderr else ""
        raise RuntimeError(
            f"No response from RevitMCPBridge2026.{detail} "
            "Verify Revit is running and the bridge plugin is active."
        )

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON from RevitMCPBridge2026: {stdout!r}"
        ) from exc


def revit_call(method: str, params: dict | None = None) -> str:
    """
    Thin wrapper around call_revit_pipe that always returns a JSON string
    suitable for TextContent. Handles errors uniformly.
    """
    try:
        response = call_revit_pipe(method, params or {})
        return json.dumps(response, indent=2)
    except RuntimeError as exc:
        error_payload = {"success": False, "error": str(exc)}
        return json.dumps(error_payload, indent=2)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Enumerate all Revit tools available to the AI agent."""
    return [

        # ── QUERY ──────────────────────────────────────────────────────────

        Tool(
            name="revit_ping",
            description=(
                "Check whether the RevitMCPBridge2026 named pipe is reachable "
                "and Revit is responding. Call this first to verify connectivity "
                "before running any other Revit tools."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="revit_get_project_info",
            description=(
                "Get high-level metadata about the open Revit project: project "
                "name, project number, client name, building address, status, "
                "and author. Useful for orientation at the start of a session."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="revit_get_levels",
            description=(
                "Get all levels defined in the Revit project, including their "
                "names and elevations. Levels are required by most creation tools "
                "(walls, rooms, floor plans) — always fetch these first."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="revit_get_rooms",
            description=(
                "Get all rooms in the project with their name, room number, "
                "area (square feet), level, and occupancy. Useful for space "
                "analysis, compliance checks, and schedule generation."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="revit_get_walls",
            description=(
                "Get all walls in the model with wall type name, length, height, "
                "level, and element ID. Use element IDs for subsequent "
                "modification or door/window placement."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="revit_get_elements",
            description=(
                "Get elements filtered by Revit built-in category. "
                "Common categories: Doors, Windows, Columns, Beams, Floors, "
                "Roofs, Stairs, Railings, Furniture, Plumbing Fixtures, "
                "Mechanical Equipment, Electrical Fixtures. "
                "Returns element IDs, family/type names, and host info."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Revit category name (e.g. 'Doors', 'Windows', "
                            "'Furniture'). Case-insensitive."
                        ),
                    }
                },
                "required": ["category"],
            },
        ),
        Tool(
            name="revit_get_views",
            description=(
                "Get all views in the project: floor plans, ceiling plans, "
                "sections, elevations, 3D views, drafting views, and legends. "
                "Returns view IDs and names — needed for sheet placement."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="revit_get_sheets",
            description=(
                "Get all sheets in the project with their sheet number, sheet "
                "name, and the list of views placed on each sheet. "
                "Useful for documentation status and QC."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="revit_get_warnings",
            description=(
                "Get all active Revit model warnings (overlapping elements, "
                "unjoined walls, room separation issues, etc.). "
                "Returns warning description, severity, and affected element IDs. "
                "Use this for model QC audits."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="revit_get_element_count",
            description=(
                "Count all elements in the model grouped by Revit category. "
                "Returns a summary dict like {'Walls': 142, 'Doors': 38, ...}. "
                "Use for quick model size assessment or progress tracking."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),

        # ── CREATION ───────────────────────────────────────────────────────

        Tool(
            name="revit_create_walls",
            description=(
                "Create one or more straight walls in the model. Each wall is "
                "defined by a start point (X, Y), end point (X, Y), the target "
                "level name, wall height in feet, and wall type name. "
                "Coordinates are in decimal feet in the Revit project coordinate "
                "system. Returns the created element IDs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "walls": {
                        "type": "array",
                        "description": "List of wall definitions to create",
                        "items": {
                            "type": "object",
                            "properties": {
                                "startX":       {"type": "number", "description": "Start X coordinate (feet)"},
                                "startY":       {"type": "number", "description": "Start Y coordinate (feet)"},
                                "endX":         {"type": "number", "description": "End X coordinate (feet)"},
                                "endY":         {"type": "number", "description": "End Y coordinate (feet)"},
                                "levelName":    {"type": "string", "description": "Target level name (e.g. 'Level 1')"},
                                "height":       {"type": "number", "description": "Wall height in feet"},
                                "wallTypeName": {"type": "string", "description": "Wall type name (e.g. 'Generic - 8\"')"},
                            },
                            "required": ["startX", "startY", "endX", "endY", "levelName", "height", "wallTypeName"],
                        },
                    }
                },
                "required": ["walls"],
            },
        ),
        Tool(
            name="revit_place_door",
            description=(
                "Place a door family instance on an existing wall. "
                "Requires the host wall element ID, the X/Y insertion point "
                "along the wall, and the door type name. "
                "Returns the new door element ID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "wallId":       {"type": "integer", "description": "Element ID of the host wall"},
                    "locationX":    {"type": "number",  "description": "X coordinate of door center (feet)"},
                    "locationY":    {"type": "number",  "description": "Y coordinate of door center (feet)"},
                    "doorTypeName": {"type": "string",  "description": "Door family+type name (e.g. 'Single-Flush: 36\" x 84\"')"},
                },
                "required": ["wallId", "locationX", "locationY", "doorTypeName"],
            },
        ),
        Tool(
            name="revit_place_window",
            description=(
                "Place a window family instance on an existing wall. "
                "Requires the host wall element ID, the X/Y insertion point, "
                "sill height, and the window type name. "
                "Returns the new window element ID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "wallId":         {"type": "integer", "description": "Element ID of the host wall"},
                    "locationX":      {"type": "number",  "description": "X coordinate of window center (feet)"},
                    "locationY":      {"type": "number",  "description": "Y coordinate of window center (feet)"},
                    "sillHeight":     {"type": "number",  "description": "Sill height above level in feet (default 2.5)"},
                    "windowTypeName": {"type": "string",  "description": "Window family+type name (e.g. 'Fixed: 24\" x 48\"')"},
                },
                "required": ["wallId", "locationX", "locationY", "windowTypeName"],
            },
        ),
        Tool(
            name="revit_create_room",
            description=(
                "Create a new room at a specified point on a given level. "
                "The point must be inside a closed boundary of wall faces. "
                "Returns the new room element ID, auto-assigned number, and area."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "levelName": {"type": "string", "description": "Level name where the room will be placed"},
                    "x":         {"type": "number", "description": "X coordinate inside the room boundary (feet)"},
                    "y":         {"type": "number", "description": "Y coordinate inside the room boundary (feet)"},
                    "roomName":  {"type": "string", "description": "Optional room name (e.g. 'Office 101')"},
                },
                "required": ["levelName", "x", "y"],
            },
        ),
        Tool(
            name="revit_tag_room",
            description=(
                "Place a room tag on an existing room element. "
                "The tag will display the room name and number at the room centroid. "
                "Returns the tag element ID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "roomId": {"type": "integer", "description": "Element ID of the room to tag"},
                },
                "required": ["roomId"],
            },
        ),
        Tool(
            name="revit_create_floor_plan",
            description=(
                "Create a new floor plan view for the specified level. "
                "The view will use the default floor plan view template. "
                "Returns the new view element ID and view name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "levelName": {"type": "string", "description": "Level name to create the floor plan for"},
                    "viewName":  {"type": "string", "description": "Optional custom name for the view"},
                },
                "required": ["levelName"],
            },
        ),

        # ── MODIFICATION ───────────────────────────────────────────────────

        Tool(
            name="revit_set_parameter",
            description=(
                "Set the value of an instance or type parameter on any Revit "
                "element. Works with text, integer, real number, and yes/no "
                "parameter types. Use revit_get_elements to find element IDs first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "elementId":     {"type": "integer", "description": "Revit element ID"},
                    "parameterName": {"type": "string",  "description": "Exact parameter name as shown in Revit properties"},
                    "value":         {"description": "New value (string, number, or boolean)"},
                },
                "required": ["elementId", "parameterName", "value"],
            },
        ),
        Tool(
            name="revit_move_element",
            description=(
                "Move an element by a translation vector (dx, dy, dz) in feet. "
                "All three offset components are required; use 0 for no movement "
                "on an axis. Works on most hosted and unhosted elements."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "elementId": {"type": "integer", "description": "Revit element ID to move"},
                    "dx":        {"type": "number",  "description": "X offset in feet"},
                    "dy":        {"type": "number",  "description": "Y offset in feet"},
                    "dz":        {"type": "number",  "description": "Z offset in feet"},
                },
                "required": ["elementId", "dx", "dy", "dz"],
            },
        ),
        Tool(
            name="revit_delete_elements",
            description=(
                "Permanently delete one or more elements from the model by their "
                "element IDs. This action is wrapped in a Revit transaction and "
                "cannot be undone from the AI session (use Ctrl+Z in Revit). "
                "Returns the count of successfully deleted elements."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "elementIds": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of element IDs to delete",
                    }
                },
                "required": ["elementIds"],
            },
        ),

        # ── DOCUMENTATION ──────────────────────────────────────────────────

        Tool(
            name="revit_create_sheet",
            description=(
                "Create a new drawing sheet in the project with a specified sheet "
                "number and name. The sheet will use the default title block loaded "
                "in the project. Returns the new sheet element ID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sheetNumber": {"type": "string", "description": "Sheet number (e.g. 'A-101')"},
                    "sheetName":   {"type": "string", "description": "Sheet name (e.g. 'FLOOR PLAN - LEVEL 1')"},
                },
                "required": ["sheetNumber", "sheetName"],
            },
        ),
        Tool(
            name="revit_place_view_on_sheet",
            description=(
                "Place an existing view onto an existing sheet at the specified "
                "X/Y position (in feet, relative to sheet origin). "
                "A view can only be placed on one sheet at a time. "
                "Returns the viewport element ID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sheetId": {"type": "integer", "description": "Element ID of the target sheet"},
                    "viewId":  {"type": "integer", "description": "Element ID of the view to place"},
                    "x":       {"type": "number",  "description": "X position on sheet in feet"},
                    "y":       {"type": "number",  "description": "Y position on sheet in feet"},
                },
                "required": ["sheetId", "viewId", "x", "y"],
            },
        ),
        Tool(
            name="revit_add_dimension",
            description=(
                "Add a linear dimension string between two or more element "
                "references in a view. Provide the view ID, list of element IDs "
                "to dimension, and the dimension line direction vector. "
                "Returns the new dimension element ID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "viewId":     {"type": "integer", "description": "View element ID where the dimension will be placed"},
                    "elementIds": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Ordered list of element IDs to dimension between",
                    },
                    "directionX": {"type": "number", "description": "X component of dimension line direction vector"},
                    "directionY": {"type": "number", "description": "Y component of dimension line direction vector"},
                },
                "required": ["viewId", "elementIds", "directionX", "directionY"],
            },
        ),
        Tool(
            name="revit_create_schedule",
            description=(
                "Create a Revit element schedule for the specified category. "
                "The schedule will include default fields (Type, Level, Area, etc.) "
                "appropriate for the category. Returns the schedule element ID and "
                "the default fields included."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "category":     {"type": "string", "description": "Revit category to schedule (e.g. 'Rooms', 'Doors', 'Walls')"},
                    "scheduleName": {"type": "string", "description": "Name for the new schedule view"},
                },
                "required": ["category", "scheduleName"],
            },
        ),

        # ── QC / VALIDATION ────────────────────────────────────────────────

        Tool(
            name="revit_validate_model",
            description=(
                "Run a comprehensive model validation sweep. Checks for: "
                "unenclosed rooms, duplicate room numbers, walls not joined at "
                "corners, missing level assignments, zero-length walls, and other "
                "common BIM quality issues. Returns a structured report with "
                "pass/fail status and affected element IDs for each check."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="revit_check_room_compliance",
            description=(
                "Check all rooms against configurable code requirements for "
                "minimum area (sq ft), minimum dimension, and required adjacencies. "
                "Returns a compliance report per room with pass/fail status and "
                "the delta from the minimum requirement."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "minAreaSqFt":    {"type": "number", "description": "Minimum room area in square feet (default: 70)"},
                    "minDimensionFt": {"type": "number", "description": "Minimum room dimension in feet (default: 7)"},
                },
                "required": [],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route incoming MCP tool calls to the appropriate Revit pipe method."""

    try:
        result_json = _dispatch(name, arguments)
        return [TextContent(type="text", text=result_json)]
    except Exception as exc:  # noqa: BLE001
        error = json.dumps({"success": False, "error": str(exc)}, indent=2)
        return [TextContent(type="text", text=error)]


def _dispatch(name: str, args: dict) -> str:
    """Map MCP tool name → Revit JSON-RPC method + params."""

    # ── QUERY ──────────────────────────────────────────────────────────────
    if name == "revit_ping":
        return revit_call("ping")

    if name == "revit_get_project_info":
        return revit_call("getProjectInfo")

    if name == "revit_get_levels":
        return revit_call("getLevels")

    if name == "revit_get_rooms":
        return revit_call("getRooms")

    if name == "revit_get_walls":
        return revit_call("getWalls")

    if name == "revit_get_elements":
        return revit_call("getElementsByCategory", {"category": args["category"]})

    if name == "revit_get_views":
        return revit_call("getViews")

    if name == "revit_get_sheets":
        return revit_call("getSheets")

    if name == "revit_get_warnings":
        return revit_call("getWarnings")

    if name == "revit_get_element_count":
        return revit_call("getElementCount")

    # ── CREATION ───────────────────────────────────────────────────────────
    if name == "revit_create_walls":
        return revit_call("createWalls", {"walls": args["walls"]})

    if name == "revit_place_door":
        return revit_call("placeDoor", {
            "wallId":       args["wallId"],
            "locationPoint": {"x": args["locationX"], "y": args["locationY"]},
            "doorTypeName": args["doorTypeName"],
        })

    if name == "revit_place_window":
        return revit_call("placeWindow", {
            "wallId":         args["wallId"],
            "locationPoint":  {"x": args["locationX"], "y": args["locationY"]},
            "sillHeight":     args.get("sillHeight", 2.5),
            "windowTypeName": args["windowTypeName"],
        })

    if name == "revit_create_room":
        return revit_call("createRoom", {
            "levelName": args["levelName"],
            "point":     {"x": args["x"], "y": args["y"]},
            "roomName":  args.get("roomName", ""),
        })

    if name == "revit_tag_room":
        return revit_call("tagRoom", {"roomId": args["roomId"]})

    if name == "revit_create_floor_plan":
        return revit_call("createFloorPlan", {
            "levelName": args["levelName"],
            "viewName":  args.get("viewName", ""),
        })

    # ── MODIFICATION ───────────────────────────────────────────────────────
    if name == "revit_set_parameter":
        return revit_call("setParameter", {
            "elementId":     args["elementId"],
            "parameterName": args["parameterName"],
            "value":         args["value"],
        })

    if name == "revit_move_element":
        return revit_call("moveElement", {
            "elementId": args["elementId"],
            "translation": {"x": args["dx"], "y": args["dy"], "z": args["dz"]},
        })

    if name == "revit_delete_elements":
        return revit_call("deleteElements", {"elementIds": args["elementIds"]})

    # ── DOCUMENTATION ──────────────────────────────────────────────────────
    if name == "revit_create_sheet":
        return revit_call("createSheet", {
            "sheetNumber": args["sheetNumber"],
            "sheetName":   args["sheetName"],
        })

    if name == "revit_place_view_on_sheet":
        return revit_call("placeViewOnSheet", {
            "sheetId": args["sheetId"],
            "viewId":  args["viewId"],
            "point":   {"x": args["x"], "y": args["y"]},
        })

    if name == "revit_add_dimension":
        return revit_call("addDimension", {
            "viewId":     args["viewId"],
            "elementIds": args["elementIds"],
            "direction":  {"x": args["directionX"], "y": args["directionY"]},
        })

    if name == "revit_create_schedule":
        return revit_call("createSchedule", {
            "category":     args["category"],
            "scheduleName": args["scheduleName"],
        })

    # ── QC / VALIDATION ────────────────────────────────────────────────────
    if name == "revit_validate_model":
        return revit_call("validateModel")

    if name == "revit_check_room_compliance":
        return revit_call("checkRoomCompliance", {
            "minAreaSqFt":    args.get("minAreaSqFt", 70),
            "minDimensionFt": args.get("minDimensionFt", 7),
        })

    # Fallback — unknown tool
    return json.dumps({
        "success": False,
        "error": f"Unknown tool: '{name}'. This tool is not registered in the Revit MCP proxy.",
    }, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Start the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
