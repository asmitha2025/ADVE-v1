"""
frameroute.mcp_server — expose the router to any MCP client, no SDK needed.

An MCP client (Claude Desktop, Cursor, Cline, Zed) can then ask an assistant
to price, route or audit a video directly:

    cost_video      what would each sampling policy cost? (no model, no GPU,
                    seconds — the number to put in front of a customer)
    route_video     select frames under a budget, with the hard-trigger and
                    max-gap guarantees
    signals_video   summarize the change-signal track
    audit_video     the full Frame Budget Audit one-pager (needs CLIP/OCR;
                    can take minutes)

Run it:

    python -m frameroute.mcp_server        # stdio transport
    frameroute-mcp                         # same, via the installed script

Claude Desktop / Cursor config:

    {"mcpServers": {"frameroute": {
        "command": "python", "args": ["-m", "frameroute.mcp_server"]}}}

Protocol notes: newline-delimited JSON-RPC 2.0 on stdio, stdout carries ONLY
protocol messages — tool output is captured and printed to stderr.
"""
from __future__ import annotations

import contextlib
import json
import sys
from typing import Any, Dict, List, Optional

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "frameroute"

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "cost_video",
        "description": (
            "Price the sampling policies on a video without running any model. "
            "Counts calls per hour for full compute, uniform 1fps, scene-cut and "
            "frameroute, multiplies by the price you supply, and returns the "
            "saving. Runs in seconds on CPU."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "video": {"type": "string", "description": "path to a video file"},
                "budget_per_hour": {"type": "number", "default": 240},
                "usd_per_call": {"type": "number", "default": 0.002,
                                 "description": "price of one downstream model call in USD"},
                "max_frames": {"type": "integer", "default": 0,
                               "description": "0 = all analysed frames"},
                "weights": {"type": "string", "default": "default",
                            "description": "default | static_cam | surveillance | ego_motion"},
            },
            "required": ["video"],
        },
    },
    {
        "name": "route_video",
        "description": (
            "Select frames under a call budget and return their timestamps with "
            "per-pick reasons and the hard-trigger / max-gap guarantees. No model "
            "in the hot path."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "video": {"type": "string"},
                "budget": {"type": "integer", "description": "max model calls"},
                "budget_per_hour": {"type": "number"},
                "max_gap_sec": {"type": "number", "default": 30.0},
                "max_frames": {"type": "integer", "default": 0},
                "weights": {"type": "string", "default": "default"},
                "limit_picks": {"type": "integer", "default": 50,
                                "description": "how many picks to list"},
            },
            "required": ["video"],
        },
    },
    {
        "name": "signals_video",
        "description": "Summarize the change-signal track for a video (state, mean cost, fps).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "video": {"type": "string"},
                "max_frames": {"type": "integer", "default": 0},
                "weights": {"type": "string", "default": "default"},
            },
            "required": ["video"],
        },
    },
    {
        "name": "audit_video",
        "description": (
            "Run the full Frame Budget Audit (retrieval vs OCR ground truth, "
            "latency, money) and return the customer-facing one-pager. Needs "
            "torch+CLIP (and EasyOCR for retrieval); can take minutes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "video": {"type": "string"},
                "price_per_call": {"type": "number", "default": 0.005},
                "volume_hours": {"type": "number",
                                 "description": "customer's monthly video-hours (optional)"},
                "max_frames": {"type": "integer", "default": 400},
                "budget": {"type": "integer"},
                "domain": {"type": "string", "default": "generic"},
                "skip_retrieval": {"type": "boolean", "default": False},
                "skip_latency": {"type": "boolean", "default": False},
            },
            "required": ["video"],
        },
    },
]


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

def _tool_cost(args: Dict[str, Any]) -> str:
    from frameroute.policies import UniformFPS, UniformN, SceneCut, RouterPolicy
    from frameroute.signals import analyze_video

    video = args["video"]
    usd = float(args.get("usd_per_call", 0.002))
    budget_per_hour = float(args.get("budget_per_hour", 240.0))
    max_frames = int(args.get("max_frames", 0)) or None

    track = analyze_video(video, max_frames=max_frames,
                          weights=args.get("weights", "default"))
    hours = track.duration_sec / 3600.0
    budget = max(2, int(round(budget_per_hour * hours)))

    rows = [("full_compute", track.n_analyzed)]
    for pol in (UniformN(), UniformFPS(1.0), SceneCut(),
                RouterPolicy(policy="coverage"), RouterPolicy(policy="hybrid")):
        rows.append((pol.name, len(pol.select(track, budget=budget))))

    lines = [
        f"{video} — {track.duration_sec:.0f}s, {track.n_analyzed} frames analysed, "
        f"${usd}/call",
        f"budget: {budget} calls ({budget_per_hour:g}/hour x {hours:.3f} hours)",
        "",
        "| policy | calls | cost | vs full |",
        "|---|---:|---:|---:|",
    ]
    full = track.n_analyzed
    for name, n in rows:
        cost = n * usd
        ratio = f"{full / max(n, 1):.1f}x" if name != "full_compute" else "—"
        lines.append(f"| {name} | {n} | ${cost:.2f} | {ratio} |")
    lines += [
        "",
        "These are counted calls, not estimates. What they are WORTH depends on "
        "retrieval parity — run audit_video before quoting the saving.",
    ]
    return "\n".join(lines)


def _tool_route(args: Dict[str, Any]) -> str:
    from frameroute.router import FrameRouter

    video = args["video"]
    router = FrameRouter(
        min_gap_sec=float(args.get("min_gap_sec", 0.0)),
        max_gap_sec=float(args.get("max_gap_sec", 30.0)),
    )
    sel, track = router.route(
        video,
        budget=args.get("budget"),
        budget_per_hour=args.get("budget_per_hour"),
        max_frames=int(args.get("max_frames", 0)) or None,
        weights=args.get("weights", "default"),
    )
    limit = int(args.get("limit_picks", 50))
    payload = {
        "video": video,
        "duration_sec": round(sel.duration_sec, 2),
        "frames_analyzed": sel.frames_analyzed,
        "n_calls": sel.n_calls,
        "calls_per_hour": round(sel.calls_per_hour, 1),
        "reason_counts": sel.reason_counts(),
        "guarantees": sel.guarantees(track),
        "picks": [{"frame_idx": p.idx, "t": round(p.t, 3), "reason": p.reason}
                  for p in sel.picks[:limit]],
        "picks_truncated": max(0, len(sel.picks) - limit),
    }
    return json.dumps(payload, indent=2)


def _tool_signals(args: Dict[str, Any]) -> str:
    from frameroute.signals import analyze_video
    track = analyze_video(args["video"],
                          max_frames=int(args.get("max_frames", 0)) or None,
                          weights=args.get("weights", "default"))
    return json.dumps(track.summary(), indent=2)


def _tool_audit(args: Dict[str, Any]) -> str:
    from bench.audit import run_audit
    rep = run_audit(
        args["video"],
        budget=args.get("budget"),
        max_frames=int(args.get("max_frames", 400)),
        price_per_call=float(args.get("price_per_call", 0.005)),
        volume_hours=args.get("volume_hours"),
        domain=args.get("domain", "generic"),
        skip_retrieval=bool(args.get("skip_retrieval", False)),
        skip_latency=bool(args.get("skip_latency", False)),
    )
    return rep.one_pager()


TOOL_FUNCS = {
    "cost_video": _tool_cost,
    "route_video": _tool_route,
    "signals_video": _tool_signals,
    "audit_video": _tool_audit,
}


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                "isError": True}
    # stdout is the protocol channel; any library print must not touch it.
    with contextlib.redirect_stdout(sys.stderr):
        text = fn(arguments or {})
    return {"content": [{"type": "text", "text": text}], "isError": False}


# --------------------------------------------------------------------------
# JSON-RPC 2.0 / MCP dispatch
# --------------------------------------------------------------------------

def handle_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = msg.get("method")
    mid = msg.get("id")
    if mid is None:  # notification (e.g. notifications/initialized)
        return None

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": "0.1.0"},
        }}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        try:
            result = call_tool(params.get("name", ""), params.get("arguments") or {})
        except Exception as e:  # tool errors are results, not protocol errors
            result = {
                "content": [{"type": "text",
                             "text": f"{e.__class__.__name__}: {e}"}],
                "isError": True,
            }
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    return {"jsonrpc": "2.0", "id": mid, "error": {
        "code": -32601, "message": f"Method not found: {method}"}}


def serve(stdin=None, stdout=None) -> int:
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle_message(msg)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
    return 0


def main() -> int:
    print("[frameroute-mcp] stdio server ready", file=sys.stderr, flush=True)
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
