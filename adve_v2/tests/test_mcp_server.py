"""
Integration test for the MCP server: spawn it exactly as a client would and
speak newline-delimited JSON-RPC over stdio. No model, no GPU — the cost tool
only needs the change signals.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _rpc(proc, obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    assert line, "server closed stdout without a response"
    return json.loads(line)


def test_mcp_stdio_protocol(tmp_path):
    from bench.selftest import make_test_video

    video = str(tmp_path / "clip.mp4")
    make_test_video(video, w=160, h=90, fps=30)

    proc = subprocess.Popen(
        [sys.executable, "-m", "frameroute.mcp_server"],
        cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
    )
    try:
        init = _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {}})
        assert init["result"]["serverInfo"]["name"] == "frameroute"
        assert "tools" in init["result"]["capabilities"]

        tools = _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in tools["result"]["tools"]}
        assert {"cost_video", "route_video", "signals_video", "audit_video"} <= names
        for t in tools["result"]["tools"]:
            assert t["inputSchema"]["required"] == ["video"]

        cost = _rpc(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "cost_video", "arguments": {
                "video": video, "max_frames": 60, "usd_per_call": 0.002}},
        })
        assert cost["result"]["isError"] is False
        text = cost["result"]["content"][0]["text"]
        assert "full_compute" in text and "router_coverage" in text

        bad_tool = _rpc(proc, {
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        })
        assert bad_tool["result"]["isError"] is True

        missing = _rpc(proc, {"jsonrpc": "2.0", "id": 5, "method": "does/not/exist"})
        assert missing["error"]["code"] == -32601
    finally:
        proc.stdin.close()
        proc.terminate()
        proc.wait(timeout=10)
