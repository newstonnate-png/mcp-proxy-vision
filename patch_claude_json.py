"""Patch the stale 'vision' MCP entry in ~/.claude.json to point at the fixed
server (direct venv python -m), matching the updated .mcp.json."""
import json
import os

p = os.path.expanduser("~/.claude.json")
with open(p, encoding="utf-8") as f:
    d = json.load(f)

ms = d.get("mcpServers", {})
if "vision" not in ms:
    print("No vision entry in top-level mcpServers — nothing to patch.")
    raise SystemExit(0)

old = ms["vision"]
ms["vision"] = {
    "type": "stdio",
    "command": r"C:\Users\ADMIN\mcp_proxy_vision\.venv\Scripts\python.exe",
    "args": ["-m", "mcp_proxy_vision.server"],
    "env": {},
    "cwd": "C:\\Users\\ADMIN\\mcp_proxy_vision",
}

with open(p, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

print("Updated vision entry in .claude.json:")
print("  OLD:", json.dumps(old))
print("  NEW:", json.dumps(ms["vision"]))
