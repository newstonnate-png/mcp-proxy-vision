# mcp_proxy_vision

Standalone MCP server that exposes an **on-demand vision tool** for Claude Code sessions running a non-vision model (e.g. `deepseek-v4-flash`).

It lets the active Claude Code model ask a vision-capable Command Code model (e.g. `gpt-5.6-luna`) about an image file, reusing the Claude Code Proxy's caption pipeline, system prompt, and cache.

## What it provides

One tool: **`analyze_image(image_path, question="")`** — analyze a local image file with the proxy's configured vision model and return a detailed description. Use it when the model cannot see an image and the user asks about its contents (people, camera angle, lighting, scene, visible text, specific details).

## How it works

- Imports the proxy's `src/` at runtime (via `CLAUDE_CODE_PROXY_DIR`, default `C:\Users\ADMIN\Document 2\proxy`), so it reuses `CommandCodeClient`, `config`, `KeyManager`, `convert_claude_to_commandcode`, and the vision caption cache — no code duplication.
- Reads the proxy's `.env` (auto-loaded by importing `src`) for `VISION_MODEL`, `COMMANDCODE_API_KEY`, `COMMANDCODE_KEYS_FILE`, `COMMANDCODE_BASE_URL`, `REQUEST_TIMEOUT`.
- Uses the **same** system prompt as the proxy's automatic fallback, so captions are neutral, faithful, and detailed.

## Setup

```bash
# From this directory (C:\Users\ADMIN\mcp_proxy_vision)
uv sync
```

## Registration (Claude Code)

Add to `~/.claude/.mcp.json` (user-global so it's available in all projects):

```json
{
  "mcpServers": {
    "vision": {
      "command": "uv",
      "args": ["run", "mcp-vision"],
      "cwd": "C:\\Users\\ADMIN\\mcp_proxy_vision"
    }
  }
}
```

`cwd` is set to this project so its own venv and the proxy path resolve.

## Testing

```bash
# Confirm the server registers its tool over stdio
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}\n{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n' | uv run mcp-vision
```

`tools/list` should show `analyze_image`.
