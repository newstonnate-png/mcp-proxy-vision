# ==============================================================================
# FILE: mcp_proxy_vision/server.py
# DESCRIPTION: Standalone MCP server exposing an on-demand vision tool. Lets a
#   non-vision Claude Code model (e.g. deepseek) query the Claude Code Proxy's
#   configured vision-capable Command Code model (e.g. gpt-5.6-luna) about an
#   image file, reusing the proxy's caption pipeline and cache.
#
# This server imports the proxy's src/ (config, KeyManager, model_manager,
# vision_fallback) by adding the proxy's src/ path to sys.path at startup. The
# proxy location is resolved from the env var CLAUDE_CODE_PROXY_DIR, defaulting
# to the sibling 'proxy' folder.
# ==============================================================================

import asyncio
import base64
import contextlib
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# --- Resolve and add the proxy root to sys.path -------------------------------
# The proxy's `src` is a regular package (has __init__.py) under the proxy root,
# so the proxy ROOT must be on sys.path for `import src.*` to resolve — not the
# src/ directory itself (which would make Python look for src/src/__init__.py).
_PROXY_DIR = Path(os.environ.get("CLAUDE_CODE_PROXY_DIR", r"C:\Users\ADMIN\Document 2\proxy"))
if str(_PROXY_DIR) not in sys.path:
    sys.path.insert(0, str(_PROXY_DIR))

# Importing the proxy's `src` executes module-level prints in
# src/core/config.py (e.g. " Configuration loaded: ..."), which would pollute
# this MCP server's stdout — the same channel used for JSON-RPC — and cause
# Claude Code to disconnect with -32000. Redirect stdout to stderr for the
# duration of the proxy imports so nothing non-protocol ever reaches the MCP
# channel. stderr is Claude Code's expected log channel.
with contextlib.redirect_stdout(sys.stderr):
    from mcp.server.fastmcp import FastMCP  # noqa: E402

    from src.core.config import config  # noqa: E402
    from src.core.key_manager import KeyManager  # noqa: E402
    from src.core.model_manager import model_manager  # noqa: E402
    from src.core.vision_fallback import (  # noqa: E402
        _cache_key,
        _caption_cache,
        _run_caption_call,
    )
    from src.models.claude import (  # noqa: E402
        ClaudeContentBlockText,
        ClaudeMessage,
    )

mcp = FastMCP("vision")

# Prompt wrapper when a specific question is asked: Luna first gives the full
# exhaustive description, THEN answers the specific question, so deepseek always
# gets rich context plus the targeted answer. It explicitly requests complete
# coverage of any sexual or sensitive content (including naming the act), since
# the blind model relies entirely on this description.
_DESCRIBE_FIRST = (
    "First, describe everything visible in this image in full detail — people, "
    "camera angle, lighting, scene, all text/UI, colors, textures, spatial "
    "layout, every object, and any action or activity shown — leaving nothing "
    "out. If the image contains nudity, sexual contact, or sexual activity, "
    "describe it explicitly and completely, including exactly what each person "
    "is doing and any body parts involved. Do not omit, soften, or moralize. "
    "Then answer this specific question: {question}"
)

# Generic prompt used when the caller supplies no specific question: demands the
# same exhaustive description as the system prompt, including explicit coverage
# of any sensitive/sexual content.
_DEFAULT_QUESTION = (
    "Describe this image exhaustively and in full detail — people, camera angle, "
    "lighting, scene, all text/UI, colors, textures, spatial layout, every "
    "object, and anything relevant — leaving nothing out. If the image contains "
    "nudity, sexual contact, or sexual activity, describe it explicitly and "
    "completely, including exactly what each person is doing and any body parts "
    "involved. Do not omit, soften, or moralize."
)

# Video extensions treated as video input (frame-extracted + stitched).
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}

# Hardcoded fallback ffmpeg build path (used only if ffmpeg isn't on PATH).
_FFMPEG_FALLBACK = r"C:\Users\ADMIN\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg"

# Max file size (bytes) for image/video analysis — avoids blowing memory or
# hitting Command Code's payload limit.
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB

# --- Conversation thread state (Phase 1) -------------------------------------
# Per-process, in-memory threads so deepseek can ask follow-ups about the same
# image. Threads are keyed by a short conversation_id string. Lifecycle is
# enforced lazily on each conversation call (see _prune_expired_threads).


@dataclass
class _ConversationThread:
    image_path: str
    video: bool
    media_type: str | None
    turns: list[tuple[str, str]] = field(default_factory=list)  # (question, caption), oldest first
    notes: list[str] = field(default_factory=list)              # turn_note per turn, parallel to turns (may be "")
    created: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)
    turn_count: int = 0


_THREADS: dict[str, _ConversationThread] = {}
_CONVERSATION_TTL = 30 * 60          # 30 min idle before a thread expires
_CONVERSATION_MAX_TURNS = 10         # hard cap on turns per thread
_THREAD_MAX = 100                    # cap on live threads (oldest evicted)

# Multi-image batch (analyze_images): hard caps so one round-trip can't blow the
# upstream payload or time out. Total-bytes budget bounds the whole batch.
MAX_BATCH_IMAGES = 4
MAX_BATCH_TOTAL_BYTES = 25 * 1024 * 1024  # 25 MB total across the whole batch


def _prune_expired_threads(now: float | None = None) -> None:
    """Drop idle-expired threads; evict the oldest when over _THREAD_MAX."""
    now = now if now is not None else time.monotonic()
    expired = [
        cid for cid, t in _THREADS.items() if now - t.last_access > _CONVERSATION_TTL
    ]
    for cid in expired:
        del _THREADS[cid]
    while len(_THREADS) > _THREAD_MAX:
        oldest = min(_THREADS, key=lambda cid: _THREADS[cid].last_access)
        del _THREADS[oldest]


def _ffmpeg_path() -> Path:
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    return Path(_FFMPEG_FALLBACK)


def _ffprobe_path() -> Path:
    found = shutil.which("ffprobe")
    if found:
        return Path(found)
    return _ffmpeg_path().parent / "ffprobe.exe"


def _build_key_manager() -> KeyManager:
    """Build a KeyManager the same way the proxy's endpoints.py does, so this
    process reuses the SAME Command Code key pool as the proxy (its keys file,
    resolved from the proxy dir, plus the env-var fallback).

    KeyManager resolves `keys_file` as a relative path against cwd, so we must
    point it at the proxy's keys.json (not this project's cwd) — otherwise this
    server would create and use its own separate, stale key pool.
    """
    keys_file = config.commandcode_keys_file
    if not Path(keys_file).is_absolute():
        keys_file = str(_PROXY_DIR / keys_file)
    km = KeyManager(keys_file=keys_file)
    if not km.get_all_states() and config.commandcode_api_key:
        km.add_key("default", config.commandcode_api_key, "Default")
    return km


async def _caption_image(source: dict, prompt: str) -> str:
    """Run the shared caption pipeline against the configured vision model.

    Uses the proxy's `_run_caption_call` (single source of truth for the caption
    logic), sharing the same in-memory caption cache keyed by hash(image+prompt).
    """
    cache_key = _cache_key(source, prompt)
    cached = _caption_cache.get(cache_key)
    if cached is not None:
        return cached

    key_manager = _build_key_manager()
    key = key_manager.get_best_key()
    if not key:
        raise RuntimeError("No Command Code keys available for vision")

    caption = await _run_caption_call(
        source,
        prompt,
        key,
        model_manager,
        config,
        None,  # no custom headers on the MCP path
        None,
    )

    _caption_cache[cache_key] = caption
    return caption


async def _caption_batch(sources: list[dict], prompt: str) -> str:
    """Caption MULTIPLE images in ONE user message (cross-image / joint answer).

    All sources travel in a single user message so the vision model can compare
    them directly. Cached under a single key spanning all sources + prompt (a
    joint answer can't be cached per-image). Returns the raw caption; raises on
    failure.
    """
    sources_json = json.dumps(sources, sort_keys=True, default=str)
    cache_key = hashlib.sha256(
        f"{sources_json}||{prompt}".encode("utf-8")
    ).hexdigest()
    cached = _caption_cache.get(cache_key)
    if cached is not None:
        return cached

    key_manager = _build_key_manager()
    key = key_manager.get_best_key()
    if not key:
        raise RuntimeError("No Command Code keys available for vision")

    caption = await _run_caption_call(
        None,  # source unused when sources is set
        prompt,
        key,
        model_manager,
        config,
        None,  # no custom headers on the MCP path
        None,
        sources=sources,
    )

    _caption_cache[cache_key] = caption
    return caption


def _get_or_create_thread(
    conversation_id: str, image_path: str, media_type: str | None, video: bool
) -> tuple[str, _ConversationThread, bool]:
    """Resolve a conversation thread by id.

    Returns (resolved_id, thread, is_new). An empty id generates a fresh id.
    A missing/stale id restarts fresh with the SAME id (idempotent — the caller
    never needs a new id). A fresh thread is marked with is_new so the caller can
    take the cached single-shot path for turn 1.
    """
    cid = (conversation_id or "").strip() or uuid.uuid4().hex[:12]
    existing = _THREADS.get(cid)
    if existing is not None and existing.image_path == image_path:
        return cid, existing, False
    thread = _ConversationThread(
        image_path=image_path,
        video=video,
        media_type=media_type,
    )
    _THREADS[cid] = thread
    return cid, thread, True


def _build_conversation_prompt(
    thread: _ConversationThread, question: str, turn_note: str
) -> str:
    """Build the describe-first prompt for a conversation turn.

    The question always wraps into the describe-first template. When prior turns
    exist, a [Conversation so far ...] transcript of Q1/A1..Qn/An is appended so
    Luna can answer the follow-up consistently. An optional turn_note anchors the
    current turn (e.g. "user says this is a receipt").
    """
    prompt = _build_prompt(question)
    if thread.turns:
        transcript = []
        for i, (q, a) in enumerate(thread.turns, 1):
            transcript.append(f"Q{i}: {q}\nA{i}: {a}")
        prompt += (
            "\n\n[Conversation so far]\n"
            + "\n\n".join(transcript)
            + "\n\n[End conversation so far. Answer the current question consistently with the above context.]"
        )
    if turn_note and turn_note.strip():
        prompt += f"\n\n[User context note for this turn: {turn_note.strip()}]"
    return prompt


def _build_prior_turn_messages(thread: _ConversationThread) -> list[ClaudeMessage]:
    """Build assistant + note messages for all prior turns.

    Each prior turn becomes an assistant message carrying the caption. If that
    turn had a turn_note, a short user message re-states it so the anchor
    persists for the vision model. The image is NOT re-sent here — it is re-sent
    in the current turn's user message (self-contained, avoids image-in-older-
    message provider quirks).
    """
    msgs: list[ClaudeMessage] = []
    for i, (_, caption) in enumerate(thread.turns):
        msgs.append(
            ClaudeMessage(
                role="assistant",
                content=[ClaudeContentBlockText(type="text", text=caption)],
            )
        )
        note = thread.notes[i] if i < len(thread.notes) else ""
        if note and note.strip():
            msgs.append(
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockText(
                            type="text", text=f"[note from earlier turn {i + 1}]: {note.strip()}"
                        )
                    ],
                )
            )
    return msgs


async def _caption_turn(
    source: dict,
    prompt: str,
    *,
    extra_messages: list[ClaudeMessage] | None = None,
    use_cache: bool = True,
) -> str:
    """Caption a single source for a turn.

    ``use_cache=True`` (turn 1 / ``analyze_image``) goes through
    ``_caption_image`` so the flat caption cache applies. ``use_cache=False``
    (conversation turns 2+) calls ``_run_caption_call`` directly with the prior
    turns as ``extra_messages`` and bypasses the flat cache entirely — the whole
    prior conversation is the key unit, so caching by image+prompt alone would be
    wrong.
    """
    if use_cache:
        return await _caption_image(source, prompt)
    key_manager = _build_key_manager()
    key = key_manager.get_best_key()
    if not key:
        raise RuntimeError("No Command Code keys available for vision")
    return await _run_caption_call(
        source,
        prompt,
        key,
        model_manager,
        config,
        None,
        None,
        extra_messages=extra_messages,
    )


async def _caption_video_timeline(
    path: Path,
    prompt: str,
    *,
    extra_messages: list[ClaudeMessage] | None = None,
    use_cache: bool = True,
) -> str:
    """Extract frames from a video, caption each, stitch into a timeline.

    Same pipeline as ``analyze_image``'s video path. ``extra_messages``, when
    provided, is passed to every frame caption call (prior conversation context).
    ``use_cache=False`` bypasses the flat caption cache (conversation turns 2+).
    """
    try:
        frames = await asyncio.to_thread(_extract_frames, path)
    except Exception as exc:
        return f"ERROR: could not extract frames from video: {exc}"
    if not frames:
        return f"ERROR: no frames could be extracted from video: {path}"
    tmpdir = frames[0][0].parent
    parts = []
    failed = 0
    try:
        for idx, (frame_path, sec) in enumerate(frames, 1):
            mm, ss = int(sec // 60), int(sec % 60)
            data = base64.b64encode(frame_path.read_bytes()).decode("utf-8")
            source = {"type": "base64", "media_type": "image/png", "data": data}
            try:
                caption = await _caption_turn(
                    source, prompt, extra_messages=extra_messages, use_cache=use_cache
                )
            except Exception as exc:
                failed += 1
                caption = f"[frame {idx} analysis failed: {exc}]"
            parts.append(f"[Frame {idx} ({mm:02d}:{ss:02d})]: {caption}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)  # clean up temp frames
    summary = f"\n\n({len(frames)} frames analyzed, {failed} failed)"
    return "\n\n".join(parts) + summary


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_EXTS


def _build_prompt(question: str) -> str:
    """Build the user prompt: exhaustive-describe-first, then answer the question."""
    q = question.strip() if question else ""
    if q:
        return _DESCRIBE_FIRST.format(question=q)
    return _DEFAULT_QUESTION


def _extract_frames(video_path: Path, max_frames: int = 10) -> list[tuple[Path, float]]:
    """Extract up to max_frames evenly-spaced frames from a video via ffmpeg.

    Returns a list of (frame_path, seconds) tuples. Uses ffprobe to get
    duration, then samples evenly. Frames go to a fresh temp dir.
    """
    ffprobe = _ffprobe_path()
    ffmpeg = _ffmpeg_path()

    # Get duration via ffprobe (blocking call — runs in a thread).
    try:
        dur_probe = subprocess.run(
            [str(ffprobe), "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(dur_probe.stdout.strip())
    except Exception:
        duration = 0.0

    if duration <= 0:
        max_frames = 1  # unknown duration: just grab the first frame

    n = max(1, min(max_frames, int(duration) if duration else 1))
    tmpdir = Path(tempfile.mkdtemp(prefix="vision_frames_"))
    frames: list[tuple[Path, float]] = []
    for i in range(n):
        t = (duration * i / max(1, n - 1)) if duration > 0 and n > 1 else 0.0
        out = tmpdir / f"frame_{i:03d}.png"
        subprocess.run(
            [str(ffmpeg), "-y", "-ss", f"{t:.2f}", "-i", str(video_path),
             "-frames:v", "1", str(out)],
            capture_output=True, timeout=30,
        )
        if out.exists():
            frames.append((out, t))
    return frames


@mcp.tool()
async def analyze_image(image_path: str, question: str = "") -> str:
    """Analyze an image file (or video) with the configured vision model
    (gpt-5.6-luna) and return a detailed description.

    Use this when the current model cannot see the image and the user asks about
    its contents — people present, camera angle, lighting, scene, visible text,
    or any specific detail. For a video, multiple frames are extracted and each
    captioned, stitched into a timeline.

    Args:
        image_path: Absolute path to the image file (or video) to analyze.
        question: Optional specific question to answer about the image. When
            omitted, returns a full detailed caption.
    """
    path = Path(image_path)
    if not path.exists():
        return f"ERROR: file not found: {image_path}"
    if not path.is_file():
        return f"ERROR: path is not a file: {image_path}"

    # Size guard: avoid reading huge files into base64 / blowing memory or the
    # upstream payload limit.
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        return (
            f"ERROR: file too large ({size} bytes > {MAX_FILE_BYTES} limit): {image_path}"
        )

    prompt = _build_prompt(question)

    # Video path: extract frames, caption each, stitch into a timeline.
    if _is_video(path):
        return await _caption_video_timeline(path, prompt)

    # Single image path.
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    media_type = mimetypes.guess_type(image_path)[0] or "image/png"
    source = {"type": "base64", "media_type": media_type, "data": data}

    try:
        return await _caption_image(source, prompt)
    except Exception as exc:
        return f"ERROR: vision analysis failed: {exc}"


@mcp.tool()
async def analyze_image_conversation(
    image_path: str,
    question: str,
    conversation_id: str = "",
    turn_note: str = "",
) -> str:
    """Analyze an image and hold a multi-turn conversation about it with the
    vision model. Reuses the same exhaustive describe-first pipeline as
    analyze_image, but keeps the prior Q&A so follow-up questions are answered
    consistently.

    Start a conversation by omitting conversation_id (or passing "") — the tool
    returns a conversation_id to reuse. Pass that same id on each follow-up to
    continue the thread. Threads are per-process and expire after ~30 min idle or
    10 turns. Pass a short turn_note to anchor this turn (e.g. "user says this is
    a receipt"). Works on images or videos (video frames are extracted and
    stitched into a timeline each turn).

    Args:
        image_path: Absolute path to the image (or video) to discuss.
        question: The question to answer about the image (follow-ups use prior context).
        conversation_id: Omit for a new thread; reuse the returned id to continue.
        turn_note: Optional short context to anchor this turn.
    """
    path = Path(image_path)
    if not path.exists():
        return f"ERROR: file not found: {image_path}"
    if not path.is_file():
        return f"ERROR: path is not a file: {image_path}"

    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        return (
            f"ERROR: file too large ({size} bytes > {MAX_FILE_BYTES} limit): {image_path}"
        )

    _prune_expired_threads()
    video = _is_video(path)
    media_type = None if video else (mimetypes.guess_type(image_path)[0] or "image/png")
    cid, thread, is_new = _get_or_create_thread(conversation_id, image_path, media_type, video)

    if thread.turn_count >= _CONVERSATION_MAX_TURNS:
        return (
            f"ERROR: conversation {cid} has reached the {_CONVERSATION_MAX_TURNS}-turn "
            f"limit. Start a new conversation (omit conversation_id)."
        )

    prompt = _build_conversation_prompt(thread, question, turn_note)

    if video:
        caption = await _caption_video_timeline(
            path,
            prompt,
            extra_messages=(
                None if is_new else _build_prior_turn_messages(thread)
            ),
            use_cache=is_new,
        )
    else:
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        source = {"type": "base64", "media_type": media_type, "data": data}
        try:
            caption = await _caption_turn(
                source,
                prompt,
                extra_messages=(
                    None if is_new else _build_prior_turn_messages(thread)
                ),
                use_cache=is_new,
            )
        except Exception as exc:
            return f"ERROR: vision analysis failed: {exc}"

    thread.turns.append((question, caption))
    thread.notes.append(turn_note)
    thread.turn_count += 1
    thread.last_access = time.monotonic()

    return f"{caption}\n\n[conversation_id={cid}, turn {thread.turn_count}/{_CONVERSATION_MAX_TURNS}]"


@mcp.tool()
async def analyze_images(image_paths: list[str], question: str = "") -> str:
    """Analyze MULTIPLE image files at once with the configured vision model
    (gpt-5.6-luna) and return ONE joint answer.

    Use this when you need a true cross-image comparison or joint question in a
    single round-trip — e.g. "do these show the same person?", "how does the
    scene change between these frames?", "which of these two has text?". ALL
    images are sent to the model together in one message, so it can compare them
    directly (this differs from analyze_image, which handles one image/video).

    Order matters: the images are sent in the order you list them, so reference
    them by position (e.g. "the first image", "image 2 vs image 4") for
    "frame 1 vs frame 3"-style questions.

    Args:
        image_paths: List of absolute paths to image files to analyze together.
        question: Optional specific cross-image question. When omitted, returns a
            joint exhaustive description of all images.
    """
    if not image_paths:
        return "ERROR: no image paths provided"
    if len(image_paths) > MAX_BATCH_IMAGES:
        return (
            f"ERROR: too many images ({len(image_paths)} > {MAX_BATCH_IMAGES} max per batch)"
        )

    total_size = 0
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            return f"ERROR: file not found: {p}"
        if not path.is_file():
            return f"ERROR: path is not a file: {p}"
        if _is_video(path):
            return (
                f"ERROR: video not supported in analyze_images batch: {p} "
                f"(use analyze_image or analyze_image_conversation for videos)"
            )
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            return f"ERROR: file too large ({size} bytes > {MAX_FILE_BYTES} limit): {p}"
        total_size += size
    if total_size > MAX_BATCH_TOTAL_BYTES:
        return (
            f"ERROR: total batch size {total_size} bytes exceeds "
            f"{MAX_BATCH_TOTAL_BYTES} limit"
        )

    prompt = _build_prompt(question)
    sources: list[dict] = []
    for p in image_paths:
        data = base64.b64encode(Path(p).read_bytes()).decode("utf-8")
        media_type = mimetypes.guess_type(p)[0] or "image/png"
        sources.append({"type": "base64", "media_type": media_type, "data": data})

    try:
        return await _caption_batch(sources, prompt)
    except Exception as exc:
        return f"ERROR: vision analysis failed: {exc}"


def main() -> None:
    """Run the MCP vision server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
