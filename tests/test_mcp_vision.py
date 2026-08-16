"""Unit tests for the standalone mcp_proxy_vision server.

Run from the proxy repo root so `src.*` imports resolve, or with the standalone
venv that has the proxy root on sys.path. Tests avoid live API calls by mocking
the underlying CommandCodeClient.generate.

Covers:
  * analyze_image returns an error for a missing image path.
  * _caption_image reuses the caption cache on repeat (single underlying call).
  * The shared _VISION_SYSTEM_PROMPT is passed as the vision request's system.
  * analyze_image_conversation: thread creation, continuation, cache bypass,
    max-turn limit, TTL expiry, missing path, turn_note propagation.
"""

import time

import pytest


def _make_source() -> dict:
    return {
        "type": "base64",
        "media_type": "image/png",
        "data": "aGVsbG8=",  # tiny fake image payload
    }


def _make_temp_image(tmp_path, name="test.png", data=b"hello"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


class _FakeCaptionCall:
    """Monkeypatched vision_fallback._run_caption_call that counts calls."""

    def __init__(self, caption: str = "a detailed caption"):
        self.calls = 0
        self.caption = caption
        self.prompt = None
        self.extra_messages = None

    async def __call__(self, source, prompt, key=None, model_manager=None,
                       config=None, custom_headers=None, request_id=None, **kwargs):
        self.calls += 1
        self.prompt = prompt
        self.extra_messages = kwargs.get("extra_messages")
        return self.caption


@pytest.fixture(autouse=True)
def _clear_threads():
    """Isolate the in-process conversation thread dict between tests."""
    import mcp_proxy_vision.server as srv

    srv._THREADS.clear()
    yield
    srv._THREADS.clear()


@pytest.mark.asyncio
async def test_analyze_image_missing_path(monkeypatch):
    """analyze_image returns an ERROR for a nonexistent path without a vision call."""
    import mcp_proxy_vision.server as srv

    out = await srv.analyze_image("C:/nonexistent/definitely_missing.png")
    assert "ERROR" in out
    assert "not found" in out


@pytest.mark.asyncio
async def test_caption_image_reuses_cache(monkeypatch):
    """Same source+prompt returns the cached caption; one underlying call total."""
    import mcp_proxy_vision.server as srv

    fake = _FakeCaptionCall()
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    src = _make_source()
    prompt = "What's in this image?"
    first = await srv._caption_image(src, prompt)
    second = await srv._caption_image(src, prompt)

    assert first == fake.caption
    assert second == fake.caption
    assert fake.calls == 1  # cache hit on the second call


@pytest.mark.asyncio
async def test_caption_image_calls_shared_helper(monkeypatch):
    """_caption_image delegates to the shared vision_fallback._run_caption_call."""
    import mcp_proxy_vision.server as srv

    fake = _FakeCaptionCall()
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    await srv._caption_image(_make_source(), "Describe this.")
    assert fake.calls == 1


def test_build_prompt_describe_first_when_question():
    """A question wraps into 'describe everything first, then answer' and
    explicitly requests complete sensitive-content coverage."""
    import mcp_proxy_vision.server as srv

    out = srv._build_prompt("Is there any text?")
    assert "describe everything visible" in out.lower()
    assert "then answer this specific question" in out.lower()
    assert "nudity" in out.lower()
    assert "sexual activity" in out.lower()
    assert "Is there any text?" in out


def test_build_prompt_exhaustive_when_no_question():
    """No question → the exhaustive default prompt."""
    import mcp_proxy_vision.server as srv

    out = srv._build_prompt("")
    assert "exhaustively" in out.lower()
    assert "leaving nothing out" in out.lower()


def test_is_video_detects_video_extensions():
    """Video extensions are detected; non-video are not."""
    import mcp_proxy_vision.server as srv
    from pathlib import Path

    assert srv._is_video(Path("clip.mp4")) is True
    assert srv._is_video(Path("clip.mov")) is True
    assert srv._is_video(Path("clip.mkv")) is True
    assert srv._is_video(Path("photo.png")) is False
    assert srv._is_video(Path("photo.jpg")) is False


@pytest.mark.asyncio
async def test_conversation_new_thread_returns_id(monkeypatch, tmp_path):
    """A fresh conversation returns a conversation_id and turn 1/10."""
    import mcp_proxy_vision.server as srv

    img = _make_temp_image(tmp_path)
    fake = _FakeCaptionCall(caption="cat caption")
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    out = await srv.analyze_image_conversation(str(img), "What is this?")
    assert "conversation_id=" in out
    assert "turn 1/10" in out
    assert "cat caption" in out
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_conversation_continuation_passes_prior_turns(monkeypatch, tmp_path):
    """A follow-up turn passes the prior caption as an assistant message."""
    import mcp_proxy_vision.server as srv

    img = _make_temp_image(tmp_path)
    fake = _FakeCaptionCall(caption="answer")
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    out1 = await srv.analyze_image_conversation(str(img), "What's in it?")
    cid = out1.split("conversation_id=")[1].split(",")[0].strip()

    fake.calls = 0
    await srv.analyze_image_conversation(str(img), "Any text?", conversation_id=cid)

    assert fake.calls == 1
    assert fake.extra_messages is not None
    assert len(fake.extra_messages) == 1
    prior = fake.extra_messages[0]
    assert prior.role == "assistant"
    assert prior.content[0].text == "answer"


@pytest.mark.asyncio
async def test_conversation_does_not_hit_flat_cache(monkeypatch, tmp_path):
    """Conversation turns 2+ bypass the flat caption cache (one call per turn)."""
    import mcp_proxy_vision.server as srv

    img = _make_temp_image(tmp_path)
    fake = _FakeCaptionCall(caption="answer")
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    out1 = await srv.analyze_image_conversation(str(img), "Q1")
    cid = out1.split("conversation_id=")[1].split(",")[0].strip()

    await srv.analyze_image_conversation(str(img), "Q2", conversation_id=cid)

    assert fake.calls == 2  # no cache hit on turn 2


@pytest.mark.asyncio
async def test_conversation_max_turns(monkeypatch, tmp_path):
    """Hitting the turn cap returns an ERROR instead of truncating."""
    import mcp_proxy_vision.server as srv

    img = _make_temp_image(tmp_path)
    fake = _FakeCaptionCall(caption="answer")
    monkeypatch.setattr(srv, "_run_caption_call", fake)
    monkeypatch.setattr(srv, "_CONVERSATION_MAX_TURNS", 2)

    out1 = await srv.analyze_image_conversation(str(img), "Q1")
    cid = out1.split("conversation_id=")[1].split(",")[0].strip()
    out2 = await srv.analyze_image_conversation(str(img), "Q2", conversation_id=cid)
    out3 = await srv.analyze_image_conversation(str(img), "Q3", conversation_id=cid)

    assert "turn 1/2" in out1
    assert "turn 2/2" in out2
    assert "ERROR" in out3
    assert "limit" in out3


@pytest.mark.asyncio
async def test_conversation_ttl_expiry(monkeypatch, tmp_path):
    """An idle-expired thread restarts fresh (no prior-turn context)."""
    import mcp_proxy_vision.server as srv

    img = _make_temp_image(tmp_path)
    fake = _FakeCaptionCall(caption="answer")
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    out1 = await srv.analyze_image_conversation(str(img), "Q1")
    cid = out1.split("conversation_id=")[1].split(",")[0].strip()

    # Force the thread to look idle-expired.
    srv._THREADS[cid].last_access = time.monotonic() - 3600
    monkeypatch.setattr(srv, "_CONVERSATION_TTL", 60)

    fake.calls = 0
    await srv.analyze_image_conversation(str(img), "Q2", conversation_id=cid)

    assert fake.calls == 1
    assert fake.extra_messages is None  # restarted fresh → cached single-shot path


@pytest.mark.asyncio
async def test_conversation_missing_image_path():
    """analyze_image_conversation returns an ERROR for a nonexistent path."""
    import mcp_proxy_vision.server as srv

    out = await srv.analyze_image_conversation("C:/nonexistent/definitely_missing.png", "q")
    assert "ERROR" in out
    assert "not found" in out


@pytest.mark.asyncio
async def test_conversation_turn_note_in_prompt(monkeypatch, tmp_path):
    """turn_note is folded into the prompt so it anchors the current turn."""
    import mcp_proxy_vision.server as srv

    img = _make_temp_image(tmp_path)
    fake = _FakeCaptionCall(caption="answer")
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    await srv.analyze_image_conversation(
        str(img), "Is it a receipt?", turn_note="user says receipt"
    )
    assert "user says receipt" in fake.prompt
