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

import asyncio
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
        self.sources = None

    async def __call__(self, source, prompt, key=None, model_manager=None,
                       config=None, custom_headers=None, request_id=None, **kwargs):
        self.calls += 1
        self.prompt = prompt
        self.extra_messages = kwargs.get("extra_messages")
        self.sources = kwargs.get("sources")
        return self.caption


class _GaugeFakeCaptionCall(_FakeCaptionCall):
    """Tracks max in-flight concurrency and lets frames finish out of order.

    ``delay_per_call`` cycles through per-call delays so some calls finish before
    others that started earlier, making out-of-order completion observable.
    """

    def __init__(self, delay_per_call=(0.0,), caption_prefix="cap"):
        super().__init__()
        self.delay_per_call = list(delay_per_call)
        self.caption_prefix = caption_prefix
        self.max_inflight = 0
        self.inflight = 0
        self.source_data = []

    async def __call__(self, source, prompt, key=None, model_manager=None,
                       config=None, custom_headers=None, request_id=None, **kwargs):
        self.source_data.append(source["data"])
        delay = self.delay_per_call[min(len(self.source_data) - 1, len(self.delay_per_call) - 1)]
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if delay:
                await asyncio.sleep(delay)
            self.calls += 1
            return f"{self.caption_prefix} {len(self.source_data)}"
        finally:
            self.inflight -= 1


def _make_fake_frames(tmp_path, n, data=b"fake-png-bytes"):
    """Create n fake (frame_path, seconds) pairs WITHOUT ffmpeg.

    Uses a real subdir of tmp_path as the 'temp dir' so the function's
    shutil.rmtree cleanup removes only the files we created. ``data`` may be a
    bytes payload (written to every frame) or a sequence of distinct bytes
    payloads (frame i gets data[i]). Returns the frames list.
    """
    d = tmp_path / "vision_frames_test"
    d.mkdir(exist_ok=True)
    frames = []
    for i in range(n):
        p = d / f"frame_{i:03d}.png"
        payload = data[i] if isinstance(data, (list, tuple)) else data
        p.write_bytes(payload)
        frames.append((p, i * 1.5))  # seconds 0, 1.5, 3.0, ...
    return frames


class _StubKey:
    api_key = "sk-test"
    key_id = "test-key"


class _StubKeyManager:
    def get_best_key(self):
        return _StubKey()


@pytest.fixture(autouse=True)
def _clear_state():
    """Isolate the in-process conversation thread dict AND caption cache between
    tests — the caption cache is keyed by source bytes+prompt, so identical test
    images across tests would collide and poison each other's call counts."""
    import mcp_proxy_vision.server as srv

    srv._THREADS.clear()
    srv._caption_cache.clear()
    yield
    srv._THREADS.clear()
    srv._caption_cache.clear()


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


@pytest.mark.asyncio
async def test_analyze_images_missing_path(monkeypatch, tmp_path):
    """A missing path in the batch returns an ERROR without a vision call."""
    import mcp_proxy_vision.server as srv

    fake = _FakeCaptionCall()
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    out = await srv.analyze_images(
        ["C:/nonexistent/definitely_missing.png"]
    )
    assert "ERROR" in out
    assert "not found" in out
    assert fake.calls == 0


@pytest.mark.asyncio
async def test_analyze_images_two_valid_images_one_call(monkeypatch, tmp_path):
    """Two valid images → ONE call with 2 sources (all in one user message)."""
    import mcp_proxy_vision.server as srv

    img1 = _make_temp_image(tmp_path, "a.png", b"aaaa")
    img2 = _make_temp_image(tmp_path, "b.png", b"bbbb")
    fake = _FakeCaptionCall(caption="joint answer")
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    out = await srv.analyze_images([str(img1), str(img2)], "Do they match?")

    assert out == "joint answer"
    assert fake.calls == 1
    assert fake.sources is not None
    assert len(fake.sources) == 2
    assert all(s["type"] == "base64" for s in fake.sources)
    assert fake.extra_messages is None
    assert "describe everything visible" in fake.prompt


@pytest.mark.asyncio
async def test_analyze_images_invalid_one_of_two(monkeypatch, tmp_path):
    """One invalid member → fail-fast ERROR, no partial vision call."""
    import mcp_proxy_vision.server as srv

    good = _make_temp_image(tmp_path, "good.png", b"good")
    fake = _FakeCaptionCall()
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    out = await srv.analyze_images(
        [str(good), "C:/nonexistent/missing.png"], "compare"
    )
    assert "ERROR" in out
    assert "not found" in out
    assert fake.calls == 0


@pytest.mark.asyncio
async def test_analyze_images_batch_cap(monkeypatch, tmp_path):
    """More than MAX_BATCH_IMAGES images → ERROR, no vision call."""
    import mcp_proxy_vision.server as srv

    imgs = [_make_temp_image(tmp_path, f"img{i}.png", b"x") for i in range(3)]
    fake = _FakeCaptionCall()
    monkeypatch.setattr(srv, "_run_caption_call", fake)
    monkeypatch.setattr(srv, "MAX_BATCH_IMAGES", 2)

    out = await srv.analyze_images([str(p) for p in imgs], "compare")
    assert "ERROR" in out
    assert "too many" in out
    assert fake.calls == 0


@pytest.mark.asyncio
async def test_analyze_images_total_bytes_cap(monkeypatch, tmp_path):
    """Batch total exceeding the bytes budget → ERROR, no vision call."""
    import mcp_proxy_vision.server as srv

    img = _make_temp_image(tmp_path, "a.png", b"123456")
    fake = _FakeCaptionCall()
    monkeypatch.setattr(srv, "_run_caption_call", fake)
    monkeypatch.setattr(srv, "MAX_BATCH_TOTAL_BYTES", 1)

    out = await srv.analyze_images([str(img)], "compare")
    assert "ERROR" in out
    assert "total batch size" in out
    assert fake.calls == 0


@pytest.mark.asyncio
async def test_analyze_images_video_rejected(monkeypatch, tmp_path):
    """A video in the batch → ERROR directing to single-image tools."""
    import mcp_proxy_vision.server as srv

    img = _make_temp_image(tmp_path, "a.png", b"x")
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"not a real video but a real file")  # exists, .mp4 ext
    fake = _FakeCaptionCall()
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    out = await srv.analyze_images([str(img), str(vid)], "compare")
    assert "ERROR" in out
    assert "video not supported" in out
    assert fake.calls == 0


@pytest.mark.asyncio
async def test_analyze_images_reuses_batch_cache(monkeypatch, tmp_path):
    """Two identical batch calls → one underlying call (batch cache hit)."""
    import mcp_proxy_vision.server as srv

    img1 = _make_temp_image(tmp_path, "a.png", b"aaaa")
    img2 = _make_temp_image(tmp_path, "b.png", b"bbbb")
    fake = _FakeCaptionCall(caption="joint answer")
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    paths = [str(img1), str(img2)]
    out1 = await srv.analyze_images(paths, "Do they match?")
    out2 = await srv.analyze_images(paths, "Do they match?")

    assert out1 == out2 == "joint answer"
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_video_timeline_parallel_preserves_order(monkeypatch, tmp_path):
    """Frames caption in parallel but the timeline output stays in frame order."""
    import mcp_proxy_vision.server as srv

    frames = _make_fake_frames(tmp_path, 8)
    fake = _GaugeFakeCaptionCall(delay_per_call=(0.06, 0.02), caption_prefix="cap")
    monkeypatch.setattr(srv, "_extract_frames", lambda path, max_frames=10: frames)
    monkeypatch.setattr(srv, "_run_caption_call", fake)
    monkeypatch.setattr(
        srv, "_build_key_manager",
        lambda: _StubKeyManager(),
    )

    out = await srv._caption_video_timeline(
        tmp_path / "clip.mp4", "Q", use_cache=False
    )

    assert fake.max_inflight >= 2  # real concurrency, not serialized
    # Frame positions strictly ascending in the output.
    positions = [out.index(f"[Frame {i} (") for i in range(1, 9)]
    assert positions == sorted(positions)
    assert "(8 frames analyzed, 0 failed)" in out


@pytest.mark.asyncio
async def test_video_timeline_bounded_concurrency(monkeypatch, tmp_path):
    """At most _VIDEO_MAX_CONCURRENCY calls are in flight, and the bound is hit."""
    import mcp_proxy_vision.server as srv

    frames = _make_fake_frames(tmp_path, 8)
    fake = _GaugeFakeCaptionCall(delay_per_call=(0.03,), caption_prefix="cap")
    monkeypatch.setattr(srv, "_extract_frames", lambda path, max_frames=10: frames)
    monkeypatch.setattr(srv, "_run_caption_call", fake)
    monkeypatch.setattr(srv, "_build_key_manager", lambda: _StubKeyManager())
    monkeypatch.setattr(srv, "_VIDEO_MAX_CONCURRENCY", 2)

    await srv._caption_video_timeline(tmp_path / "clip.mp4", "Q", use_cache=False)

    assert fake.max_inflight <= 2  # bound never exceeded
    assert fake.max_inflight == 2  # bound actually reached with 8 frames


@pytest.mark.asyncio
async def test_video_timeline_error_isolation(monkeypatch, tmp_path):
    """One frame's caption failure doesn't kill the rest of the timeline."""
    import mcp_proxy_vision.server as srv

    frames = _make_fake_frames(tmp_path, 4)

    class _BoomFake(_GaugeFakeCaptionCall):
        async def __call__(self, source, prompt, key=None, model_manager=None,
                           config=None, custom_headers=None, request_id=None, **kwargs):
            self.source_data.append(source["data"])
            if len(self.source_data) == 3:  # frame 3 fails
                raise RuntimeError("boom")
            self.calls += 1
            return f"cap {len(self.source_data)}"

    fake = _BoomFake()
    monkeypatch.setattr(srv, "_extract_frames", lambda path, max_frames=10: frames)
    monkeypatch.setattr(srv, "_run_caption_call", fake)
    monkeypatch.setattr(srv, "_build_key_manager", lambda: _StubKeyManager())

    out = await srv._caption_video_timeline(
        tmp_path / "clip.mp4", "Q", use_cache=False
    )

    assert "cap 1" in out
    assert "cap 2" in out
    assert "[Frame 3 (00:03)] analysis failed: boom" in out
    assert "cap 4" in out
    assert "(4 frames analyzed, 1 failed)" in out


@pytest.mark.asyncio
async def test_video_timeline_output_format(monkeypatch, tmp_path):
    """The stitched timeline format and tmpdir cleanup are correct."""
    import mcp_proxy_vision.server as srv

    frames = _make_fake_frames(tmp_path, 3)
    fake = _FakeCaptionCall(caption="a detailed caption")
    monkeypatch.setattr(srv, "_extract_frames", lambda path, max_frames=10: frames)
    monkeypatch.setattr(srv, "_run_caption_call", fake)
    monkeypatch.setattr(srv, "_build_key_manager", lambda: _StubKeyManager())

    out = await srv._caption_video_timeline(
        tmp_path / "clip.mp4", "Q", use_cache=False
    )

    expected_lines = [
        "[Frame 1 (00:00)]: a detailed caption",
        "[Frame 2 (00:01)]: a detailed caption",
        "[Frame 3 (00:03)]: a detailed caption",
    ]
    assert "\n\n".join(expected_lines) + "\n\n(3 frames analyzed, 0 failed)" == out
    # tmpdir cleaned up after the gather.
    assert not (tmp_path / "vision_frames_test").exists()


@pytest.mark.asyncio
async def test_video_timeline_cache_path_uses_flat_cache(monkeypatch, tmp_path):
    """With use_cache=True, repeated calls hit the flat cache (no re-caption)."""
    import mcp_proxy_vision.server as srv

    # Distinct byte content per frame so each has its own flat-cache key.
    fake = _FakeCaptionCall(caption="cached caption")

    def _fresh_frames(path, max_frames=10):
        # Recreate the frames dir each call (the function's finally rmtree deletes
        # it), with the SAME distinct-byte content so each flat-cache key matches.
        return _make_fake_frames(
            tmp_path, 3,
            data=(b"frame-one", b"frame-two", b"frame-three"),
        )

    monkeypatch.setattr(srv, "_extract_frames", _fresh_frames)
    monkeypatch.setattr(srv, "_run_caption_call", fake)
    monkeypatch.setattr(srv, "_build_key_manager", lambda: _StubKeyManager())

    # use_cache=True → each frame goes through _caption_image (flat cache).
    await srv._caption_video_timeline(tmp_path / "clip.mp4", "Q", use_cache=True)
    await srv._caption_video_timeline(tmp_path / "clip.mp4", "Q", use_cache=True)

    assert fake.calls == 3  # first call captions all 3; second is all cache hits


@pytest.mark.asyncio
async def test_video_timeline_analysis_image_video_integration(monkeypatch, tmp_path):
    """Full analyze_image video path produces a timeline via parallel captioning."""
    import mcp_proxy_vision.server as srv

    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"not a real video but a real file")
    frames = _make_fake_frames(tmp_path, 3)
    fake = _FakeCaptionCall(caption="frame caption")
    monkeypatch.setattr(srv, "_extract_frames", lambda path, max_frames=10: frames)
    monkeypatch.setattr(srv, "_run_caption_call", fake)

    # Stub the key manager so the test is hermetic (no real keys.json needed).
    monkeypatch.setattr(srv, "_build_key_manager", lambda: _StubKeyManager())

    out = await srv.analyze_image(str(vid), "What happens?")

    assert "[Frame 1 (00:00)]: frame caption" in out
    assert "[Frame 3 (00:03)]: frame caption" in out
    assert "(3 frames analyzed, 0 failed)" in out
