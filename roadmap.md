# MCP Vision Tool — Roadmap

## Current state

`analyze_image(image_path, question="")` is a **single-shot, single-image** query:

- Deepseek passes one image path + one question.
- Luna returns one caption/answer.
- No back-and-forth, no multi-turn conversation with Luna, no way for deepseek to ask follow-up questions.

The tool shares Luna's system prompt (`_VISION_SYSTEM_PROMPT`, imported from the proxy's `vision_fallback`) and the caption cache with the auto-fallback path, but the **user-message prompt** differs by design:
- Auto-fallback → carries the user's real surrounding text.
- MCP tool → carries deepseek's synthesized `question`, wrapped in a describe-first-then-answer template.

## The gap

**Deepseek cannot ask Luna arbitrary/random questions or hold a real conversation about an image.** It's one image + one question + one answer. This limits how deeply deepseek can understand an image when the user is probing multiple aspects ("what's in the top-left?", "what does the text say?", "what about the lighting?").

## Proposed roadmap

### Phase 1 — Multi-turn conversation tool (planned but not implemented)

Add a conversation-capable tool so deepseek can ask follow-ups about the same image:

- **`analyze_image_conversation(image_path, question, conversation_id="", turn_note="")`**
  - Maintains a per-`conversation_id` thread in memory (image + prior Q&A turns).
  - Each call appends the new question; Luna gets the full prior context + the new question, so it can answer consistently.
  - `conversation_id=""` starts a new thread; reusing an id continues it.
  - Threads expire after a TTL or max turns to bound memory.

### Phase 2 — Multi-image / batch processing

Accept a **batch** of images (e.g. multiple video frames) in one call with an overall question, so deepseek can ask a cross-frame question ("does the scene change?") in one round-trip.

**Motivation:** the current video path captions each frame **sequentially** (~8.4s per frame → 8 frames ≈ 67s), which is slow and can time out. Batch support should:
- Caption frames **in parallel** (bounded concurrency) instead of sequentially, to cut wall-clock time dramatically.
- Accept an explicit list of image paths (not just a video), so deepseek can pass several related images and get a joint answer.
- Return a single stitched timeline/narrative (reusing the existing `[Frame N (MM:SS)]` format).

### Phase 2b — Richer multi-modal input (stretch)

- Optionally accept an audio transcript alongside a video to combine visual + audio context.

### Phase 3 — Deepseek composing richer questions (prompt guidance)

- Guidance (in this project's docs or the tool description) that when the user is probing an image, deepseek should send a **single, richly-composed question** covering all aspects at once — since each tool call is a fresh caption, not a conversation — until Phase 1 lands.

## Design notes / constraints

- Keep the **same `_VISION_SYSTEM_PROMPT`** for consistency across auto-fallback and all tool paths.
- Reuse the **caption cache** where safe; conversation turns that depend on prior turns should **not** be cache-collided (key by conversation_id, not just image+question).
- The describe-first-then-answer wrapper should stay; multi-turn builds on it (prior turns = additional context in the message).
- Video support (frame extraction + timeline stitching) already exists and should compose with conversation (each turn can be video or image).
- **Performance:** captioning is currently sequential (~8.4s/frame), so a batch/video of N frames is ~N×8.4s. Any batch feature must parallelize (bounded concurrency) and/or cap frame count — otherwise videos time out (as seen in testing).

## Open questions for the planning session (another day)

- Should conversation threads persist across Claude Code sessions, or be per-process only?
- Max turns / TTL before a thread resets?
- Should deepseek be able to reference the same image across conversations, or is a fresh thread per question-tree enough?
- Do we want a "summarize the conversation so far" endpoint so deepseek can compact context?
