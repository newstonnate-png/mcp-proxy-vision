# MCP Vision Tool — Roadmap

## Status

Three tools are live and verified against Luna:

- **`analyze_image`** — single image or video (frames extracted + stitched into a `[Frame N (MM:SS)]` timeline).
- **`analyze_image_conversation`** — multi-turn Q&A about one image/video; per-thread, in-memory (30 min idle TTL, 10-turn cap, 100 threads); prior answers become `assistant` messages.
- **`analyze_images`** — batch of images sent in ONE message for a joint/cross-image answer; 4-image / 25 MB caps, whole-batch cache.

Shared pipeline in the proxy (`vision_fallback._run_caption_call`) carries `extra_messages` (conversation) and `sources` (batch) kwargs — both purely additive.

## Remaining

### 1. Parallelize video frame captioning (highest value)

The video path captions frames **sequentially** (~8.4s/frame → 8 frames ≈ 67s), so long videos can time out. Fix: caption frames with **bounded concurrency** (semaphore) instead of sequential, and/or cap frame count. Applies to `analyze_image` and `analyze_image_conversation` video paths.

This was in the original Phase 2 spec but was set aside when we built the joint `analyze_images` tool instead. Per-image captioning, not joint.

### 2. Phase 2b — Audio transcript alongside video (stretch)

Optionally accept an audio transcript alongside a video to combine visual + audio context.

### 3. Phase 3 — Richer composed questions (low priority)

Guidance (docs or tool descriptions) teaching deepseek to send a single, richly-composed question per probe. Largely superseded by the conversation tool, but still a useful habit.

## Design notes / constraints

- Keep the same `_VISION_SYSTEM_PROMPT` everywhere.
- Conversation turns key by conversation_id, not image+question — never cache-collide with the flat cache.
- Parallel captioning must be bounded (concurrency + frame count) or videos time out.

## Resolved questions

- Thread persistence → per-process only (expire on session end).
- Max turns / TTL → 10 turns, 30 min idle, 100-thread cap.
- Cross-conversation image references → not needed; a fresh thread per question-tree is enough.
- "Summarize the conversation" endpoint → not needed; the 10-turn cap bounds context.
