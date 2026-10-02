# ComfyUI Trim Video for MiniMax H3

A ComfyUI V3 custom node that prepares long source videos for **MiniMax H3 Reference to Video** workflows that use **FILM VFI at ×2 interpolation**.

The node divides the source video into sequential reference chunks, extracts the source frames required for each chunk, chooses the corresponding MiniMax H3 duration, and exposes both video and image-batch outputs for the downstream workflow.

> This project was **vibe coded**.

## What it does

`Trim Video for MiniMax H3 Interpolated Reference` accepts a ComfyUI `VIDEO` and a zero-based `chunk_index`.

For each execution it:

1. analyzes the active video's timing,
2. respects an upstream trim,
3. determines the effective source frame rate and duration,
4. builds a logical reference timeline at **2× the source sampling rate**,
5. splits that timeline into chunks of at most **345 reference frames**,
6. extracts the source frames needed for the selected chunk,
7. pads the end of the final chunk with held copies of the last frame when required,
8. returns the frames for FILM VFI ×2,
9. calculates the whole-second MiniMax H3 duration for that chunk,
10. reports the total number of chunks/runs required.

The node is intended for a workflow where the interpolated reference clip is played at **24 fps** before being passed to MiniMax H3.

## Fixed behavior

The current version is intentionally built around one specific workflow:

- **FILM interpolation multiplier:** ×2
- **Maximum reference frames per chunk:** 345
- **Reference/H3 playback fps:** 24
- **MiniMax H3 duration range used by the node:** 4–15 seconds
- **Minimum prepared reference length:** 48 frames, or 2 seconds at 24 fps

A full chunk uses:

```text
173 source frames
        ↓
FILM VFI ×2
        ↓
345 interpolated frames
        ↓
Create Video at 24 fps
        ↓
MiniMax H3 Reference to Video
duration_seconds = 14
```

The 345-frame reference clip itself spans **14.375 seconds at 24 fps**. The node supplies **14** to MiniMax H3's duration input because the H3 frame grid for a 14-second request produces 345 output frames.

## Recommended workflow

```text
Load Video
    │
    │ VIDEO
    ▼
Trim Video for MiniMax H3 Interpolated Reference
    │
    ├── frames
    │      │
    │      ▼
    │   FILM VFI
    │   multiplier = 2
    │      │
    │      ▼
    │   Create Video
    │   fps = 24
    │      │
    │      ▼
    │   MiniMax H3 Reference to Video
    │   reference video
    │
    └── duration_seconds ───────────────► MiniMax H3 duration
```

Drive `chunk_index` with an integer node configured to increment, beginning at `0`.

`total_runs` tells you how many executions are needed for the complete input video.

For example:

```text
chunk_index = 0  → first chunk
chunk_index = 1  → second chunk
chunk_index = 2  → third chunk
...
```

If `chunk_index` is past the final available segment, the node stops with an error instead of silently wrapping back to the beginning.

To regenerate a specific chunk, set `chunk_index` directly to that chunk's zero-based index.

## Outputs

| Output | Type | Description |
|---|---|---|
| `trimmed_video` | `VIDEO` | The prepared source-frame chunk as a silent video at the source frame rate. |
| `frames` | `IMAGE` | The same source frames as an image batch, intended for FILM VFI ×2. |
| `duration_seconds` | `INT` | Whole-second duration to connect to MiniMax H3's duration input. |
| `total_runs` | `INT` | Number of chunks required to cover the complete active video. |
| `current_index` | `INT` | The `chunk_index` used for this execution. |
| `segment_info` | `STRING` | JSON containing exact timing, frame windows, padding, H3 information, and reference details. |

## Timing analysis

The node attempts to preserve source timing rather than assuming a common rate such as 24, 30, or 60 fps.

For file-backed video it uses PyAV to inspect the first video stream and works with rational frame-rate and timestamp values internally.

For constant-frame-rate video, the detected rate is kept as an exact rational value.

For variable-frame-rate video, the node creates a constant-rate base sampling timeline using the video's measured or declared average rate and maps source presentation timestamps onto that timeline.

The active ComfyUI trim window is respected, so an upstream trim is analyzed instead of accidentally using the complete underlying file.

If reliable timing cannot be established, the node returns an error rather than inventing a frame rate.

## Chunk calculation

Let:

```text
f = effective source sampling rate
F = f × 2
T = active video duration
```

The planned interpolated reference-frame count for the whole video is:

```text
N = ceil(T × F)
```

The number of required runs is:

```text
total_runs = ceil(N / 345)
```

Each full chunk therefore represents:

```text
345 / F seconds
```

of source motion.

Examples:

| Source fps | Reference sampling fps after ×2 | Source motion represented by a full 345-frame chunk |
|---:|---:|---:|
| 24 | 48 | 7.1875 s |
| 30 | 60 | 5.75 s |
| 60 | 120 | 2.875 s |

## Final chunk and padding

The final chunk can contain fewer than 345 logical reference frames.

MiniMax H3 accepts whole-second duration values and its output frame counts follow its internal frame grid. The node chooses the smallest supported whole-second duration whose H3 frame count covers the planned chunk.

Very short final chunks are padded to a minimum reference length of **48 frames** by repeating the final source frame. This is intended to avoid reference clips that are too short for H3.

The padding belongs to the reference clip only. This version does **not** automatically trim the generated H3 result back to the original logical source duration.

## FILM frame-count assumption

This node is designed for an endpoint-retaining FILM ×2 implementation with:

```text
n source frames → 2n - 1 interpolated frames
```

That is why a full chunk uses:

```text
173 source frames → 345 FILM frames
```

The implementation was built for the behavior used by Fannovel16's ComfyUI Frame Interpolation FILM workflow.

For some short final chunks, the requested H3 length can be even. With the `2n - 1` FILM rule, the prepared reference batch can then contain one additional frame. The node reports this in `segment_info`.

## Known limitations

### Approximate chunk boundaries

Chunk boundaries are currently approximate.

A chunk begins at the source frame at or immediately before its exact logical start. When an exact boundary falls between source frames, neighboring chunks can therefore repeat or skip one interpolated sample at their join.

This does not affect the node's chunk-count planning, but it means this version does not guarantee a mathematically gap-free interpolated timeline across chunk boundaries.

### FILM implementation dependency

The frame calculations assume an endpoint-retaining FILM implementation:

```text
n → 2n - 1
```

A different FILM wrapper with different output-count behavior may not match the node's calculations.

### Final reference padding

Held copies of the final source frame may be appended to make the final reference clip long enough for H3.

The generated H3 output is not automatically cropped afterward.

### Manual chunk progression

This version uses the visible `chunk_index` input. It does not maintain its own persistent internal run counter.

If a downstream node fails after this node has executed, simply set or restore the desired `chunk_index` and run that chunk again.

## Installation

### Option A: clone the repository

Clone this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/oi0lo/ComfyUI-Trim-Video-MiniMax-H3.git
```

Restart ComfyUI.

The included `__init__.py` exposes the V3 `comfy_entrypoint` while leaving the original node source unchanged.

### Option B: install the single file

You can also copy:

```text
oi0lo_trim_video_minimax_h3.py
```

directly into:

```text
ComfyUI/custom_nodes/
```

and restart ComfyUI.

The node appears under:

```text
video
└── Trim Video for MiniMax H3 Interpolated Reference
```

## Requirements

- A current ComfyUI build with the V3 custom-node API and core `VIDEO` support
- PyTorch, as provided by ComfyUI
- PyAV / `av`, used for video probing and decoding
- A FILM VFI node configured for **×2 interpolation**
- MiniMax H3 Reference to Video for the intended downstream workflow

No model weights are included in this repository.

## Node inputs

### `video`

A ComfyUI `VIDEO`, typically from **Load Video**.

### `chunk_index`

Zero-based index of the chunk to output.

Start at `0`. For automated sequential generation, connect an integer node configured to increment after each generation.

## Diagnostics

The node displays information such as:

```text
Total workflow runs required: 10
H3 duration for this run: 14 s (345 frames)

Run: 1 of 10   chunk_index 0
Reference: 173 source frames → 345 after FILM ×2
Source segment: 0.00–5.75 s
Source fps: 30   Reference fps: 60
```

The `segment_info` output exposes substantially more detail as JSON, including:

- exact source fps,
- source duration,
- constant or variable frame-rate status,
- selected frame window,
- logical source segment,
- planned reference-frame count,
- expected FILM output count,
- held-frame padding,
- MiniMax H3 duration,
- MiniMax H3 output frame count.

## Repository contents

```text
ComfyUI-Trim-Video-MiniMax-H3/
├── __init__.py
├── oi0lo_trim_video_minimax_h3.py
├── README.md
├── requirements.txt
└── .gitignore
```

The actual custom-node implementation is kept in `oi0lo_trim_video_minimax_h3.py`.

## Status

This is a purpose-built workflow utility rather than a general video-splitting node. The fixed FILM ×2 and MiniMax H3 assumptions are deliberate.

If you use a different interpolation multiplier, interpolation implementation, or downstream model, verify the frame-count behavior before relying on the calculated chunk boundaries and durations.