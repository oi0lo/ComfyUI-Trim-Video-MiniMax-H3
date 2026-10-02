"""
Trim Video for MiniMax H3 Interpolated Reference
================================================

Single-file ComfyUI custom node (V3 API).
Install: copy this file into ComfyUI/custom_nodes/ and restart ComfyUI.
Menu: video > Trim Video for MiniMax H3 Interpolated Reference

Wiring
------
    Load Video -> video
    Int node (control after generate: increment, start at 0) -> chunk_index
    frames -> FILM VFI (multiplier 2) -> Create Video (fps 24)
           -> MiniMax H3 Reference to Video (reference video)
    duration_seconds -> MiniMax H3 Reference to Video (duration)

Each run outputs one chunk of the video. total_runs says how many runs the
whole video needs. chunk_index 0 is the first chunk. Past the last chunk the
node stops with an error, so queue exactly total_runs runs, or let the error
end the batch. Set the Int node back to 0 for the next video. To redo one
chunk, type its index.

Fixed settings
--------------
FILM x2. Each chunk covers at most 345 reference frames after interpolation,
which is 14.375 s when played at 24 fps, and asks H3 for 14 s (345 frames).
A shorter final chunk asks for the fewest whole seconds whose H3 length
covers it. A final chunk shorter than 2 s is padded with copies of the last
frame, because H3 rejects reference videos under 1.8 s (H3 Max: 2 s).

Known limits of this version
----------------------------
* Boundaries are approximate. A chunk starts at the source frame at or before
  its exact start time, so a join between chunks can repeat or skip one
  interpolated sample.
* The frame counts assume a FILM node that keeps both end frames:
  n source frames -> 2n - 1 frames (true for Fannovel16's FILM VFI).
* When the H3 length for a chunk is even (short final chunks only), FILM
  returns one frame more than that length. The reference clip is then 1/24 s
  longer; H3 takes the reference as a clip, so no frame count has to match.
* The last chunk's reference ends with held copies of the last frame
  (padding). Generated H3 output is not trimmed back to the source length.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction

import av
import torch

from comfy_api.latest import ComfyExtension, IO, InputImpl, Types

try:
    from server import PromptServer
except Exception:  # imported outside a running ComfyUI server
    PromptServer = None


NODE_ID = "TrimVideoForMiniMaxH3InterpolatedReference"
DISPLAY_NAME = "Trim Video for MiniMax H3 Interpolated Reference"

MULTIPLIER = 2               # FILM x2, fixed in this version
MAX_REFERENCE_FRAMES = 345   # per chunk, after interpolation
H3_FPS = 24                  # H3 plays reference frames and renders output at 24 fps
H3_MIN_SECONDS = 4           # MiniMax H3 duration range (H3 Max starts at 5)
H3_MAX_SECONDS = 15
MIN_REFERENCE_FRAMES = 48    # 2.0 s at 24 fps; H3 rejects references under 1.8 s (Max: 2.0 s)

COLOR_SPACES = ("sRGB", "HDR", "HDR PQ")
LOG_PREFIX = "[TrimVideoMiniMaxH3]"


# ---------------------------------------------------------------------------
# MiniMax H3 length rules
# ---------------------------------------------------------------------------

def h3_grid_length(frames: int) -> int:
    """Smallest length on H3's 17k + 5 grid that covers `frames` (minimum 5)."""
    n = max(5, int(frames))
    return n + ((5 - n % 17) % 17)


def h3_frames_for_seconds(seconds: int) -> int:
    """Frames H3 renders for a whole-second duration (4 s -> 107, 14 s -> 345, 15 s -> 362)."""
    return h3_grid_length(seconds * H3_FPS)


def h3_seconds_covering(frames: int) -> int:
    """Fewest whole seconds whose H3 length is at least `frames`."""
    for seconds in range(H3_MIN_SECONDS, H3_MAX_SECONDS + 1):
        if h3_frames_for_seconds(seconds) >= frames:
            return seconds
    raise ValueError(f"{frames} frames exceed the H3 maximum of {H3_MAX_SECONDS} s.")


# ---------------------------------------------------------------------------
# Chunk table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkPlan:
    index: int
    total_runs: int
    planned_total: int        # N: reference frames for the whole video
    reference_start: int      # first reference index of this chunk
    planned_frames: int       # reference frames this chunk stands for
    frame_count: int          # H3 grid length, including padding
    duration_seconds: int     # value for the H3 duration input
    h3_output_frames: int     # frames H3 renders at that duration
    source_first: int         # first base-timeline frame supplied to FILM
    source_count: int
    source_last: int
    expected_film_frames: int
    logical_start: Fraction   # seconds on the source timeline
    logical_end: Fraction


def table_size(duration: Fraction, rate: Fraction) -> tuple[int, int]:
    """(planned reference frames N, total runs) for a video of `duration` s at `rate` fps."""
    planned_total = math.ceil(duration * rate * MULTIPLIER)  # exact: Fraction ceil
    if planned_total <= 0:
        raise ValueError("The video has no duration.")
    total_runs = (planned_total + MAX_REFERENCE_FRAMES - 1) // MAX_REFERENCE_FRAMES
    return planned_total, total_runs


def plan_chunk(index: int, duration: Fraction, rate: Fraction) -> ChunkPlan:
    planned_total, total_runs = table_size(duration, rate)
    if not 0 <= index < total_runs:
        raise IndexError(index)
    reference_rate = rate * MULTIPLIER
    reference_start = index * MAX_REFERENCE_FRAMES
    planned = min(MAX_REFERENCE_FRAMES, planned_total - reference_start)
    frame_count = h3_grid_length(max(planned, MIN_REFERENCE_FRAMES))
    seconds = h3_seconds_covering(frame_count)
    source_first = reference_start // MULTIPLIER
    source_count = (frame_count + 2) // 2  # ceil((frame_count + 1) / 2) for FILM x2
    return ChunkPlan(
        index=index,
        total_runs=total_runs,
        planned_total=planned_total,
        reference_start=reference_start,
        planned_frames=planned,
        frame_count=frame_count,
        duration_seconds=seconds,
        h3_output_frames=h3_frames_for_seconds(seconds),
        source_first=source_first,
        source_count=source_count,
        source_last=source_first + source_count - 1,
        expected_film_frames=MULTIPLIER * (source_count - 1) + 1,
        logical_start=Fraction(reference_start) / reference_rate,
        logical_end=min(duration, Fraction(reference_start + planned) / reference_rate),
    )


# ---------------------------------------------------------------------------
# Video timing analysis
# ---------------------------------------------------------------------------

@dataclass
class Timing:
    """Timing of the active video, on one constant-rate base timeline."""
    constant_rate: bool
    rate: Fraction            # base sampling rate f
    rate_source: str
    duration: Fraction        # T in seconds, including the last frame's display time
    frame_total: int          # frames in the active video
    base_total: int           # ceil(T * f): samples on the base timeline
    index_method: str         # "demux", "decoder" or "in-memory"
    last_frame_duration_source: str
    time_base: Fraction | None = None
    pts: list[int] | None = None                # sorted presentation timestamps
    keyframes: list[bool] | None = None
    sample_to_frame: list[int] | None = None    # None means identity


class _NeedsDecoderIndex(Exception):
    """The packet index cannot be trusted; rebuild it from decoded frames."""


class _FrameCountMismatch(Exception):
    pass


def _fraction(value) -> Fraction | None:
    if value is None:
        return None
    value = Fraction(value)
    return value if value > 0 else None


def _scan(source, use_decoder: bool):
    """Collect (pts, duration, keyframe) for every frame of the first video stream."""
    if isinstance(source, io.BytesIO):
        source.seek(0)
    entries = []
    missing_pts = 0
    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("The input has no video stream.")
        stream = container.streams.video[0]
        meta = {
            "time_base": _fraction(stream.time_base),
            "average_rate": _fraction(stream.average_rate),
            "base_rate": _fraction(stream.base_rate),
            "guessed_rate": _fraction(stream.guessed_rate),
            "stream_start": stream.start_time,
            "stream_duration": stream.duration,
        }
        if use_decoder:
            stream.codec_context.thread_type = "AUTO"
            for packet in container.demux(stream):
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError:
                    continue  # ComfyUI's own decoder skips these packets too
                for frame in frames:
                    if frame.pts is None:
                        missing_pts += 1
                        continue
                    entries.append((frame.pts, frame.duration or 0, bool(frame.key_frame)))
        else:
            for packet in container.demux(stream):
                if packet.pts is None:
                    if packet.dts is not None or packet.size:
                        missing_pts += 1
                    continue  # the empty flush packet at the end
                if packet.is_discard:
                    continue  # edit-list preroll: decoded but never shown
                entries.append((packet.pts, packet.duration or 0, bool(packet.is_keyframe)))
    return meta, entries, missing_pts


def _is_constant_rate(rel_ticks: list[int], time_base: Fraction, rate: Fraction) -> bool:
    """Every frame k starts within max(1 tick, a quarter frame) of k / rate."""
    a, b = time_base.numerator, time_base.denominator
    c, d = rate.numerator, rate.denominator
    # |t*a/b - k*d/c| <= max(a/b, d/(4c)), scaled by 4bc to stay in integers
    limit = max(4 * a * c, d * b)
    for k, t in enumerate(rel_ticks):
        if abs(4 * (t * a * c - k * d * b)) > limit:
            return False
    return True


def _nearest_start_map(rel_ticks: list[int], time_base: Fraction, rate: Fraction, base_total: int) -> list[int]:
    """Sample j shows the last frame that starts before (j + 1/2) / rate.

    This is the rule of ffmpeg's fps filter with round=near.
    """
    a, b = time_base.numerator, time_base.denominator
    c, d = rate.numerator, rate.denominator
    starts = [2 * c * a * t for t in rel_ticks]  # t_k scaled by 2bc
    mapping = []
    k = 0
    last = len(starts) - 1
    for j in range(base_total):
        bound = (2 * j + 1) * d * b
        while k < last and starts[k + 1] < bound:
            k += 1
        mapping.append(k)
    return mapping


def _file_timing(video, use_decoder: bool) -> Timing:
    source = video.get_stream_source()
    start_time, trim_duration = video.get_active_trim_window()
    meta, entries, missing_pts = _scan(source, use_decoder)
    tb = meta["time_base"]
    if tb is None:
        raise ValueError("The video stream has no time base. Remux it to MP4 or MKV and try again.")
    if missing_pts:
        if not use_decoder:
            raise _NeedsDecoderIndex()
        raise ValueError(
            f"{missing_pts} frames have no presentation timestamp, so the frame rate cannot be "
            "established. Remux or re-encode the video to MP4 and try again."
        )

    # Same window rule as ComfyUI's VideoFromFile decoder, so upstream trims line up.
    start_pts = int(start_time / tb)
    end_pts = int((start_time + trim_duration) / tb) if trim_duration else None
    entries = sorted(e for e in entries if e[0] >= start_pts and (end_pts is None or e[0] < end_pts))
    if not entries:
        raise ValueError("The video has no decodable frames (after any upstream trim).")
    pts = [e[0] for e in entries]
    if len(set(pts)) != len(pts):
        if not use_decoder:
            raise _NeedsDecoderIndex()
        raise ValueError("Several frames share one timestamp. Re-encode the video to MP4 and try again.")

    rel = [p - pts[0] for p in pts]
    frame_total = len(pts)
    keyframes = [e[2] for e in entries]
    method = "decoder" if use_decoder else "demux"

    # Constant frame rate: a declared rate (or the timestamp span) must explain every timestamp.
    candidates = []
    for name in ("average_rate", "base_rate", "guessed_rate"):
        rate = meta[name]
        if rate is not None and all(rate != other for _, other in candidates):
            candidates.append((name, rate))
    if frame_total >= 2 and rel[-1] > 0:
        span_rate = Fraction(frame_total - 1) / (rel[-1] * tb)
        if all(span_rate != other for _, other in candidates):
            candidates.append(("timestamp span", span_rate))
    if frame_total == 1 and not candidates:
        raise ValueError("A single-frame video has no frame rate. Load it as an image instead.")
    for name, rate in candidates:
        if _is_constant_rate(rel, tb, rate):
            return Timing(
                constant_rate=True,
                rate=rate,
                rate_source=name,
                duration=Fraction(frame_total) / rate,
                frame_total=frame_total,
                base_total=frame_total,
                index_method=method,
                last_frame_duration_source="1 / frame rate (constant frame rate)",
                time_base=tb,
                pts=pts,
                keyframes=keyframes,
            )

    # Variable frame rate: the last frame's display time must be known from the stream.
    last_ticks = entries[-1][1]
    last_source = "frame duration"
    if not last_ticks and meta["stream_duration"] and meta["stream_start"] is not None:
        last_ticks = meta["stream_start"] + meta["stream_duration"] - pts[-1]
        last_source = "video stream end"
    if end_pts is not None and last_ticks > 0:
        last_ticks = min(last_ticks, end_pts - pts[-1])
        last_source += ", capped at the upstream trim"
    if last_ticks <= 0:
        raise ValueError(
            "Variable frame rate video, and the display time of its last frame is unknown. "
            "Convert it to a constant frame rate first, for example: "
            "ffmpeg -i input -vf fps=30 -c:v libx264 -crf 12 output.mp4"
        )
    duration = Fraction(rel[-1] + last_ticks) * tb
    measured = Fraction(frame_total) / duration
    declared = meta["average_rate"]
    if declared is not None and abs(declared - measured) <= measured / 100:
        rate, rate_source = declared, "average_rate (variable frame rate, normalized)"
    else:
        rate, rate_source = measured, "frames / duration (variable frame rate, normalized)"
    base_total = math.ceil(duration * rate)
    return Timing(
        constant_rate=False,
        rate=rate,
        rate_source=rate_source,
        duration=duration,
        frame_total=frame_total,
        base_total=base_total,
        index_method=method,
        last_frame_duration_source=last_source,
        time_base=tb,
        pts=pts,
        keyframes=keyframes,
        sample_to_frame=_nearest_start_map(rel, tb, rate, base_total),
    )


_CACHE: OrderedDict = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_SIZE = 8


def _source_key(video):
    source = video.get_stream_source()
    trim = video.get_active_trim_window()
    if isinstance(source, (str, os.PathLike)):
        path = os.path.realpath(os.fspath(source))
        stat = os.stat(path)
        return ("file", path, stat.st_size, stat.st_mtime_ns, trim)
    if isinstance(source, io.BytesIO):
        with source.getbuffer() as view:
            digest = hashlib.blake2b(view, digest_size=16).hexdigest()
        return ("bytes", digest, trim)
    return None


def _cached_file_timing(video, need_decoder: bool) -> Timing:
    key = _source_key(video)
    with _CACHE_LOCK:
        cached = _CACHE.get(key) if key is not None else None
        if cached is not None and (not need_decoder or cached.index_method == "decoder"):
            _CACHE.move_to_end(key)
            return cached
        timing = None
        if not need_decoder:
            try:
                timing = _file_timing(video, use_decoder=False)
            except _NeedsDecoderIndex:
                timing = None
        if timing is None:
            timing = _file_timing(video, use_decoder=True)
        if key is not None:
            _CACHE[key] = timing
            _CACHE.move_to_end(key)
            while len(_CACHE) > _CACHE_SIZE:
                _CACHE.popitem(last=False)
        return timing


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _base_samples(plan: ChunkPlan, base_total: int) -> list[int]:
    """Base-timeline samples for this chunk; past the end, repeat the last one."""
    last = base_total - 1
    return [min(j, last) for j in range(plan.source_first, plan.source_last + 1)]


def _decode_span(video, timing: Timing, k_first: int, k_last: int) -> torch.Tensor:
    """Frames k_first..k_last, decoded by ComfyUI's own VideoFromFile code path.

    Using as_trimmed() + get_components() keeps rotation, crop, color conversion
    and bit-depth handling identical to Load Video -> Get Video Components.
    The window is centred on timestamp ticks so rounding cannot shift it.
    """
    tb = timing.time_base
    active_start, _ = video.get_active_trim_window()
    start = (Fraction(timing.pts[k_first]) + Fraction(1, 2)) * tb
    span = Fraction(timing.pts[k_last] - timing.pts[k_first] + 1) * tb
    window = video.as_trimmed(float(start) - active_start, float(span), strict_duration=False)
    if window is None:
        raise ValueError(f"Could not open source frames {k_first}-{k_last}.")
    return window.get_components().images


def _decode_frames(video, timing: Timing, frames_needed: list[int]) -> torch.Tensor:
    k_first, k_last = min(frames_needed), max(frames_needed)
    expected = k_last - k_first + 1
    images = _decode_span(video, timing, k_first, k_last)
    offset = k_first
    if images.shape[0] != expected:
        # A seek can land after the first wanted frame; start from the previous keyframe.
        got = images.shape[0]
        images = None
        k_key = next((k for k in range(k_first - 1, -1, -1) if timing.keyframes[k]), None)
        if k_key is not None:
            retry = _decode_span(video, timing, k_key, k_last)
            if retry.shape[0] == k_last - k_key + 1:
                images, offset = retry, k_key
        if images is None:
            raise _FrameCountMismatch(
                f"Decoded {got} frames where the index expects {expected} "
                f"(source frames {k_first}-{k_last})."
            )
    positions = [k - offset for k in frames_needed]
    if positions == list(range(images.shape[0])):
        return images
    return images.index_select(0, torch.tensor(positions, dtype=torch.long, device=images.device))


def _prepare_file(video, chunk_index: int):
    need_decoder = False
    while True:
        timing = _cached_file_timing(video, need_decoder)
        plan = _plan_or_stop(chunk_index, timing)
        samples = _base_samples(plan, timing.base_total)
        frames_needed = samples if timing.sample_to_frame is None else [timing.sample_to_frame[j] for j in samples]
        try:
            return timing, plan, _decode_frames(video, timing, frames_needed)
        except _FrameCountMismatch as err:
            if timing.index_method == "decoder":
                raise ValueError(
                    f"{err} The file may be damaged or use an unusual frame structure. "
                    "Re-encode it to MP4 (H.264) and try again."
                ) from None
            logging.warning("%s %s Rebuilding the frame index by decoding.", LOG_PREFIX, err)
            need_decoder = True


def _prepare_memory(video, chunk_index: int):
    components = video.get_components()
    images = components.images
    frame_total = int(images.shape[0])
    if frame_total == 0:
        raise ValueError("The video has no frames.")
    rate = _fraction(components.frame_rate)
    if rate is None:
        raise ValueError("The video has no valid frame rate.")
    timing = Timing(
        constant_rate=True,
        rate=rate,
        rate_source="declared by the in-memory video",
        duration=Fraction(frame_total) / rate,
        frame_total=frame_total,
        base_total=frame_total,
        index_method="in-memory",
        last_frame_duration_source="1 / frame rate (in-memory video)",
    )
    plan = _plan_or_stop(chunk_index, timing)
    samples = _base_samples(plan, timing.base_total)
    return timing, plan, images.index_select(0, torch.tensor(samples, dtype=torch.long, device=images.device))


def _plan_or_stop(chunk_index: int, timing: Timing) -> ChunkPlan:
    _, total_runs = table_size(timing.duration, timing.rate)
    if chunk_index >= total_runs:
        raise ValueError(
            f"All segments have been output. This video needs {total_runs} runs "
            f"(chunk_index 0 to {total_runs - 1}), and chunk_index is {chunk_index}. "
            "Set your Int node back to 0 to start the next video."
        )
    return plan_chunk(chunk_index, timing.duration, timing.rate)


# ---------------------------------------------------------------------------
# Display and info
# ---------------------------------------------------------------------------

def _exact(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _rate_text(rate: Fraction) -> str:
    if rate.denominator == 1:
        return str(rate.numerator)
    if rate.denominator > 100000:
        return f"{float(rate):.6g}"
    return f"{float(rate):.6g} ({rate.numerator}/{rate.denominator})"


def _details(timing: Timing, plan: ChunkPlan, frames: torch.Tensor) -> tuple[str, dict]:
    held = max(0, plan.source_last - (timing.base_total - 1))
    real = plan.source_count - held
    film_from_video = MULTIPLIER * (real - 1) + 1
    padding = plan.expected_film_frames - film_from_video
    reference_rate = timing.rate * MULTIPLIER

    held_text = f" ({held} repeat the last frame)" if held else ""
    timing_text = "" if timing.constant_rate else " (variable, normalized)"
    display = "\n".join([
        f"Total workflow runs required: {plan.total_runs}",
        f"H3 duration for this run: {plan.duration_seconds} s ({plan.h3_output_frames} frames)",
        "",
        f"Run: {plan.index + 1} of {plan.total_runs}   chunk_index {plan.index}",
        f"Reference: {plan.source_count} source frames{held_text} → {plan.expected_film_frames} after FILM ×2",
        f"Source segment: {float(plan.logical_start):.2f}–{float(plan.logical_end):.2f} s",
        f"Source fps: {_rate_text(timing.rate)}{timing_text}   Reference fps: {_rate_text(reference_rate)}",
    ])

    info = {
        "chunk_index": plan.index,
        "run": plan.index + 1,
        "total_runs": plan.total_runs,
        "h3": {
            "duration_seconds": plan.duration_seconds,
            "output_frames": plan.h3_output_frames,
            "fps": H3_FPS,
        },
        "reference": {
            "planned_frames": plan.planned_frames,
            "grid_length": plan.frame_count,
            "padded_to_minimum": plan.frame_count > h3_grid_length(plan.planned_frames),
            "expected_film_frames": plan.expected_film_frames,
            "film_frames_from_video": film_from_video,
            "film_padding_frames": padding,
            "clip_seconds_at_24_fps": round(plan.expected_film_frames / H3_FPS, 6),
            "sampling_fps": _exact(reference_rate),
            "first_reference_index": plan.reference_start,
            "end_reference_index": plan.reference_start + plan.planned_frames,
            "video_reference_frames_total": plan.planned_total,
        },
        "source": {
            "fps": _exact(timing.rate),
            "fps_decimal": float(timing.rate),
            "fps_source": timing.rate_source,
            "constant_frame_rate": timing.constant_rate,
            "duration_seconds": _exact(timing.duration),
            "duration_decimal": float(timing.duration),
            "frames": timing.frame_total,
            "base_timeline_frames": timing.base_total,
            "last_frame_duration_from": timing.last_frame_duration_source,
            "frame_index": timing.index_method,
            "segment_start_seconds": _exact(plan.logical_start),
            "segment_end_seconds": _exact(plan.logical_end),
            "segment_start_decimal": float(plan.logical_start),
            "segment_end_decimal": float(plan.logical_end),
            "window_first_frame": plan.source_first,
            "window_last_frame": plan.source_last,
            "window_frames": plan.source_count,
            "window_held_frames": held,
            "window_start_decimal": float(Fraction(plan.source_first) / timing.rate),
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
        },
        "boundaries": (
            "approximate: the window starts at the source frame at or before the segment start, "
            "so joins between chunks can repeat or skip one interpolated sample"
        ),
    }
    return display, info


def _send_text(node_id, text: str) -> None:
    if PromptServer is None or not node_id:
        return
    try:
        PromptServer.instance.send_progress_text(text, node_id)
    except Exception:
        logging.debug("%s could not update the node display", LOG_PREFIX, exc_info=True)


def _output_video_settings(video) -> tuple[int, str]:
    try:
        bit_depth = int(video.get_bit_depth())
    except Exception:
        bit_depth = 8
    try:
        color_space = video.get_color_space()
    except Exception:
        color_space = "sRGB"
    return bit_depth, color_space if color_space in COLOR_SPACES else "sRGB"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

HELP = (
    "Cuts a video into chunks for MiniMax H3 reference-to-video after FILM x2 interpolation, "
    "one chunk per run. Drive chunk_index with an Int node set to increment, starting at 0; "
    "queue total_runs runs. Past the last chunk the node stops with an error. "
    "Wire frames -> FILM VFI (x2) -> Create Video (24 fps) -> H3 reference video, and "
    "duration_seconds -> H3 duration. A full chunk is 173 source frames -> 345 FILM frames "
    "-> 14 s in H3. Chunk joins are approximate (one sample may repeat or be skipped). "
    "A final chunk under 2 s is padded with copies of the last frame. The index advances "
    "even when a later node fails, so set it back to redo a chunk."
)


class TrimVideoForMiniMaxH3InterpolatedReference(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id=NODE_ID,
            display_name=DISPLAY_NAME,
            category="video",
            search_aliases=["minimax h3", "h3 reference", "chunk video", "split video", "film interpolation"],
            description=HELP,
            inputs=[
                IO.Video.Input("video", tooltip="Video from Load Video, or any node that outputs VIDEO."),
                IO.Int.Input(
                    "chunk_index",
                    default=0,
                    min=0,
                    max=1_000_000,
                    step=1,
                    tooltip="Zero-based chunk to output. Connect an Int node set to increment and "
                    "start it at 0. Type a number to redo one chunk.",
                ),
            ],
            outputs=[
                IO.Video.Output(
                    "trimmed_video",
                    display_name="trimmed_video",
                    tooltip="The chunk's source frames as a silent video at the source frame rate.",
                ),
                IO.Image.Output(
                    "frames",
                    display_name="frames",
                    tooltip="The same source frames as an image batch. Connect to FILM VFI (x2).",
                ),
                IO.Int.Output(
                    "duration_seconds",
                    display_name="duration_seconds",
                    tooltip="Whole seconds for the MiniMax H3 duration input: 14 for a full chunk.",
                ),
                IO.Int.Output(
                    "total_runs",
                    display_name="total_runs",
                    tooltip="Runs needed for the whole video.",
                ),
                IO.Int.Output(
                    "current_index",
                    display_name="current_index",
                    tooltip="The chunk_index used for this run.",
                ),
                IO.String.Output(
                    "segment_info",
                    display_name="segment_info",
                    tooltip="JSON with exact rates, times, frame windows and padding for this chunk.",
                ),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, video, chunk_index: int) -> IO.NodeOutput:
        chunk_index = int(chunk_index)
        if chunk_index < 0:
            raise ValueError("chunk_index cannot be negative.")
        node_id = cls.hidden.unique_id
        _send_text(node_id, "Analyzing video…")

        if isinstance(video, InputImpl.VideoFromFile):
            timing, plan, frames = _prepare_file(video, chunk_index)
        else:
            timing, plan, frames = _prepare_memory(video, chunk_index)

        if frames.shape[0] != plan.source_count:
            raise RuntimeError(
                f"Prepared {frames.shape[0]} frames instead of {plan.source_count}; please report this."
            )

        bit_depth, color_space = _output_video_settings(video)
        trimmed_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=frames, frame_rate=timing.rate),
            bit_depth=bit_depth,
            color_space=color_space,
        )
        display, info = _details(timing, plan, frames)
        _send_text(node_id, display)
        return IO.NodeOutput(
            trimmed_video,
            frames,
            plan.duration_seconds,
            plan.total_runs,
            plan.index,
            json.dumps(info, indent=2, ensure_ascii=False),
        )


class TrimVideoMiniMaxH3Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [TrimVideoForMiniMaxH3InterpolatedReference]


async def comfy_entrypoint() -> TrimVideoMiniMaxH3Extension:
    return TrimVideoMiniMaxH3Extension()