from __future__ import annotations

from bitarray import bitarray
from EasySceneGraphFusion.integrations.nuscenes.database import DatabaseManager

from EasySceneGraphFusion.integrations.nuscenes.util.types import TimeSegment, Window, make_segment_key
import numpy as np

def remap_segments_to_scenes(
    segment_time_ranges: dict[str, TimeSegment],
) -> dict[str, list[TimeSegment]]:
    """Remap ``{segment_key: TimeSegment}`` to ``{scene_name: [TimeSegment]}``."""
    scene_segments: dict[str, list[TimeSegment]] = {}
    for segment_key, segment in segment_time_ranges.items():
        scene_name = segment_key.rsplit("+", 1)[0]
        scene_segments.setdefault(scene_name, []).append(segment)
    for segments in scene_segments.values():
        segments.sort()
    return scene_segments

def window_channel_folder_key(window: Window, channel_name: str) -> str:
    """Return ``scene__channel__start-end`` folder key for a window/channel."""
    start_ts = min(segment.start_ts for segment in window.segments)
    end_ts = max(segment.end_ts for segment in window.segments)
    return f"{window.scene_name}__{channel_name}__{start_ts}-{end_ts}"

def exact_segment_channel_folder_key(
    scene_name: str,
    section_id: int,
    channel_name: str,
    segment_time_ranges: dict[str, TimeSegment],
) -> str | None:
    """Return folder key for exactly one section_id segment in a scene."""
    segment_key = make_segment_key(scene_name, section_id)
    segment = segment_time_ranges.get(segment_key)
    if segment is None:
        return None
    return f"{scene_name}__{channel_name}__{segment.start_ts}-{segment.end_ts}"


def build_scene_windows_by_length(
    db: DatabaseManager,
    window_length: int,
    window_overlap: int,
) -> dict[str, list[Window]]:
    """Build fixed-length, optionally-overlapping windows per scene."""
    if window_length < 1:
        raise ValueError("--window-length must be >= 1")
    if window_overlap < 0:
        raise ValueError("--window-overlap must be >= 0")
    if window_overlap >= window_length:
        raise ValueError("--window-overlap must be smaller than --window-length")

    from EasySceneGraphFusion.integrations.nuscenes.util.types import get_scene_segment_time_ranges
    segment_time_ranges = get_scene_segment_time_ranges(db)
    scene_segments = remap_segments_to_scenes(segment_time_ranges)
    step = window_length - window_overlap
    scene_windows: dict[str, list[Window]] = {}

    for scene_name, segments in scene_segments.items():
        if not segments:
            continue
        if len(segments) <= window_length:
            scene_windows[scene_name] = [Window(scene_name=scene_name, segments=segments)]
            continue

        starts = list(range(0, len(segments) - window_length + 1, step))
        tail_start = len(segments) - window_length
        if starts[-1] != tail_start:
            starts.append(tail_start)

        windows = [
            Window(
                scene_name=scene_name,
                segments=segments[start_i : start_i + window_length],
            )
            for start_i in starts
        ]
        scene_windows[scene_name] = windows

    return scene_windows


def build_channel_folder_paths(
    scene_name: str,
    section_id: int | None,
    channels_seen: set[str],
    segment_time_ranges: dict[str, TimeSegment],
    scene_windows: dict[str, list[Window]] | None = None,
) -> list[str]:
    """Return output folder keys for one section/channel set."""
    if section_id is None:
        return []

    if scene_windows is None:
        folder_keys: list[str] = []
        for channel_name in channels_seen:
            key = exact_segment_channel_folder_key(
                scene_name=scene_name,
                section_id=section_id,
                channel_name=channel_name,
                segment_time_ranges=segment_time_ranges,
            )
            if key is None:
                return []
            folder_keys.append(key)
        return folder_keys

    matched_windows = [
        w
        for w in scene_windows.get(scene_name, [])
        if any(s.segment_key == section_id for s in w.segments)
    ]
    if not matched_windows:
        return []

    folder_key_set:set[str] = {
        window_channel_folder_key(window, channel_name)
        for window in matched_windows
        for channel_name in channels_seen
    }
    return sorted(folder_key_set)


def get_scene_segment_time_ranges(db: DatabaseManager) -> dict[str, TimeSegment]:
    """Return ``{segment_key: TimeSegment}`` for every segment in *action_results*.

    For each ``(scene_name, section_id)`` pair, spans
    ``min(start_aligned_timestamp)`` to ``max(end_aligned_timestamp)`` across
    all instance rows.
    """
    action_results = db.get_rows("action_results")
    ranges: dict[str, list[int]] = {}
    for row in action_results:
        start_ts = row.get("start_aligned_timestamp")
        end_ts = row.get("end_aligned_timestamp")
        if start_ts is None or end_ts is None:
            continue
        key = make_segment_key(row["scene_name"], row["section_id"])
        if key not in ranges:
            ranges[key] = [start_ts, end_ts, int(row["section_id"])]
        else:
            ranges[key][0] = min(ranges[key][0], start_ts)
            ranges[key][1] = max(ranges[key][1], end_ts)
    return {k: TimeSegment(v[0], v[1], v[2]) for k, v in ranges.items()}

def row_binary_comparison(rows: list[DBRow],columns:list[str], windows: list[TimeSegment]) -> np.ndarray:
    """Compute pairwise similarity between windows.

    Args:
        rows (list[DBRow]): rows to compare
        columns (list[str]): columns to include in the bitmask comparison, must be boolean or truthy/falsey values
        windows (list[TimeSegment]): list of time segments to compare

    Returns:
        np.ndarray: pairwise similarity matrix between windows
    """
    
    bitarray_per_window: dict[int, dict[str, bitarray]] = {} #? Build {obs_id: {instance_token: bitarray}} for every window.
    for window in windows:
        instance_bits: dict[str, bitarray] = {}
        for row in [r for r in rows if r.get("section_id") == window.segment_key]:
            ba = bitarray(len(columns))
            ba.setall(0)
            for j, column in enumerate(columns):
                if bool(row.get(column)):
                    ba[j] = 1
            instance_bits[row["instance_token"]] = ba
        bitarray_per_window[window.segment_key] = instance_bits

    similarity_between_windows: np.ndarray = np.zeros((len(windows), len(windows)))
    for i in range(len(windows)):
        for j in range(i + 1, len(windows)):
            bits_i = bitarray_per_window[windows[i].segment_key]
            bits_j = bitarray_per_window[windows[j].segment_key]
            union_keys = set(bits_i.keys()) | set(bits_j.keys())

            similarity_scores: list[float] = []
            for key in union_keys:
                ba_i = bits_i.get(key) or bitarray(len(columns))
                ba_j = bits_j.get(key) or bitarray(len(columns))
                num_differences = (ba_i ^ ba_j).count()
                num_union = len(ba_i | ba_j)
                similarity_scores.append(1 - num_differences / num_union if num_union > 0 else 1.0)

            similarity_between_windows[i, j] = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 1.0
    # similarity_between_windows += similarity_between_windows.T # make symmetric just in case
    return similarity_between_windows


def group_segments_to_windows(
    scene_name: str,
    segments: list[TimeSegment],
    similarity_between_windows: np.ndarray,
    threshold: float = 0.5,
    overlap: bool = False,
    min_timestamps: int = 1,
    max_timestamps: int = 100,
) -> list[Window]:
    """Group TimeSegments into Windows greedily by pairwise similarity.

    Starting from segment *i*, extends right (increasing *j*) as long as
    ``similarity_between_windows[i, j] >= threshold``.  On failure, closes
    the current Window and moves to the next starting index: *j* when
    ``overlap=False`` (no segment is reused), or *i+1* when ``overlap=True``.
    """

    windows: list[Window] = []
    i = 0
    while i < len(segments):
        current: list[TimeSegment] = [segments[i]]
        j = i + 1
        for j in range(i + 1, len(segments)):
            if similarity_between_windows[i, j] >= threshold:
                current.append(segments[j])
                if len(current) >= max_timestamps: 
                    break # hard cut if we exceed max timestamps, even if similarity is high
            else:
                break  # threshold not met — move to next row
        windows.append(Window(scene_name, segments=current))
        
        i = j if not overlap else i + 1
    windows = [w for w in windows if min_timestamps <= w.length]
    return windows
