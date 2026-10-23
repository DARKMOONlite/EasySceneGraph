from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


def make_segment_key(scene_name: str, observation_id: int | str) -> str:
    """Return the canonical segment directory name used for output folders."""
    return f"{scene_name}+{observation_id}"


@dataclass
class Frame:
    """One timestep in a vehicle instance's annotation track."""

    sample_token: str
    annotation_token: str  # NuScenes annotation for the target instance at this frame.


@dataclass
class ObjectFrameSequence:
    """A single tracked object with its ordered sequence of frames."""

    instance_token: str
    
    category_name: str
    scene_name: str
    track: list[Frame]  # Ordered chronologically by scene sample order.
    section_id: int | None = None #? The position of this instance within the scene's segmentation, used for output folder naming.  Optional because some instances may not be assigned to a specific segment (e.g. if they only appear in one segment or if segment boundaries are based on other criteria).
    num_sections: int | None = None #? Total number of segments in the scene, used for display purposes.  Optional because it may not be known at the time of ObjectFrameSequence creation (e.g. if segmentation is done later based on other criteria).
    confidence: float | None = None #? Confidence score for the instance's action classification, used for sorting.  Optional because it may not be available for all instances.

@dataclass(order=True)
class TimeSegment:
    """A time-bounded segment for one section of a scene."""

    start_ts: int 
    end_ts: int
    segment_key: int  # the position of this timesegment in the whole scene, used for output folder naming

    def contains(self, timestamp: int) -> bool:
        """Return True if *timestamp* falls within this segment (inclusive)."""
        return self.start_ts <= timestamp <= self.end_ts

    def __hash__(self):
        return hash((self.start_ts, self.end_ts, self.segment_key))

@dataclass
class Window:
    """A (possibly merged) group of adjacent Time Segments within a single scene."""

    scene_name: str
    segments: list[TimeSegment]

    @property
    def segment_key(self) -> str:
        """Return an output folder key spanning all segments in this window."""
        keys = [s.segment_key for s in self.segments]
        if len(keys) == 1:
            return make_segment_key(self.scene_name, keys[0])
        return f"{self.scene_name}+{min(keys)}-{max(keys)}"

    @property
    def length(self) -> int:
        """Return the total number of segments contained in this window."""
        return len(self.segments)

    @classmethod
    def from_folder_name(
        cls,
        folder_path: Path,
        segment_time_ranges: dict[str, TimeSegment] | None = None,
    ) -> Window | None:
        """Parse a window folder name into a :class:`Window`.

        Accepts names in the form ``scene-0061+0`` (single segment) or
        ``scene-0061+0-2`` (merged range).  When *segment_time_ranges* (from
        :func:`~multiframe_behaviours.nuscenes.util.windowing.get_scene_segment_time_ranges`) is
        provided, real ``start_ts``/``end_ts`` values are populated for each
        segment; otherwise both timestamps default to zero.

        Args:
            folder_path: Path whose ``name`` matches the expected pattern.
            segment_time_ranges: Optional pre-built ``{segment_key: TimeSegment}``
                mapping used to look up real timestamps.  Pass ``None`` when
                only segment IDs are needed (e.g. pure folder routing).

        Returns:
            A :class:`Window` with the parsed segments, or ``None`` when the
            folder name does not match the ``<scene>+<N>`` / ``<scene>+<N>-<M>``
            pattern.
        """
        name = folder_path.name
        if "+" not in name:
            return None
        scene_name, segment_part = name.rsplit("+", 1)
        parts = segment_part.split("-")
        try:
            obs_ids = (
                [int(parts[0])]
                if len(parts) == 1
                else list(range(int(parts[0]), int(parts[-1]) + 1))
            )
        except ValueError:
            return None

        def _resolve(oid: int) -> TimeSegment:
            if segment_time_ranges is not None:
                key = make_segment_key(scene_name, oid)
                if key in segment_time_ranges:
                    return segment_time_ranges[key]
            return TimeSegment(start_ts=0, end_ts=0, segment_key=oid)

        return cls(
            scene_name=scene_name,
            segments=[_resolve(oid) for oid in obs_ids],
        )
