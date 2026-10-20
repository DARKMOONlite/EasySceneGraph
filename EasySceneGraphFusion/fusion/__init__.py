"""Public API for scene-graph models, standardisation, I/O, and fusion.

Modules:
    classes: Canonical graphs, objects, relationships, and bounding boxes.
    fusion: Object matching and multi-source scene-graph merging.
    io: JSON, Visual Genome, and COCO loading and JSON serialisation.
    standardiser: Label normalisation and graph element blacklisting.
    wordnet: WordNet lookup, hierarchy, and semantic similarity helpers.
"""

from .fusion import FusionConfig, ObjectMatch, SceneGraphFusion
from .io import (
    load_coco_detections,
    load_scene_graph_json,
    load_scene_graphs_json,
    load_visual_genome,
    save_scene_graph_json,
    scene_graph_to_dict,
)
from .classes import BoundingBox, Relationship, SceneGraph, SceneGraphJsonEncoder, SceneObject
from .standardiser import Standardiser

__all__ = [
    "BoundingBox",
    "FusionConfig",
    "ObjectMatch",
    "Relationship",
    "SceneGraph",
    "SceneGraphFusion",
    "SceneGraphJsonEncoder",
    "SceneObject",
    "Standardiser",
    "load_coco_detections",
    "load_scene_graph_json",
    "load_scene_graphs_json",
    "load_visual_genome",
    "save_scene_graph_json",
    "scene_graph_to_dict",
]
