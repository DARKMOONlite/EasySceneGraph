# Scene Graph Project Instructions

## Setup and commands

- This is a Python package targeting Python 3.8+. Install it from the repository root with:
  ```bash
  pip install .
  ```
- The fusion workflow requires spaCy's English model, which is not installed by the package declaration:
  ```bash
  python -m spacy download en_core_web_sm
  ```
- Filter a JSON directory tree with the installed console command:
  ```bash
  filter-scene-graphs INPUT_ROOT OUTPUT_ROOT
  ```
  The equivalent source invocation is `python scene_graph_project/cli/filter.py INPUT_ROOT OUTPUT_ROOT`.
- Fuse matching JSON files across two or more source trees:
  ```bash
  python scene_graph_project/cli/fuse.py SOURCE_A SOURCE_B -o OUTPUT_ROOT
  ```
- The repository has no tracked test modules and no configured test runner or linter. Do not invent project commands for them.

## Architecture

- `scene_graph_project.fusion` is the main reusable API. It exports the canonical `SceneGraph`, `SceneObject`, `Relationship`, and `BoundingBox` models; JSON/Visual Genome/COCO loaders; `Standardiser`; and `SceneGraphFusion`.
- Fusion is an ordered data flow: load source-specific data into the canonical models, call `Standardiser.standardise(graph)` in place, optionally call `Standardiser.blacklist(graph)`, then call `SceneGraphFusion.fuse(graphs)`. The fusion engine uses canonical object labels plus bounding-box IoU to form cross-source object groups, remaps relationship endpoints to the retained object UUIDs, and consolidates relationships according to `FusionConfig`.
- `cli/filter.py` applies the standardisation-and-blacklist stage to every JSON graph in a directory tree. `cli/fuse.py` finds the intersection of relative JSON paths across source trees and writes fused graphs under the same relative paths.
- `temporal/` adds cross-frame entity tracking. `integrations/nuscenes/stabilise.py` is a NuScenes-specific orchestration script and relies on local dataset paths, a SQLite-backed external `neurosymbolic_ILP` project, and image data; keep those environment dependencies separate from the generic fusion API.

## Project conventions

- Use the classes re-exported by `scene_graph_project.fusion` for shared fusion code. Source-format adapters belong in `fusion/io.py` and must convert to the canonical models rather than pass raw dictionaries downstream.
- Construct graph contents through `SceneGraph.add_object()` and `SceneGraph.add_relationship()` so each item receives the owning `scene_graph_id`. Relationships refer to objects by UUID, not position or serialized JSON IDs.
- `SceneObject` labels and `Relationship` predicates are normalized to stripped lowercase at construction. Preserve raw `label`/`predicate`; write normalization results to `canonical_label`/`canonical_predicate`. Call standardisation before any matching or blacklist operation.
- Generic scene-graph JSON serializes object UUIDs to sequential string IDs. When loading, relationship `subject`/`object` (or `subject_id`/`object_id`) must reference those input IDs; do not assume serialized IDs remain stable across a load/save cycle.
- Keep fusion deterministic: file collection is sorted, candidate ties use model fields, source names are deduplicated and sorted, and merged object attributes are sorted. Avoid unordered iteration when producing persisted results.
- The WordNet helper uses the repository author's mounted corpus path (`/mnt/sda1/Datasets/NLTK/wordnet-2022`) and `install_wordnet()` mutates that location. Make WordNet paths configurable before adapting this code for another environment.
