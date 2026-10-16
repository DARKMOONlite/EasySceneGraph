# EasySceneGraph
A project for creating, fusing and manipulating spatial scene graphs,  


## Data sets
### Tested:
- [**NuScenes**](https://www.nuscenes.org/): `Tested`: A driving centric dataset containing multiple camera perspectives and highly annotated images.
- [**Visual Genome**](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html):`Tested` an ongoing effort to connect structured image concepts to language

## Installation

```bash

pip install .

```

## Usage

view the [**cli**] folder for additional scripts on how to use this package


### Fusion

```bash
python3 scene_graph_project/cli/fuse.py \
  {any number of folders} \
  -o {output folder}
```

### Stabalise Nuscenes 
> [!NOTE]
> this requires the nuscenes dataset loaded into a sqlite3 db. I've added the databasemanager class and some utility to make it easier to use.

```bash
python3 scene_graph_project/integrations/nuscenes/stabilise.py \
--save -o /mnt/sda1/Datasets/nuscenes/v1.0-mini/scene_graphs/stabalised_graphs_action2 \
 --samples-folder /mnt/sda1/Datasets/nuscenes/v1.0-mini/scene_graphs/samples/merged/ \
  --sweeps-folder /mnt/sda1/Datasets/nuscenes/v1.0-mini/scene_graphs/sweeps/merged/
```

### View Scene Graph

```bash
python3 scene_graph_project/cli/view.py {scene graph path}
```

## Package layout

- `scene_graph_project.fusion` contains the reusable models, format adapters,
  standardisation, and graph fusion API.
- `scene_graph_project.temporal` contains cross-frame tracking.
- `scene_graph_project.cli` contains command-line workflows.
- `scene_graph_project.integrations.nuscenes` contains NuScenes-specific
  utilities and orchestration.

The former `scene_graph_project.scene_graph_fusion` paths remain as
compatibility imports.

## Interesting Links
[**Scallop Interpreting over Graphs**](https://www.scallop-lang.org/log22/slides/log-2022-tutorial-scallop-part.pdf)


## TODO

#### - Conversion of Ego-centric, Allo-centric and Exo-centric scene graphs.
   - [VLM-Grounder](https://github.com/InternRobotics/VLM-Grounder)
   - [VIZOR](https://vivekmadhavaram.github.io/vizor/)
   - [Vil3DRel](https://arxiv.org/pdf/2211.09646)


