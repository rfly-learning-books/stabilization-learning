# Third-party provenance

## Stable Learning Control

The optional LAC integration uses [rickstaa/stable-learning-control](https://github.com/rickstaa/stable-learning-control), commit `27b6fdfed6407a46192d2304027b307750c18ba6`. Source and the original MIT license are retained under `third_party/stable-learning-control`. The toolbox's M-PPO and task implementations are separate additions.

## Fixed-wing dynamics

Fixed-wing reference sources derive from [mavsim_public](https://github.com/byu-magicc/mavsim_public), commit `10363e0eaeb16a61a7463505f8e122a9eb7e2b4e`. The corresponding GPL-3.0 text is retained in `migration/fixed_wing_preparation/20261023_v2/sources/LICENSE` and `migration/next_envs_v1/fixed_wing_longitudinal/LICENSE`. This component is not relabeled MIT.

## Indoor scenes

Static textures and meshes from Poly Haven are CC0. The source URLs, asset IDs and original hashes are recorded in `third_party/visual_original/assets/scene_dependencies/bright_natural_dense_servo_20261013/asset_provenance.json`. These are rendering resources, not training observations or checkpoints.

## FastDepth

The optional integration in `third_party/visual_original/runtime/dependencies/fastdepth_mobilenet_l1` originates from [idonahum/FastDepth](https://github.com/idonahum/FastDepth), commit `b1b2d469e45d4f4c25120e638fcab8699026e65b`. The retained workspace did not contain a separate license for these files; no new license is asserted here. Downloaded model weights are excluded.

## Research implementations

`third_party/push_original`, `third_party/so101_timeaware_v1`, `third_party/so101_timeaware_v2`, and the book RGB implementation preserve the project's research mechanisms. Their placement under `third_party` records implementation provenance rather than asserting authorship by Stable Learning Control. Calibration banks, historical learner states, replay data, learned depth weights and all experiment outputs remain external.
