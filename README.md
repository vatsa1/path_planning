# UR5 Path Planning in PyBullet (RRT + Grasping)

A simulation project for moving a UR5 robot arm between bins while avoiding collisions, grasping objects with a Robotiq 2F-85 gripper, and executing collision-free trajectories planned with Rapidly-exploring Random Trees (RRT).

![Project demo](assets/visualisation.gif)

## Highlights

- UR5 motion control via PyBullet inverse kinematics.
- Top-down grasping with Robotiq 2F-85 (yaw-based grasp angle).
- RRT planner in joint space with collision checking against scene obstacles.
- Path execution pipeline:
  - execute planned path,
  - visualize joint-5 trajectory with red sphere markers,
  - drop object in target bin,
  - retrace back to source side.

## Repository Structure

- `main.py` - entrypoint for part-wise testing.
- `sim.py` - PyBullet environment, robot/gripper control, grasp helpers.
- `rrt.py` - RRT planner and path execution logic.
- `clear_bin.py` - integrated bin-clearing pipeline.
- `gen_seg_data.py`, `train_seg_model.py` - segmentation data/training utilities.
- `icp.py`, `transforms.py`, `camera.py`, `image.py` - geometry/vision helpers.
- `assets/` - URDFs, meshes, obstacles, and demo GIF.
- `environment.yaml` - conda environment specification.

## Setup

Use Conda to create the environment:

```bash
conda env create -f environment.yaml
conda activate comsw4733_hw3
```

If your local toolchain uses newer package versions, you can create a custom env and install equivalents from `environment.yaml`.

## Run

Run from the repository root:

```bash
python main.py -part 1 -n 3 -disp
python main.py -part 2 -n 3 -disp
python main.py -part 3 -n 3 -disp
```

Where:

- `-part 1` tests IK-based end-effector movement.
- `-part 2` tests grasping.
- `-part 3` tests grasp + RRT path planning + transfer.
- `-disp` enables PyBullet GUI visualization.

## RRT Showcase Flow (Part 3)

1. Grasp object from the source bin.
2. Build an RRT in joint space from home to goal configuration.
3. Execute the planned path while visualizing joint-5 waypoints.
4. Open gripper above destination bin to place object.
5. Close gripper and retrace the same path back.
6. Remove temporary visual markers.

## Notes and Limitations

- Collision checking is done at sampled configurations (`q_new`) during RRT expansion.
- Planning performance depends on `delta_q` and goal-bias probability.
- The project is simulation-focused (no real robot hardware integration in this repo).

