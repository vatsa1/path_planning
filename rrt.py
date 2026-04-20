import sim
from sim import PyBulletSim
import pybullet as p
import numpy as np
import time


#pyb = PyBulletSim()

MAX_ITERS = 10000
delta_q = .5

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    """
    Draw a line between two positions corresponding to the input configurations
    :param q_1: configuration 1
    :param q_2: configuration 2
    :param env: environment
    :param color: color of the line, please leave unchanged.
    """
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)

def construct_path(parents, q_init, q_goal):
    path = []
    path.append(q_goal)
    current = q_goal
    while not np.array_equal(current, q_init):
        parent = parents[tuple(current)]
        if parent is None:
            return None
        path.append(parent)
        current = parent
    path.reverse()
    return path

def semi_random_sample(q_goal, steer_goal_p, dof):
    """
    With steer_goal_p probability, sample q_goal.
    Otherwise sample uniformly from joint range [-pi, pi].
    """
    if np.random.rand() < steer_goal_p:
        return np.array(q_goal, dtype=float)
    return np.random.uniform(low=-np.pi, high=np.pi, size=dof)

def nearest(vertices, q_rand):
    """
    Returns nearest vertex to q_rand using L1 distance.
    """
    return min(vertices, key=lambda q: np.sum(np.abs(q - q_rand)))

def steer(q_near, q_rand, delta_q):
    direction = q_rand - q_near
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.array(q_near, dtype=float)
    q_new = q_near + delta_q * direction / norm
    return q_new

def obstacle_free(q_new, env):
    """
    For this assignment, it is sufficient to check q_new only.
    """
    return not env.check_collision(q_new)

def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env, visualize_edge_fn=None):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :param visualize_edge_fn: optional callback to visualize an edge (q_1, q_2, env, color)
    :returns path: list of configurations (joint angles) if found a path within MAX_ITERS, else None
    """
    q_init = np.array(q_init, dtype=float)
    q_goal = np.array(q_goal, dtype=float)
    dof = len(q_init)

    vertices = [q_init]
    edges = []
    parents = {tuple(q_init): None}

    for _ in range(MAX_ITERS):
        q_rand = semi_random_sample(q_goal, steer_goal_p, dof)
        q_nearest = nearest(vertices, q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)

        if obstacle_free(q_new, env):
            vertices.append(q_new)
            edges.append((q_nearest, q_new))
            parents[tuple(q_new)] = q_nearest

            if visualize_edge_fn is not None:
                visualize_edge_fn(q_nearest, q_new, env, color=[0, 1, 0])
            else:
                visualize_path(q_nearest, q_new, env, color=[0, 1, 0])

            if np.linalg.norm(q_new - q_goal) < delta_q:
                vertices.append(q_goal)
                edges.append((q_new, q_goal))
                parents[tuple(q_goal)] = q_new
                path = construct_path(parents, q_init, q_goal)
                return path

    return None

def execute_path(path_conf, env):
    # ========= TODO: Problem 3 ========
    # 1. Execute the path while visualizing the location of joint 5 
    #    (see Figure 2 in homework manual)
    #    You can get the position of joint 5 with:
    #         p.getLinkState(env.robot_body_id, 9)[0]
    #    To visualize the position, you should use sim.SphereMarker
    #    (Hint: declare a list to store the markers)
    # 2. Drop the object (Hint: open gripper, step the simulation, close gripper)
    # 3. Return the robot to original location by retracing the path 
    # env.open_gripper()
    # env.set_joint_positions(path_conf[0])
    # joint_5_pos = p.getLinkState(env.robot_body_id, 9)[0]
    # marker_1 = sim.SphereMarker(joint_5_pos, 0.02, [1, 0, 0])
    # markers.append(marker_1)
    if path_conf is None or len(path_conf) == 0:
        return None

    markers = []
    speed = 0.07

    # Execute forward path and visualize joint-5 positions with red spheres.
    for conf in path_conf:
        env.move_joints(conf, speed=speed)
        joint_5_pos = p.getLinkState(env.robot_body_id, 9)[0]
        markers.append(
            sim.SphereMarker(joint_5_pos, radius=0.02, rgba_color=[1, 0, 0, 0.8])
        )
        time.sleep(0.05)

    # Drop object at destination bin.
    env.open_gripper()
    env.step_simulation(120)
    env.close_gripper()
    env.step_simulation(120)

    # Retrace path back to the origin.
    for conf in reversed(path_conf):
        env.move_joints(conf, speed=speed)
        time.sleep(0.05)

    # Delete all joint-5 visualizations once robot returns.
    while markers:
        marker = markers.pop()
        del marker
    # ==================================
    return None