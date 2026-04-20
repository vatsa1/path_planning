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
        #print(parent)
        path.append(parent)
        #print(path)
        current = parent
    path.reverse()
    return path

def steer(q_near, q_rand, delta_q):
    q_new = q_near + delta_q * (q_rand - q_near) / np.linalg.norm(q_rand - q_near)
    return q_new

def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: list of configurations (joint angles) if found a path within MAX_ITERS, else None
    """
    # ========= TODO: Problem 3 ========
    # Implement RRT code here. This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init
    # Use visualize_path() to visualize the edges in the exploration tree for part (b)
    #env_sim = PyBulletSim()
    v = [q_init]
    e= {}
    parents = {tuple(q_init): None}
    for i in range(MAX_ITERS):
        if np.random.rand() < steer_goal_p:
            q_rand = q_goal
        else:
            q_rand= np.random.uniform(low=-np.pi, high=np.pi, size=6)

        q_nearest = min(v, key=lambda q: np.linalg.norm(q - q_rand)) #compute nearest node to q_rand
        q_new = steer(q_nearest, q_rand, delta_q) #steer towards q_rand
        if PyBulletSim.check_collision(env,q_new) == False:
            v.append(q_new)
            #e[q_nearest, q_new] = 1
            parents[tuple(q_new)] = q_nearest
            
            if np.linalg.norm(q_new - q_goal) < delta_q:
                v.append(q_goal)
                #e[q_new, q_goal] = 1
                parents[tuple(q_goal)] = q_new
                #print(parents)
                path= construct_path(parents,q_init, q_goal)
                # print(path)
                return path
            
            visualize_path(q_nearest, q_new, env)
    # ==================================
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
    markers=[]
    speed=.07
    
    for i in range(len(path_conf)):
        #print(path_conf)
        #env.set_joint_positions(conf)
        joint_5_pos = p.getLinkState(env.robot_body_id, 9)[0]
        #print(joint_5_pos)
        marker = sim.SphereMarker(joint_5_pos, 0.02, [1, 0, 0, 0.8])
        markers.append(marker)
        env.move_joints(path_conf[i],speed)
        time.sleep(0.50)


    env.open_gripper()
    time.sleep(0.50)
    env.close_gripper()
    time.sleep(0.50)

    for conf in reversed(path_conf):
        env.set_joint_positions(conf)

    for marker in markers:
        marker.__del__()
    # ==================================
    return None