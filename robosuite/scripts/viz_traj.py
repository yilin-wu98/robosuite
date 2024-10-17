import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np
import cv2
import argparse
import robosuite.utils.transform_utils as T
# import Pathlib
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
import os
import pickle
from matplotlib import cm
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def put_marker(env, obs, eef_trajectory, video_writer, camera_info, color_scheme):
    # Get the current EEF position
    # eef_pos = env.robots[0].controller.ee_pos
    eef_pos = env.sim.data.get_site_xpos(env.robots[0].gripper.important_sites['eef_site'])
    eef_pos_world = eef_pos.copy()
    eef_trajectory.append(eef_pos_world)
    camera_name = env.camera_names[0]
    # Capture the frame from the frontview camera
    frame = obs[f'{camera_name}_image']  # Get frontview camera image
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    # Project the 3D trajectory points into 2D camera spac
    frame = project_3d_to_2d(env, eef_trajectory, frame, camera_info, color_scheme)
    frame = cv2.flip(frame, 0)  # Flip the frame vertically
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    # Write the frame with the projected trajectory to the video
    video_writer.write(frame)
    return eef_trajectory

def project_3d_to_2d(env, points, frame, camera_info, color_scheme):
    width, height = frame.shape[1], frame.shape[0]
    camera_matrix  = get_camera_transform_matrix(env.sim, camera_info['name'],  height, width )
    coords= project_points_from_world_to_camera(np.array(points), camera_matrix, height, width)
    colormap = ListedColormap(sns.color_palette('viridis').as_hex())
    for i,(x, y) in enumerate(coords):
        # # Ensure the point is within the frame bounds
        # if i > 120:
            # color = colormap(i / 120)  # RGBA color from colormap
        # else: 
            # color = colormap2(i / 120)
        color = colormap(color_scheme[i])
        bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))  # Convert to BGR
        if 0 <= x < width and 0 <= y < height:
            # Draw a small circle on the frame for each trajectory point
            cv2.circle(frame, (y, height-x), radius=3, color=bgr_color, thickness=-1)  # Red points
    return frame

def load_env(args):
    controller_config = load_controller_config(custom_fpath="../controllers/config/osc_pose_abs.json")
    env = suite.make(
        env_name="Table",  # Use a task like "Lift" or "PickPlace"
        robots="Panda",   # Use Franka Panda robot
        # robots=[Panda(mount=robot_mount)],
        controller_configs=controller_config,
        has_renderer=True,  # Enable visualization
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=args.camera_name,  
        render_camera=args.camera_name,
        camera_widths= 1024,
        camera_heights= 1024,
        
    )
    env.robot_configs[0]['mount_type'] = None
    env.reset()
    return env



def transform_robot_pose_to_world(env, pos, quat):
    # # Transform the point from robot to world coordinates
    pose_in_base = [pos, quat]
    pose_in_base = T.pose2mat(pose_in_base)
    base_body_id = env.sim.model.body_name2id(env.robots[0].robot_model.root_body)
    robot_pos_in_world = env.sim.data.body_xpos[base_body_id]  # Base position in the world frame
    robot_orn_in_world = T.mat2quat(env.sim.data.body_xmat[base_body_id].reshape(3, 3))  # Base orientation in the world frame

    base_pose_in_world = T.pose2mat((robot_pos_in_world, robot_orn_in_world))

    pose_in_world = T.pose_in_A_to_pose_in_B(pose_A=pose_in_base, pose_A_in_B=base_pose_in_world)
    pose_in_world_quat = T.mat2pose(pose_in_world)
    pos_control, pos_quat = pose_in_world_quat[0], pose_in_world_quat[1]
    # pos_control, pos_quat = transform_world_to_controller_frame(pose_in_world_quat[0], pose_in_world_quat[1])
    pose_in_world_angle = np.concatenate([pos_control, T.quat2axisangle(pos_quat)])
    return pose_in_world_angle


def load_demo_traj(path):
    # Load the demo trajectory from a file
    with open(path, 'rb') as f:
        traj = pickle.load(f)
    return traj

def plot_in_robosuite(args):
    env = load_env(args)
    home_pos = [0.3743, -0.4294, -0.4103, -2.4497, -0.2268,  2.0513, -2.2046]
    # Apply the desired joint positions to the robot
    joint_velocities = env.sim.data.qvel[env.robots[0]._ref_joint_vel_indexes]
    env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes] = home_pos
    env.sim.data.qvel[env.robots[0]._ref_joint_vel_indexes] = np.zeros_like(joint_velocities)  # Set velocities to 0

    # Forward the simulation to apply the changes
    env.sim.forward()
    # camera information
    cam_id = env.sim.model.camera_name2id(args.camera_name)
    # Get the camera parameters (extrinsics)
    cam_pose = env.sim.data.cam_xpos[cam_id]  # Camera position
    cam_mat = env.sim.data.cam_xmat[cam_id].reshape(3, 3) 
    fovy = env.sim.model.cam_fovy[cam_id]
    camera_info = dict(name = args.camera_name, camera_pose=cam_pose, camera_matrix=cam_mat, camera_fovy = fovy)
    
    # Example real-world trajectory: list of (position, orientation)
    
    real_trajectory = load_demo_traj(args.demo)
    sim_trajs = []
    for j in range(len(real_trajectory)):
        sim_trajectory = [ ]
        for i in range(len(real_trajectory[j])):
            # real_world_trajectory.append(([0.3, 0.3, 1.2,0, 0, 0, 1]))
            # pos = transform_robot_to_world(env, [0.44677258, -0.04790852,  0.41858146])[:3]
            # pose = transform_robot_pose_to_world(env, [0.44677258, -0.04790852,  0.41858146], [0.48071266 , 0.87667538 ,-0.01194415,0.01459315])
            # pose = transform_robot_pose_to_world(env, [0.44677258, -0.04790852,  0.41858146], [0.48071266 , 0.87667538 ,-0.01194415,0.01459315])
            # print('real pose', real_trajectory[i])
            pose = transform_robot_pose_to_world(env, real_trajectory[j][i][:3], real_trajectory[j][i][3:])
            # print('world pose', pose)
            # mat = T.quat2mat([0.48071266 , 0.87667538 ,-0.01194415,0.01459315])
            # euler = T.mat2euler(mat)
            pos = pose[:3]
            for index in range(10):
                ## real gripper threshold is 0.03 so here we need to convert to -1, 1
                gripper_action = [-2 *(real_trajectory[j][i][7]>0.03) +1]
                sim_trajectory.append(np.concatenate([pos, [180,-180,0], gripper_action]))
                # sim_trajectory.append(np.concatenate([pose, [-2 *(real_trajectory[j][i][7]>0.03) +1]]))
        sim_trajs.append(sim_trajectory)
        
    # env
    video_writer = cv2.VideoWriter(
        args.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1024, 1024)
    )

    # Store the EEF trajectory in 3D space
    eef_trajectory = []
    color_scheme = []
    for j in range(len(sim_trajs)):
        sim_trajectory = sim_trajs[j]
        # color_scheme = 'rocket' #if j == 0 else 'viridis'
        for i in range(len(sim_trajectory)):
            
            action = sim_trajectory[i]
            # breakpoint()
            print('action', action)
            obs, reward, done, info= env.step(action)
            print('self.eepos', env.robots[0].controller.ee_pos)
            if i % 10 == 0 and i != 0 :
               color_scheme.append(j/2)
               eef_trajectory = put_marker(env, obs, eef_trajectory, video_writer, camera_info,color_scheme)
        # env.render()
   
    # Release the video writer
    video_writer.release()

    # Close the environment
    env.close()

def plot_in_matplotlib(args):
    colormap = ListedColormap(sns.color_palette('rocket').as_hex())
    # for traj_idx in range(10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    real_trajectory = load_demo_traj(args.demo)#args.demo.replace('actions_multi_modal.pkl', f'actions_multi_modal_{traj_idx}.pkl'))
    ## real trajectory is a list of (position, quaternion)
    ## plot them in 3d space with matplotlib
    for j in range(len(real_trajectory)):
        # real_world_trajectory.append(([0.3, 0.3, 1.2,0, 0, 0, 1]))
        traj = real_trajectory[j]
        n_points = len(traj) 
        color = colormap(j/2)  # RGBA color from colormap# Convert to BGR
        for i in range(n_points):
            pos = traj[i][:3]
            
            ax.scatter(pos[0], pos[1], pos[2], c=color, marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    plt.show()
    
def main(args):
    # traj = load_demo_traj(args)
    if args.plot_type == 'robosuite':
        plot_in_robosuite(args)
    else: 
        plot_in_matplotlib(args)
    




if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--env', type=str, default='Table')
    argparser.add_argument("--camera_name", type=str, default="birdview")
    argparser.add_argument("--demo", type=str, default="/home/yilinwu/Projects/preference-diffusion/manimo/manimo/scripts/actions_model/actions_2_contrast_samples5.pkl")
    argparser.add_argument("--plot_type", type=str, default="robosuite")
    argparser.add_argument("--video_path", type=str, default="output.mp4")
    main(argparser.parse_args()) 