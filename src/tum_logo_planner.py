import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import argparse

import rdplib.ik as ik
import rdplib.plot_helpers as ph
from rdplib.planning.PTP import PTP_quintic
import rdplib.robot as rb

# --- Constants & Configuration ---
DT = 0.004  # Time step [s]
STAGE_OUTPUT_DIR = 'stage_outputs'
WAYPOINTS_FILE = 'tumwaypoints_r_TCP.csv'

# Ensure the directory for intermediate stage files exists
os.makedirs(os.path.join(os.path.dirname(__file__), STAGE_OUTPUT_DIR), exist_ok=True)

def get_stage_path(stage_num):
    """Gets the file path for a given stage's output."""
    script_dir = os.path.dirname(__file__)
    return os.path.join(script_dir, STAGE_OUTPUT_DIR, f'stage_{stage_num}_output.npz')

def save_viewer_trajectory(filename, T_segment, q_segment, robot):
    """Generates and saves a trajectory file in the detailed viewer format."""
    print(f"Saving viewer data to {filename}...")
    total_steps = len(T_segment)
    if total_steps == 0:
        print(f"No viewer data to save for {filename}.")
        return
    viewer_data = np.zeros((total_steps, 12 * robot.size + 1))
    for i in range(total_steps):
        robot.calculate_positions(q_segment[:, i])
        viewer_data[i, 0] = T_segment[i]
        for l in range(robot.size):
            viewer_data[i, l*12+1:l*12+4] = robot.links[l].r_i__0.T
            viewer_data[i, l*12+4:l*12+13] = robot.links[l].A_i0.T.flatten()
    np.savetxt(filename, viewer_data, delimiter="\t", fmt='%.4f')
    print(f"Successfully saved {filename}")

def save_joint_trajectory(filename, T_segment, q_segment):
    """Generates and saves a trajectory file with time and joint angles (N x 7)."""
    print(f"Saving joint data to {filename}...")
    total_steps = len(T_segment)
    if total_steps == 0:
        print(f"No joint data to save for {filename}.")
        return
    T_reshaped = T_segment.reshape(-1, 1)
    q_transposed = q_segment.T
    output_data = np.hstack((T_reshaped, q_transposed))
    np.savetxt(filename, output_data, delimiter=",", fmt='%.6f')
    print(f"Successfully saved {filename}")

def show_all_plots(title, T, q, dot_q, r, v, waypoints):
    """Displays a comprehensive set of plots for a given trajectory."""
    print("Generating plots...")
    ph.plot_cartesian_trajectory_with_points(r, waypoints)
    plt.title(title)
    
    ph.plot_angles(q, T, 'q', 'Joint Angles', 'rad')
    ph.plot_angles(dot_q, T, 'q_dot', 'Joint Velocity', 'rad/s')
    ph.plot_cartesian_trajectory(r, T, 'r_TCP', 'TCP Position', 'm')
    ph.plot_cartesian_trajectory(v, T, 'v_TCP', 'TCP Velocity', 'm/s')
    
    plt.show()

def get_cumulative_data(current_stage):
    """Loads and concatenates all data up to the current stage."""
    all_T, all_q, all_dot_q, all_r, all_v, all_waypoints = [], [], [], [], [], []
    time_offset = 0.0
    for i in range(1, current_stage + 1):
        stage_file = get_stage_path(i)
        if not os.path.exists(stage_file):
            print(f"Error: Stage {i} output file not found. Cannot get cumulative data.")
            return None, None, None, None, None, None
        data = np.load(stage_file)
        if data['T'].size > 0:
            current_T_segment = data['T'] + time_offset
            all_T.append(current_T_segment)
            all_q.append(data['q'])
            all_dot_q.append(data['dot_q'])
            all_r.append(data['r'])
            all_v.append(data['v'])
            if 'waypoints' in data and data['waypoints'].size > 0:
                all_waypoints.append(data['waypoints'])
            time_offset = current_T_segment[-1] + DT

    if not all_T:
        return None, None, None, None, None, None

    T_cum = np.concatenate(all_T)
    q_cum = np.concatenate(all_q, axis=1)
    dot_q_cum = np.concatenate(all_dot_q, axis=1)
    r_cum = np.concatenate(all_r, axis=1)
    v_cum = np.concatenate(all_v, axis=1)
    waypoints_cum = np.vstack(all_waypoints) if all_waypoints else None
    return T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum

def plan_and_calculate_kinematics(robot, waypoints, T_part, is_joint_space, ik_solver=None, q_start=None, A_N0=None):
    """Plans a trajectory segment by segment to avoid time overlap issues from the planner."""
    if waypoints is None or len(waypoints) < 2:
        return np.array([]), np.array([[] for _ in range(robot.size)]), np.array([[] for _ in range(robot.size)]), np.array([[] for _ in range(robot.size)]), np.array([[],[],[]]), np.array([[],[],[]])

    all_T, all_q, all_dot_q, all_ddot_q, all_r, all_v = [], [], [], [], [], []
    time_offset = 0.0
    current_q_start = q_start

    for i in range(len(waypoints) - 1):
        segment_waypoints = np.array([waypoints[i], waypoints[i+1]])
        segment_duration = np.array([T_part[i]])
        if is_joint_space: current_q_start = segment_waypoints[0]

        if is_joint_space:
            T_seg, q_seg, dot_q_seg, ddot_q_seg = PTP_quintic(segment_waypoints, segment_duration, DT)
            r_seg, v_seg = np.zeros((3, len(T_seg))), np.zeros((3, len(T_seg)))
            for j in range(len(T_seg)):
                robot.calculate_positions(q_seg[:, j]); r_seg[:, j] = robot.r_TCP__0
                robot.calculate_velocities(dot_q_seg[:, j]); v_seg[:, j] = robot.v_TCP__0
        else: # Workspace
            T_seg, r_seg, v_seg, ddot_q_seg = PTP_quintic(segment_waypoints, segment_duration, DT) # ddot_q_seg is dummy here
            q_seg, dot_q_seg = np.zeros((robot.size, len(T_seg))), np.zeros((robot.size, len(T_seg)))
            robot.calculate_positions(current_q_start); robot.calculate_jacobis()
            for j in range(len(T_seg)):
                q_ik, dot_q_ik, _ = ik_solver.calculate_ik_rmc(robot, v_seg[:, j], r_seg[:, j], np.zeros(3), A_N0)
                q_seg[:, j], dot_q_seg[:, j] = q_ik, dot_q_ik
                robot.calculate_positions(q_ik); robot.calculate_velocities(dot_q_ik); robot.calculate_jacobis()
        
        if not is_joint_space and q_seg.size > 0: current_q_start = q_seg[:, -1]

        if T_seg.size > 0:
            all_T.append(T_seg + time_offset); all_q.append(q_seg); all_dot_q.append(dot_q_seg); all_ddot_q.append(ddot_q_seg); all_r.append(r_seg); all_v.append(v_seg)
            time_offset = all_T[-1][-1] + DT

    T_final = np.concatenate(all_T) if all_T else np.array([])
    q_final = np.concatenate(all_q, axis=1) if all_q else np.array([[] for _ in range(robot.size)])
    dot_q_final = np.concatenate(all_dot_q, axis=1) if all_dot_q else np.array([[] for _ in range(robot.size)])
    ddot_q_final = np.concatenate(all_ddot_q, axis=1) if all_ddot_q else np.array([[] for _ in range(robot.size)])
    r_final = np.concatenate(all_r, axis=1) if all_r else np.array([[],[],[]])
    v_final = np.concatenate(all_v, axis=1) if all_v else np.array([[],[],[]])
    return T_final, q_final, dot_q_final, ddot_q_final, r_final, v_final

def run_stage_1(robot, show_plots=True):
    print("--- Running Stage 1: Home -> Ready (via Cartesian midpoint) ---")
    r_ready = np.array([0.44141, -0.11596, -0.210]); A_N0_target = np.array([[1,0,0],[0,-1,0],[0,0,-1]]); q_home = robot.q_home
    ik_solver_pos = ik.IKSolverPositionDLS(); robot.calculate_positions(q_home); r_home = robot.r_TCP__0
    r_mid = (r_home + r_ready) / 2.0
    q_mid = ik_solver_pos.calculate_joint_angles(robot, r_mid, A_N0_target, q_start=q_home)
    q_ready = ik_solver_pos.calculate_joint_angles(robot, r_ready, A_N0_target, q_start=q_mid)
    waypoints_q = np.array([q_home, q_mid, q_ready])
    T, q, dot_q, ddot_q, r, v = plan_and_calculate_kinematics(robot, waypoints_q, np.array([4.0, 4.0]), True)
    plot_waypoints = np.array([r_home, r_mid, r_ready])
    np.savez(get_stage_path(1), T=T, q=q, dot_q=dot_q, r=r, v=v, q_end=q[:, -1], r_end=r[:,-1], A_N0_target=A_N0_target, waypoints=plot_waypoints)
    print("Stage 1 complete.")
    save_viewer_trajectory('trajectory_stage_1_viewer.csv', T, q, robot)
    save_joint_trajectory('trajectory_stage_1_joints.csv', T, q)
    if show_plots:
        T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum = get_cumulative_data(1)
        show_all_plots("Trajectory after Stage 1", T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum)

def run_stage_2(robot, show_plots=True):
    print("--- Running Stage 2: Ready -> Pre-load ---")
    data_in = np.load(get_stage_path(1)); q_start, r_start, A_N0_target = data_in['q_end'], data_in['r_end'], data_in['A_N0_target']
    drawing_waypoints = np.genfromtxt(os.path.join('planning_input', 'ABB_GoFa', WAYPOINTS_FILE), delimiter=','); r_preload = drawing_waypoints[0,:]
    waypoints = np.array([r_start, r_preload])
    ik_solver_vel = ik.IKSolverVelocity(np.diag([10]*4+[1]*2), k_position=10.0, k_orientation=10.0, dt=DT)
    T, q, dot_q, ddot_q, r, v = plan_and_calculate_kinematics(robot, waypoints, np.array([2.0]), False, ik_solver_vel, q_start, A_N0_target)
    np.savez(get_stage_path(2), T=T, q=q, dot_q=dot_q, r=r, v=v, q_end=q[:,-1], r_end=r[:,-1], A_N0_target=A_N0_target, waypoints=waypoints)
    print("Stage 2 complete.")
    if show_plots:
        T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum = get_cumulative_data(2)
        save_viewer_trajectory('trajectory_stage_2_viewer.csv', T_cum, q_cum, robot)
        save_joint_trajectory('trajectory_stage_2_joints.csv', T_cum, q_cum)
        show_all_plots("Trajectory after Stage 2", T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum)

def run_stage_3(robot, show_plots=True):
    print("--- Running Stage 3: Drawing ---")
    data_in = np.load(get_stage_path(2)); q_start, A_N0_target = data_in['q_end'], data_in['A_N0_target']
    waypoints = np.genfromtxt(os.path.join('planning_input', 'ABB_GoFa', WAYPOINTS_FILE), delimiter=',')
    ik_solver_vel = ik.IKSolverVelocity(np.diag([10]*4+[1]*2), k_position=10.0, k_orientation=10.0, dt=DT)
    T, q, dot_q, ddot_q, r, v = plan_and_calculate_kinematics(robot, waypoints, np.ones(len(waypoints)-1) * 2, False, ik_solver_vel, q_start, A_N0_target)
    np.savez(get_stage_path(3), T=T, q=q, dot_q=dot_q, r=r, v=v, q_end=q[:,-1], r_end=r[:,-1], A_N0_target=A_N0_target, waypoints=waypoints)
    print("Stage 3 complete.")
    if show_plots:
        T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum = get_cumulative_data(3)
        save_viewer_trajectory('trajectory_stage_3_viewer.csv', T_cum, q_cum, robot)
        save_joint_trajectory('trajectory_stage_3_joints.csv', T_cum, q_cum)
        show_all_plots("Trajectory after Stage 3", T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum)

def run_stage_4(robot, show_plots=True):
    print("--- Running Stage 4: Retract ---")
    data_in = np.load(get_stage_path(3)); q_start, r_start, A_N0_target = data_in['q_end'], data_in['r_end'], data_in['A_N0_target']
    r_retract = r_start.copy(); r_retract[2] = -0.210
    waypoints = np.array([r_start, r_retract])
    ik_solver_vel = ik.IKSolverVelocity(np.diag([10]*4+[1]*2), k_position=10.0, k_orientation=10.0, dt=DT)
    T, q, dot_q, ddot_q, r, v = plan_and_calculate_kinematics(robot, waypoints, np.array([2.0]), False, ik_solver_vel, q_start, A_N0_target)
    np.savez(get_stage_path(4), T=T, q=q, dot_q=dot_q, r=r, v=v, q_end=q[:,-1], r_end=r[:,-1], A_N0_target=A_N0_target, waypoints=waypoints)
    print("Stage 4 complete.")
    if show_plots:
        T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum = get_cumulative_data(4)
        save_viewer_trajectory('trajectory_stage_4_viewer.csv', T_cum, q_cum, robot)
        save_joint_trajectory('trajectory_stage_4_joints.csv', T_cum, q_cum)
        show_all_plots("Trajectory after Stage 4", T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum)

def run_stage_5(robot, show_plots=True):
    print("--- Running Stage 5: Return Home ---")
    data_in = np.load(get_stage_path(4)); q_start, q_home = data_in['q_end'], robot.q_home
    waypoints = np.array([q_start, q_home])
    T, q, dot_q, ddot_q, r, v = plan_and_calculate_kinematics(robot, waypoints, np.array([4.0]), True)
    np.savez(get_stage_path(5), T=T, q=q, dot_q=dot_q, r=r, v=v, q_end=q[:, -1], r_end=r[:,-1], waypoints=np.array([r[...,0], r[...,-1]]))
    print("Stage 5 complete.")
    if show_plots:
        T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum = get_cumulative_data(5)
        save_viewer_trajectory('trajectory_stage_5_viewer.csv', T_cum, q_cum, robot)
        save_joint_trajectory('trajectory_stage_5_joints.csv', T_cum, q_cum)
        show_all_plots("Trajectory after Stage 5", T_cum, q_cum, dot_q_cum, r_cum, v_cum, waypoints_cum)

def run_combine(robot, show_plots=True):
    print("--- Combining all stages for final output ---")
    T_final, q_final, dot_q_final, r_final, v_final, waypoints_final = get_cumulative_data(5)
    if T_final is not None:
        save_viewer_trajectory('trajectory_final_combined_viewer.csv', T_final, q_final, robot)
        save_joint_trajectory('trajectory_final_combined_joints.csv', T_final, q_final)
        print("Final combined files generated.")
        if show_plots:
            show_all_plots("Final Combined Trajectory", T_final, q_final, dot_q_final, r_final, v_final, waypoints_final)
    print("Combine step finished.")

def main():
    parser = argparse.ArgumentParser(description='Run trajectory planning in stages.')
    parser.add_argument('--stage', required=True, choices=['1', '2', '3', '4', '5', 'combine', 'all'], help='The stage to execute.')
    args = parser.parse_args()
    robot = rb.robot_factory(os.path.join('robot_config_files', 'ABB_GoFa.yaml'))
    
    if args.stage == 'all':
        run_stage_1(robot, show_plots=False)
        run_stage_2(robot, show_plots=False)
        run_stage_3(robot, show_plots=False)
        run_stage_4(robot, show_plots=False)
        run_stage_5(robot, show_plots=False)
        run_combine(robot, show_plots=True)
    elif args.stage == '1': run_stage_1(robot)
    elif args.stage == '2': run_stage_2(robot)
    elif args.stage == '3': run_stage_3(robot)
    elif args.stage == '4': run_stage_4(robot)
    elif args.stage == '5': run_stage_5(robot)
    elif args.stage == 'combine': run_combine(robot)

if __name__ == '__main__':
    main()