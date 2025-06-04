import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import sys

import rdplib.robot as rb
import rdplib.plot_helpers as ph
import rdplib.ik as ik

# Python 3.10 is the minimum version required for this project
assert sys.version_info >= (3, 10)

# choose robot and ik method
robot_name = 'FrankaEmika' # 'ABB_GoFa' or 'FrankaEmika'
# robot_name = 'ABB_GoFa'

# ik_method = 'rmc' # 'rmc' or 'asc'
ik_method = 'asc'

# ik_nullspace = ik.IKOptions.LIMIT # COMFORT LIMIT BOTH only matters for asc
ik_nullspace = ik.IKOptions.COMFORT

# create robot
robot = rb.robot_factory(os.path.join('robot_config_files',robot_name + '.yaml'))

# import trajectory data
files_path = 'inverse_kinematics_trajectories'

# time array
traj_t = np.genfromtxt(os.path.join(files_path, robot_name,'t.csv'), delimiter=',') 
# time step
dt = np.genfromtxt(os.path.join(files_path, robot_name,'dt.csv'), delimiter=',')
traj_r_TCP_d = np.genfromtxt(os.path.join(files_path, robot_name,'r_TCP_d.csv'), delimiter=',')
traj_v_TCP_d = np.genfromtxt(os.path.join(files_path, robot_name,'v_TCP_d.csv'), delimiter=',')
traj_A_N0_d = np.genfromtxt(os.path.join(files_path, robot_name,'A_N0_d.csv'), delimiter=',')
traj_omega_TCP_d = np.genfromtxt(os.path.join(files_path, robot_name,'omega_TCP_d.csv'), delimiter=',')
# joint configuration for start position
q0 = np.genfromtxt(os.path.join(files_path, robot_name,'q0.csv'), delimiter=',')

# number of timesteps
numsteps = len(traj_t)

# parameters suitable for the robot
if robot_name == 'ABB_GoFa' :
    W = np.diag([10.0, 10.0, 10.0, 10.0, 1.0, 1.0])
elif robot_name == 'FrankaEmika' :
    W = np.diag([10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
else :
    print('No W defined for the robot. Using scalar 1.0 instead.')
    W = 1.0

# create parameters
solver = ik.IKSolverVelocity(W,
                   k_position=10.0,
                   k_orientation=10.0,
                   dt=dt,
                   alpha = 10,
                   opt=ik_nullspace)

# initialize matrizes for the results
traj_q = np.zeros((robot.size, numsteps))
traj_dot_q = np.zeros((robot.size, numsteps))
traj_r_TCP = np.zeros((3, numsteps))
traj_v_TCP = np.zeros((3, numsteps))
traj_rotvec_N0 = np.zeros((3, numsteps))
traj_rotvec_N0_d = np.zeros((3, numsteps))
traj_omega_T = np.zeros((3, numsteps))
data = np.zeros((numsteps,12*robot.size+1))
# data for the robot controller of GoFa
data_egm = np.zeros((numsteps,8))

# initial DK
robot.calculate_positions(q0)
robot.calculate_jacobis()
solH = 0


## Inverse Kinematics
for i in range(numsteps):
    if ik_method == 'asc' :
        # Achtung: H wird wieder nicht mehr übergeben, da die Planning diese Funktion wieder so braucht, wie die Studierenden sie bekommen haben
        [q, dot_q, H] = solver.calculate_ik_asc(robot, traj_v_TCP_d[:,i], traj_r_TCP_d[:,i], traj_omega_TCP_d[:,i], np.reshape(traj_A_N0_d[:,i],(3,3)))
        solH = solH + H
    else :
        [q, dot_q, H] = solver.calculate_ik_rmc(robot, traj_v_TCP_d[:,i], traj_r_TCP_d[:,i], traj_omega_TCP_d[:,i], np.reshape(traj_A_N0_d[:,i],(3,3)))
        solH = solH + H

    robot.calculate_positions(q)
    robot.calculate_velocities(dot_q)
    robot.calculate_jacobis()

    # add data to result variables
    traj_q[:,i] = q
    traj_dot_q[:,i] = dot_q
    traj_r_TCP[:,i] = robot.r_TCP__0
    traj_v_TCP[:,i] = robot.v_TCP__0
    traj_rotvec_N0[:,i] = R.from_matrix(robot.links[-1].A_i0).as_rotvec()
    traj_rotvec_N0_d[:,i] = R.from_matrix(np.reshape(traj_A_N0_d[:,i],(3,3))).as_rotvec()
    traj_omega_T[:,i] = robot.links[-1].A_i0.T @ robot.links[-1].omega_i__i

    # build array for viewer
    for l in range(robot.size):
        data[i,0] = traj_t[i]
        data[i,l*12+1:l*12+4] = np.transpose(robot.links[l].r_i__0)
        data[i,l*12+4:l*12+13] = np.reshape(np.transpose(robot.links[l].A_i0),(1,9))

    # build array for the ABB controller
    for l in range(robot.size):
        quaternion = R.from_matrix(robot.links[-1].A_i0).as_quat()
        data_egm[i,0] = traj_t[i]
        data_egm[i,1:4] = 1000 * robot.r_TCP__0
        data_egm[i,4] = quaternion[3]
        data_egm[i,5:8] = quaternion[0:3]
#print (solH)
# save data for viewer and the ABB controller
np.savetxt('trajectory.csv', data, delimiter="\t", fmt='%.4f')
np.savetxt('egm_trajectory.csv', data_egm, delimiter=",", fmt='%.4f')

# visualization with matplotlib
ph.plot_path_3d([traj_r_TCP, traj_r_TCP_d], 'Desired and Actual Trajectory in Cartesian Space')
ph.plot_cartesian_trajectory([traj_r_TCP, traj_r_TCP_d], traj_t, ['r_TCP', 'r_TCP_d'], 'Position of TCP', 'm')
ph.plot_cartesian_trajectory([traj_v_TCP, traj_v_TCP_d], traj_t, ['v_TCP', 'v_TCP_d'], 'Velocity of TCP', 'm/s')
ph.plot_cartesian_trajectory([-traj_rotvec_N0, -traj_rotvec_N0_d], traj_t, ['rotvec_0N', 'rotvec_0N_d'], 'Rotation from 0 to Tool as rotvec', 'rad')
ph.plot_cartesian_trajectory([traj_omega_T, traj_omega_TCP_d], traj_t, ['omega_TCP', 'omega_TCP_d'], 'Angular Velocity of Tool', 'rad/s')
# error
rot_error = np.linalg.norm(traj_rotvec_N0-traj_rotvec_N0_d,axis=0)
rot_error[rot_error > np.pi] -= 2 * np.pi
ph.plot_trajectory([np.linalg.norm(traj_r_TCP-traj_r_TCP_d,axis=0), rot_error], traj_t, \
    ['|r_TCP - r_TCP_d| in m', '|rotvec_N0 - rotvec_N0_d| in rad'], 'Error from Desired Trajectory','m resp. rad')
ph.plot_angles(traj_q, traj_t, 'q', 'joint angles', 'rad')
plt.show()
