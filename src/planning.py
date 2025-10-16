from enum import Enum
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

import rdplib.ik as ik
import rdplib.plot_helpers as ph
from rdplib.planning.Blends import parabolic_blends
from rdplib.planning.PTP import PTP_quintic
from rdplib.planning.Spline import spline
import rdplib.robot as rb
# import matplotlib
# matplotlib.use('TkAgg')

# Python 3.10 is the minimum version required for this project
assert sys.version_info >= (3, 10)

# create robot
robot = rb.robot_factory(os.path.join('robot_config_files','ABB_GoFa.yaml'))

# read waypoints
waypoints_r_TCP__0 = np.genfromtxt(os.path.join('planning_input','ABB_GoFa','waypoints_r_TCP.csv'), delimiter=',')
waypoints_A_N0 = np.genfromtxt(os.path.join('planning_input','ABB_GoFa','waypoints_A_N0.csv'), delimiter=',')
numwaypoints = np.size(waypoints_r_TCP__0,0)

# settings
# planning option
PlanningOptions = Enum('PlanningOptions', ['PTPJOINTSPACE',
                                            'PTPWORKSPACE',
                                            'SPLINEJOINTSPACE',
                                            'SPLINEWORKSPACE',
                                            'BLENDSJOINTSPACE',
                                            'BLENDSWORKSPACE'
                                            ])
# opt = PlanningOptions.PTPJOINTSPACE
# opt = PlanningOptions.PTPWORKSPACE
# opt = PlanningOptions.SPLINEJOINTSPACE
# opt = PlanningOptions.SPLINEWORKSPACE
# opt = PlanningOptions.BLENDSJOINTSPACE
opt = PlanningOptions.BLENDSWORKSPACE
# time between waypoints
T_part = np.ones((numwaypoints-1))
# time step for discretization
dt = 0.004 # [s]
# parameters for blending
dot_Q_max = np.ones((6)) * 1.0 # max joint velocity [rad/s]
ddot_Q_max = np.ones((6)) * 32 # max joint acceleration [rad/s²]
dot_W_max = np.ones((3)) * 0.1 # max translational workspace velocity [m/s]
ddot_W_max = np.ones((3)) * 0.5 # max translational workspace acceleration [m/s²]

# calculate waypoints in jointspace
waypoints_jointspace = np.zeros((numwaypoints,robot.size))
# use position based IK solver
ik_solver_positions_dls = ik.IKSolverPositionDLS()
q_start = robot.q_home
for i in range(numwaypoints):
    waypoints_jointspace[i,:] = ik_solver_positions_dls.calculate_joint_angles(robot, \
        waypoints_r_TCP__0[i,:], \
        np.reshape(waypoints_A_N0[i,:],(3,3)), \
        q_start)
    q_start = waypoints_jointspace[i,:]

# initialize matrizes for the results
totalsteps = sum(int(t/dt) for t in T_part)
# data for the viewer
data = np.zeros((totalsteps,12*robot.size+1))
r_TCP__0 = np.zeros((3,totalsteps))
v_TCP__0 = np.zeros((3,totalsteps))
q = np.zeros((robot.size,totalsteps))
dot_q = np.zeros((robot.size,totalsteps))
ddot_q = np.zeros((robot.size,totalsteps))

# plan trajectory
if opt == PlanningOptions.PTPJOINTSPACE:

    [T, q, dot_q, ddot_q] = PTP_quintic(waypoints_jointspace, T_part, dt)
    print(T.shape)

elif opt == PlanningOptions.PTPWORKSPACE:

    [T, r_TCP__0, v_TCP__0, ddot_W] = PTP_quintic(waypoints_r_TCP__0, T_part, dt)

elif opt == PlanningOptions.SPLINEJOINTSPACE:

    [T, q, dot_q, ddot_q] = spline(waypoints_jointspace, T_part, dt)

elif opt == PlanningOptions.SPLINEWORKSPACE:

    [T, r_TCP__0, v_TCP__0, ddot_W] = spline(waypoints_r_TCP__0, T_part, dt)

elif opt == PlanningOptions.BLENDSJOINTSPACE:

    [T, q, dot_q, ddot_q] = parabolic_blends(waypoints_jointspace, dot_Q_max, ddot_Q_max, dt)

    # reinitialize matrizes for the results
    totalsteps = len(T)
    r_TCP__0 = np.zeros((3,totalsteps))
    v_TCP__0 = np.zeros((3,totalsteps))
    data = np.zeros((totalsteps,12*robot.size+1))
    data_egm = np.zeros((totalsteps,8))

elif opt == PlanningOptions.BLENDSWORKSPACE:

    [T, r_TCP__0, v_TCP__0, ddot_W] = parabolic_blends(waypoints_r_TCP__0, dot_W_max, ddot_W_max, dt)

    # reinitialize matrizes for the results
    totalsteps = len(T)
    q = np.zeros((6,totalsteps))
    dot_q = np.zeros((6,totalsteps))
    data = np.zeros((totalsteps,12*robot.size+1))
    data_egm = np.zeros((totalsteps,8))

else:
    raise NotImplementedError('planning option not implemented')

# direct kinematics for trajectories planned in the joint space
if opt in (PlanningOptions.PTPJOINTSPACE, PlanningOptions.SPLINEJOINTSPACE, PlanningOptions.BLENDSJOINTSPACE):
    for i in range(totalsteps):
        robot.calculate_positions(q[:,i])
        robot.calculate_velocities(dot_q[:,i])
        r_TCP__0[:,i] = robot.r_TCP__0
        v_TCP__0[:,i] = robot.v_TCP__0
        # save data for the viewer
        for l in range(robot.size):
            data[i,0] = T[i]
            data[i,l*12+1:l*12+4] = np.transpose(robot.links[l].r_i__0)
            data[i,l*12+4:l*12+13] = np.reshape(np.transpose(robot.links[l].A_i0),(1,9))

# inverse kinematics for trajectories planned in the workspace
elif opt in (PlanningOptions.PTPWORKSPACE, PlanningOptions.SPLINEWORKSPACE, PlanningOptions.BLENDSWORKSPACE):

    # set up velocity based IK solver
    W_solver = np.diag([10.0, 10.0, 10.0, 10.0, 1.0, 1.0])
    solver = ik.IKSolverVelocity(W_solver,
                    k_position=10.0,
                    k_orientation=10.0,
                    dt=dt,
                    alpha = 10,
                    opt=ik.IKOptions.COMFORT)

    # set initial position and jacobian
    robot.calculate_positions(waypoints_jointspace[0,:])
    robot.calculate_jacobis()
    for i, t in enumerate(T):
        [q_ik, dot_q_ik, _] = solver.calculate_ik_rmc(robot, v_TCP__0[:,i], r_TCP__0[:,i],
                                             np.zeros((3)), np.reshape(waypoints_A_N0[0,:], (3,3)))
        q[:,i] = q_ik
        dot_q[:,i] = dot_q_ik
        robot.calculate_positions(q_ik)
        robot.calculate_velocities(dot_q_ik)
        robot.calculate_jacobis()
        # save data for the viewer
        for l in range(robot.size):
            data[i,0] = T[i]
            data[i,l*12+1:l*12+4] = np.transpose(robot.links[l].r_i__0)
            data[i,l*12+4:l*12+13] = np.reshape(np.transpose(robot.links[l].A_i0),(1,9))
else:
    raise NotImplementedError('post processing not implemented for this option')

# write data to file
np.savetxt('trajectory.csv', data, delimiter="\t", fmt='%.4f')

# visualization
# path in workspace
ph.plot_cartesian_trajectory_with_points(r_TCP__0, waypoints_r_TCP__0)
# joint angles
ph.plot_angles(q, T, 'q', 'joint angles', 'rad')
# joint velocity
ph.plot_angles(dot_q, T, 'q', 'joint velocity', 'rad/s')
# joint acceleration -> only for planning in joint space
if(opt == PlanningOptions.PTPJOINTSPACE or
   opt == PlanningOptions.SPLINEJOINTSPACE or
   opt == PlanningOptions.BLENDSJOINTSPACE):
    ph.plot_angles(ddot_q, T, 'q', 'joint acceleration', 'rad/s²')

# position of TCP
ph.plot_cartesian_trajectory(r_TCP__0, T, 'r_TCP', 'Position of TCP', 'm')
# velocity of TCP
ph.plot_cartesian_trajectory(v_TCP__0, T, 'v_TCP', 'Velocity of TCP', 'm/s')
# acceleration of TCP -> only for planning in workspace
if opt in (PlanningOptions.PTPWORKSPACE, PlanningOptions.SPLINEWORKSPACE, PlanningOptions.BLENDSWORKSPACE):
    ph.plot_angles(ddot_W, T, 'a_TCP', 'Acceleration of TCP', 'm/s²')
plt.show()
