import numpy as np
import os
import sys

import rdplib.robot as rb
import rdplib.ik as ik

# Python 3.10 is the minimum version required for this project
assert sys.version_info >= (3, 10)

# choose robot
robot_name = 'FrankaEmika' # or 'FrankaEmika'
# robot_name = 'ABB_GoFa'

if robot_name == 'ABB_GoFa':
    q_d = np.array([0,0,0,0,0,0])
elif robot_name == 'FrankaEmika':
    q_d = np.array([0,1,-2,1,2,1,0])
else:
    raise NotImplementedError('no q_d for this robot implemented')

# create robot
robot = rb.robot_factory(os.path.join('robot_config_files',robot_name + '.yaml'))

# initialize buffer for visualization
data = np.zeros((3,12*robot.size+1)) # data for the viewer


## Direct Kinematics
robot.calculate_positions(q_d)

# save workspace configuration
r_TCP__0 = robot.r_TCP__0
A_N0 = robot.links[-1].A_i0

# build array for viewer
for l in range(robot.size):
    data[0,0] = 0
    data[0,l*12+1:l*12+4] = np.transpose(robot.links[l].r_i__0)
    data[0,l*12+4:l*12+13] = np.reshape(np.transpose(robot.links[l].A_i0),(1,9))
data[1,0] = 1
data[1,1:] = data[0,1:]


## Inverse Kinematics with Damped Least Square
ik_solver_positions_dls = ik.IKSolverPositionDLS()
q_ik = ik_solver_positions_dls.calculate_joint_angles(robot, \
    r_TCP__0, \
    A_N0, \
    robot.q_home)

# direct kinematics with the result to be able to view it
robot.calculate_positions(q_ik)

# build array for viewer
for l in range(robot.size):
    data[2,0] = 2
    data[2,l*12+1:l*12+4] = np.transpose(robot.links[l].r_i__0)
    data[2,l*12+4:l*12+13] = np.reshape(np.transpose(robot.links[l].A_i0),(1,9))

# save data for viewer
np.savetxt('trajectory.csv', data, delimiter="\t", fmt='%.4f')