import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import rdplib.robot as rb
import rdplib.plot_helpers as ph

# Python 3.10 is the minimum version required for this project
assert sys.version_info >= (3, 10)

# create robot
robot = rb.robot_factory(os.path.join('robot_config_files','SCHUNK_Power_Cube.yaml'))

# import trajectory data
files_path = os.path.join('dynamics_input_output', 'SCHUNK_Power_Cube')
Q = np.genfromtxt(os.path.join(files_path, 'q.csv'), delimiter=',')
dot_Q = np.genfromtxt(os.path.join(files_path, 'dot_q.csv'), delimiter=',')
ddot_Q = np.genfromtxt(os.path.join(files_path, 'ddot_q.csv'), delimiter=',')
# time array
t = np.genfromtxt(os.path.join(files_path, 't.csv'), delimiter=',')
numsteps = len(t)

# initialize matrizes for the results
tau_id = np.zeros((robot.size,numsteps))

# calculate inverse dynamics
for i in range(numsteps):
    robot.calculate_positions(Q[:,i])
    robot.calculate_velocities(dot_Q[:,i])
    robot.calculate_accelerations(ddot_Q[:,i])
    robot.calculate_jacobis()
    robot.calculate_id()
    tau_id[:,i] = robot.tau_id

# write joint torques to csv
np.savetxt(os.path.join(files_path,'tau_id.csv'), tau_id, delimiter=',')

# visualize results
ph.plot_angles(tau_id, t, 'tau', 'joint torques', 'Nm')
gear_ratio = 100.0
ph.plot_rot_speed_torque(np.abs(gear_ratio * dot_Q * 60 / (2*np.pi)),
                         np.abs(tau_id/gear_ratio), 
                         'Torque-Speed Plot (motor side)')
plt.show()
