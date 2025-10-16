from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import rdplib.robot as rb
import rdplib.plot_helpers as ph

# Python 3.10 is the minimum version required for this project
assert sys.version_info >= (3, 10)

# ATTENTION: You need the resulting joint torques of inverse dynamics!
#            Run the script inverse_dyanmics.py first!

# setting
TIME_INTEGRATION = Enum('SOLVER_OPTIONS', ['euler', 'rk45'])
SCENARIO = Enum('SCENARIO', ['integration' , 'real'])

substeps = 1 # steps per inported time step
scenario = SCENARIO.integration
# scenario = SCENARIO.real

if scenario == SCENARIO.integration:
    time_integration = TIME_INTEGRATION.rk45
    pd_controller = False
    # p and d gain
    K_p = 10 * np.eye(6)
    K_d = 0.1 * np.eye(6)
    # disturbance False/True and standard deviation
    disturbance = False
    std = 0.1
    disturbance_rate = 10 # noise all x timesteps
elif scenario == SCENARIO.real:
    time_integration = TIME_INTEGRATION.euler
    pd_controller = True
    K_p = 10 * np.eye(6)
    K_d = 0.1 * np.eye(6)
    disturbance = True
    std = 0.1
    disturbance_rate = 10
else:
    raise NotImplementedError('scenario not implemented')

# create robot
robot = rb.robot_factory(os.path.join('robot_config_files','SCHUNK_Power_Cube.yaml'))

# import trajectory data
q_import = np.genfromtxt(os.path.join('dynamics_input_output', 'SCHUNK_Power_Cube', 'q.csv'), delimiter=',')
dot_q_import = np.genfromtxt(os.path.join('dynamics_input_output', 'SCHUNK_Power_Cube', 'dot_q.csv'), delimiter=',')
ddot_q_import = np.genfromtxt(os.path.join('dynamics_input_output', 'SCHUNK_Power_Cube', 'ddot_q.csv'), delimiter=',')
T_import = np.genfromtxt(os.path.join('dynamics_input_output', 'SCHUNK_Power_Cube', 't.csv'), delimiter=',')
dt_import = np.genfromtxt(os.path.join('dynamics_input_output', 'SCHUNK_Power_Cube', 'dt.csv'), delimiter=',')
tau_id_import = np.genfromtxt(os.path.join('dynamics_input_output', 'SCHUNK_Power_Cube', 'tau_id.csv'), delimiter=',')
numsteps_import = len(T_import)

# interpolate imported data dependent on substeps
dt = dt_import/substeps
T = np.arange(0,T_import[-1]+dt,dt)
numsteps = len(T)
tau_id = np.empty((robot.size,numsteps))
q = np.empty((robot.size,numsteps))
dot_q = np.empty((robot.size,numsteps))
ddot_q = np.empty((robot.size,numsteps))
for i in range(robot.size):
    tau_id[i,:] =np.interp(x=T, xp=T_import, fp=tau_id_import[i,:])
    q[i,:] = np.interp(x=T, xp=T_import, fp=q_import[i,:])
    dot_q[i,:] = np.interp(x=T, xp=T_import, fp=dot_q_import[i,:])
    ddot_q[i,:] = np.interp(x=T, xp=T_import, fp=ddot_q_import[i,:])

# white noise
disturbance_matrix_size = (robot.size,int(np.ceil(numsteps/int(disturbance_rate*substeps))))
whiteNoise = np.zeros((robot.size, numsteps))
whiteNoise[:,0::int(disturbance_rate*substeps)] = \
    np.random.normal(loc=0.0,scale=std,size=disturbance_matrix_size)

# initialize matrizes for the results
q_sim = np.zeros((6,numsteps))
dot_q_sim = np.zeros((6,numsteps))
ddot_q_sim = np.zeros((6,numsteps))
# error
e = np.zeros((6,numsteps))


## simulation

# set robot to initial position
robot.calculate_positions(q[:,0])
robot.calculate_velocities(dot_q[:,0])
robot.calculate_jacobis()

for i in range(numsteps):

    # save data
    q_sim[:,i] = robot.q
    dot_q_sim[:,i] = robot.dot_q
    e[:,i] = q[:,i] - robot.q

    # closed loop control
    if pd_controller:
        e_q = q[:,i] - robot.q
        e_dot_q = dot_q[:,i] - robot.dot_q
        e_ddot_q = ddot_q[:,i] - robot.ddot_q
        robot.tau_antrieb = tau_id[:,i] + K_p @ e_q + K_d @ e_dot_q
    else:
        robot.tau_antrieb = tau_id[:,i]

    # noise
    if disturbance:
        robot.tau_antrieb += whiteNoise[:,i]

    # time integration
    if time_integration == TIME_INTEGRATION.rk45:
        robot.calculate_tmstp_rk45(dt)
    elif time_integration == TIME_INTEGRATION.euler:
        robot.calculate_tmstp_euler(dt)
    else:
        raise NotImplementedError('time integration option not implemented')


# visualization
# joint angles
ph.plot_angles([q_sim, q],T,['q_simulation', 'q_desired'],'simulated and desired joint angles', 'rad')
# joint angle error
ph.plot_angles(e,T,'err','joint angle error','rad')
plt.show()