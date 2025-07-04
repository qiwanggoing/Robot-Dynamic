import numpy as np

def calculate_velocities(Q_diff: np.array, dot_Q_max: np.array, numpoints: int, size: int) -> np.array:
    """Calculation of the joint angular velocities for the intervals

    Parameters
    ----------

    Q_diff: np.array((size, numpoints-1))
        Difference between the successive waypoints in the joint space in [rad]: Q_i+1-Q_i

    dot_Q_max: np.array((size))
        Maximum joint angular velocities in [rad/s]

    numpoints: int
        Number of waypoints

    size: int
        Number of joints

    Returns
    -------

    dot_Q : np.array((size, numpoints))
        Joint angular velocities in the respective intervals in [rad/s]
        Note: the last column can be set to zero
    """

    ###Workspace Start###
    dot_Q = np.zeros((size, numpoints))
    for i in range(size):
        for j in range(numpoints):
            # print(Q_diff[i, j])
            if j== numpoints-1:
                dot_Q[i, j] = 0
            else:
                # if Q_diff[i, j] != 0:
                    dot_Q[i, j] = np.sign(Q_diff[i, j]) * dot_Q_max[i]
                # else:
                    # print(Q_diff[i, j])
    ###Workspace End###
    # print(dot_Q)
    return dot_Q

def calculate_accelerations(dot_Q: np.array, ddot_Q_max: np.array, numpoints: int, size: int) -> np.array:
    """Calculating the joint angle accelerations when switching

    Parameters
    ----------

    dot_Q: np.array((size, numpoints))
        Joint angular velocities of the intervals in [rad/s]

    ddot_Q_max: np.array((size))
        Maximum joint angle accelerations in [rad/s²]

    numpoints: int
        Number of waypoints

    size: int
        Number of joints

    Returns
    ----------
    ddot_Q : np.array((size, numpoints))
        Joint angle accelerations at the switching points in [rad/s²]
    """

    ###Workspace Start###
    ddot_Q = np.zeros((size, numpoints))
    for i in range(size):
        for j in range(numpoints):
            if j == 0:
                # For the first point
                ddot_Q[i, j] = np.sign(dot_Q[i,j]) * ddot_Q_max[i]
            elif j== numpoints-1:
                # For the last point
                ddot_Q[i, j] = np.sign(-dot_Q[i,j-1]) * ddot_Q_max[i]
            else:
                # For the intermediate points
                ddot_Q[i, j] = np.sign(dot_Q[i, j] - dot_Q[i, j - 1]) * ddot_Q_max[i]  
    ###Workspace End###
    return ddot_Q

def calculate_acc_times(dot_Q: np.array, ddot_Q: np.array, numpoints: int, size: int, tol: float) -> np.array:
    """Calculation of the acceleration times

    Parameters
    ----------

    dot_Q: np.array((size, numpoints))
        Joint angular velocities of the intervals in [rad/s]

    ddot_Q: np.array((size, numpoints))
        Joint angle accelerations at the switching times in [rad/s²]

    numpoints: int
        Number of waypoints

    size: int
        Number of joints

    tol: float
        Tolerance for checking floating point numbers for equality

    Returns
    -------

    t_b: np.array((size, numpoints))
        Acceleration times in [s]
    """

    ###Workspace Start###
    # Hint: Use the tolerance tol to avoid dividing by 0
    t_b = np.zeros((size, numpoints))
    for i in range(size):
        for j in range(numpoints):
            # print(ddot_Q[i, j])
            if abs(ddot_Q[i, j]) > tol:
                if j == 0:
                    # For the first point
                    t_b[i, j] = dot_Q[i, j] / ddot_Q[i, j]
                elif j== numpoints - 1:
                    # For the last point
                    t_b[i, j] = -dot_Q[i, j-1] / ddot_Q[i, j]
                else:
                    # For the intermediate points
                    t_b[i, j] = (dot_Q[i, j] - dot_Q[i, j-1])/ ddot_Q[i, j]
            else:
                    #if ddot_Q[i, j]==0 refers to there is no change or two point are coinside. Assure that tb=0
                    t_b[i, j] = 0
                    # print(f"ddot_Q at joint {i}, point {j} is too small, setting acceleration time to zero.")
    # print(t_b)
    ###Workspace End###
    return t_b

def calculate_total_times(Q_diff: np.array, dot_Q: np.array, t_b: np.array,
                          numpoints: int, size: int, tol: float) -> np.array:
    """Calculation of the total times for the intervals

    Parameters
    ----------

    Q_diff: np.array((size, numpoints-1))
        Difference between the successive waypoints in the joint space in [rad]: Q_i+1-Q_i

    dot_Q: np.array((size, numpoints))
        Joint angular velocities of the intervals in [rad/s]

    t_b: np.array((size, numpoints))
        Acceleration times

    numpoints: int
        Number of waypoints

    size: int
        Number of joints

    tol: float
        Tolerance for checking floating point numbers for equality

    Returns
    -------

    t_g: np.array((size, numpoints-1))
        Total times of the intervals in [s]
    """

    ###Workspace Start###
    t_g = np.zeros((size, numpoints - 1))
    for i in range(size):
        for j in range(numpoints - 1):
            if abs(dot_Q[i, j]) > tol:
                # Calculate t_g for non-zero velocities
                if j == 0:
                    # For the first point
                    t_g[i, j] = Q_diff[i, j] / dot_Q[i, j] + 0.5 * t_b[i, j]
                elif j == numpoints - 2:
                    # For the last point
                    t_g[i, j] = Q_diff[i, j] / dot_Q[i, j] + 0.5 * t_b[i, j + 1]
                else:
                    # For the intermediate points
                    t_g[i, j] = Q_diff[i, j] / dot_Q[i, j]
            else:
                # Handle the case where dot_Q is very small or zero
                if j == 0:
                    # For the first point
                    t_g[i, j] = 0.5 * t_b[i, j]
                elif j == numpoints - 2:
                    # For the last point
                    t_g[i, j] = 0.5 * t_b[i, j + 1]
                else:
                    # For the intermediate points
                    t_g[i, j] = 0
    ###Workspace End###
    # print(t_g)
    return t_g

def calculate_velocities_sync(Q_diff: np.array, ddot_Q: np.array, t_max: np.array,
                              numpoints: int, size: int) -> np.array:
    """Calculation of the new joint angular velocities for synchronisation

    Parameters
    ----------

    Q_diff: np.array((size, numpoints-1))
        Difference between the successive waypoints in the joint space in [rad]: Q_i+1-Q_i

    ddot_Q: np.array((size, numpoints))
        Joint angular accelerations at the switching times in [rad/s²]

    t_max: np.array((numpoints-1))
        Total times of the slowest joints in the intervals in [s]

    numpoints: int
        Number of waypoints

    size: int
        Number of joints

    Returns
    -------

    dot_Q_sync : np.array((size, numpoints))
        new joint angular velocities in the respective intervals after synchronisation in [rad/s]
        Note: the last column can be set to zero
    """


    ###Workspace Start###
    ##Berechnen von dot_Q_sync
    dot_Q_sync = np.zeros((size, numpoints))

    # Synchronize velocities based on the maximum duration
    for i in range(size):
        for j in range(numpoints):
            if j == 0:
                # Calculate the initial velocity for the first segment
                dot_Q_sync[i, j] = ddot_Q[i, j] * t_max[j] - np.sign(ddot_Q[i, j]) * np.sqrt(
                    ddot_Q[i, j] ** 2 * t_max[j] ** 2 - 2 * ddot_Q[i, j] * Q_diff[i, j])
            elif j == numpoints - 2:
                # Calculate the velocity for the last segment, ensuring continuity from previous segment
                dot_Q_sync[i, j] = -ddot_Q[i, j + 1] * t_max[j] + np.sign(ddot_Q[i, j + 1]) * np.sqrt(
                    ddot_Q[i, j + 1] ** 2 * t_max[j] ** 2 + 2 * ddot_Q[i, j + 1] * Q_diff[i, j])
            elif j == numpoints - 1:# terminal point
                dot_Q_sync[i, j] = 0
            else:
                # Calculate velocities for intermediate waypoints
                dot_Q_sync[i, j] = Q_diff[i, j] / t_max[j]
    ###Workspace End###
    return dot_Q_sync

def get_switch_times(t_b:np.array, t_max: np.array, numpoints: int, size: int) -> list:
    """Calculation of the switchover times

    Parameters
    ----------

    t_b: np.array((size, numpoints))
        Acceleration times

    t_max: np.array((numpoints-1))
        Total times of the slowest joints in the intervals in [s]

    numpoints: int
        Number of waypoints

    size: int
        Number of joints

    Returns
    -------

    [t_const2acc, t_acc2const, trajectory_ok]: list[np.array((size,numpoints)), np.array((size,numpoints)), bool]
        t_const2acc: Waypoints from constant speed to acceleration in [s]
        t_acc2const: Waypoints from acceleration to constant speed in [s]
        trajectory_ok: Trajectory can be run with the given conditions
    """

    t_const2acc = np.zeros((size,numpoints))
    t_acc2const = np.zeros((size,numpoints))
    T_stuetz = np.concatenate(([0], np.cumsum(t_max)))

    # Calculation of switching times
    for g in range(size):
        t_const2acc[g,0] = 0.0
        t_acc2const[g,0] = t_b[g,0]
        for i in range(1,numpoints-1):
            t_const2acc[g,i] = T_stuetz[i] - 0.5*t_b[g,i]
            t_acc2const[g,i] = T_stuetz[i] + 0.5*t_b[g,i]
        t_const2acc[g,-1] = T_stuetz[-1] - t_b[g,-1]
        t_acc2const[g,-1] = T_stuetz[-1]

    # Check if trajectory can be run
    trajectory_ok = True
    for g in range(size):
        for i in range(numpoints-1):
            if (t_acc2const[g,i] > t_const2acc[g,i+1]):
                trajectory_ok = False
    return[t_const2acc, t_acc2const, trajectory_ok]

def get_acc_vel(t_const2acc: np.array, t_acc2const: np.array, dot_Q: np.array, ddot_Q: np.array, dt: float,
                numpoints:int, size: int) -> list:
    """Calculation of accelerations and velocities over the entire trajectory

    Parameters
    ----------

    t_const2acc: np.array((size,numpoints))
        Switching points from constant speed to acceleration in [s]

    t_acc2const: np.array((size,numpoints))
        Switching points from acceleration to constant speed in [s]

    dot_Q: np.array((size,numpoints))
        Joint angular velocities of the intervals in [rad/s]

    ddot_Q: np.array((size,numpoints))
        Joint angle accelerations at the changeover times in [rad/s²]

    dt: float
        Time period in which the acceleration is switched

    numpoints: int
        Number of interpolation points

    size: int
        Number of joints

    Returns
    -------

    [dot_q_I, t_dot_q_I, ddot_q_I, t_ddot_q_I]:
            list[np.array((size,numpoints)), np.array((size,numpoints)),
                 np.array((size,numpoints*4-2)), np.array((size,numpoints*4-2))]
        dot_q_I: Joint angular velocities in [rad/s]
        t_dot_q_I: Time points at which dot_q_I applies in [s]
        ddot_q_I: joint angular accelerations in [rad/s^2]
        t_ddot_q_I: Points in time at which ddot_q_I applies in [s]
    """

    t_ddot_q_I = np.zeros((size,numpoints*4))
    ddot_q_I = np.zeros((size,numpoints*4))
    t_dot_q_I = np.zeros((size,numpoints*2))
    dot_q_I = np.zeros((size,numpoints*2))

    # Calculating the accelerations and timing
    for i in range(numpoints*4):
        mod = i%4
        n = (int)(i/4)
        match mod:
            case 0:
                t_ddot_q_I[:,i] = t_const2acc[:,n]
                ddot_q_I[:,i] = np.zeros(size)
            case 1:
                t_ddot_q_I[:,i] = t_const2acc[:,n] + np.ones(size) * dt
                ddot_q_I[:,i] = ddot_Q[:,n]
            case 2:
                t_ddot_q_I[:,i] = t_acc2const[:,n] - np.ones(size) * dt
                ddot_q_I[:,i] = ddot_Q[:,n]
            case 3:
                t_ddot_q_I[:,i] = t_acc2const[:,n]
                ddot_q_I[:,i] = np.zeros(size)

    # Calculating the velocities and timing
    t_dot_q_I[:,0] = t_const2acc[:,0]
    dot_q_I[:,0] = np.zeros(size)
    for i in range(1,numpoints*2-1):
        mod = i%2
        n = (int)(i/2)
        match mod:
            case 0:
                t_dot_q_I[:,i] = t_const2acc[:,n]
                dot_q_I[:,i] = dot_Q[:,n-1]
            case 1:
                t_dot_q_I[:,i] = t_acc2const[:,n] + np.ones(size) * dt
                dot_q_I[:,i] = dot_Q[:,n]
    t_dot_q_I[:,-1] = t_acc2const[:,-1] + np.ones(size) * dt
    dot_q_I[:,-1] = np.zeros(size)

    return[dot_q_I, t_dot_q_I, ddot_q_I, t_ddot_q_I]

def trapez_int(t_dot_Q_stuetz: np.array, dot_Q_stuetz: np.array, T: np.array, Q_0: float)->np.array:
    """Integration using the trapezoidal rule to determine the joint angles

    -n: Number of support speeds
    -totalsteps: Total number of time steps

    Parameters
    ----------

    t_dot_Q_stuetz: np.array((n))
        Points in time at which dot_Q_stuetz applies in [s]

    dot_Q_stuetz: np.array((n))
        Joint angular velocity in [rad/s]

    T: np.array((totalsteps))
        Time vector

    Q_0: float
        Start value for the joint angle in [rad]

    Returns
    -------

    Q: np.array((totalsteps))
        Joint angle in [rad]
    """

    Q = np.zeros((len(T)))
    Q[0] = np.copy(Q_0)
    dot_Q = np.copy(dot_Q_stuetz[0])
    for t in range(1,len(T)):
        dot_Q_last = np.copy(dot_Q)
        dot_Q = np.interp(xp=t_dot_Q_stuetz,fp=dot_Q_stuetz,x=T[t])
        Q[t] = Q[t-1] + (dot_Q_last + dot_Q)/2 * (T[t]-T[t-1])
    return Q

def parabolic_blends(Q_via_points: np.array, dot_Q_max: np.array, ddot_Q_max: np.array, dt: float) -> list:
        """Planning of a trajectory with blending

        Parameters
        ----------

        Q_via_points: np.array((numpoints,size))
            Support points in the joint space

        dot_Q_max: np.array((size))
            Maximum speeds of the joints in [rad/s]

        ddot_Q_max: np.array((size))
            maximum accelerations of the joints in [rad/s^2]

        dt: float
            Time step in [s]


        Returns
        -------

        [T, Q_traj, dot_Q_traj, ddot_Q_traj]: list[np.array((totalsteps)), np.array((size,totalsteps)),...]
            T: Time vector in [s]
            Q_traj: Joint angle in [rad]
            dot_Q_traj: Joint angular velocities in [rad/s]
            ddot_Q_traj: joint angular accelerations in [rad/s^2]
        """

        # Number of waypoints
        numpoints = np.size(Q_via_points,0)
        # Number of degrees of freedom/joints
        size = np.size(Q_via_points,1)
        # Calculate the difference between the waypoints
        Q_diff = np.zeros((size,numpoints-1))
        for i in range(numpoints-1):
            Q_diff[:,i] = Q_via_points[i+1,:]-Q_via_points[i,:]
        # print(Q_diff)
        tol = 10e-5 # Tolerance for floating point number equality
        # Calculating the velocities
        dot_Q = calculate_velocities(Q_diff, dot_Q_max, numpoints, size)
        # Calculate the accelerations
        ddot_Q = calculate_accelerations(dot_Q, ddot_Q_max, numpoints, size)
        # Calculate the acceleration times
        t_b = calculate_acc_times(dot_Q, ddot_Q, numpoints, size, tol)
        # Calculate the total times
        t_g = calculate_total_times(Q_diff, dot_Q, t_b, numpoints, size, tol)
        # Determine the longest duration for synchronisation
        t_max = np.max(t_g,0)
        if np.min(t_max) <= 1e-3:
            print("Two consecutive interpolation points are too close to each other")
            return
        # Recalculation of the velocities
        dot_Q_sync = calculate_velocities_sync(Q_diff, ddot_Q, t_max, numpoints, size)
        # Recalculation of the accelerations
        ddot_Q_sync = calculate_accelerations(dot_Q_sync, ddot_Q_max, numpoints, size)
        # Recalculation of the acceleration times
        t_b_sync = calculate_acc_times(dot_Q_sync, ddot_Q_sync, numpoints, size, tol)
        # Calculation of the switching times
        [t_const2acc, t_acc2const, trajektory_ok] = get_switch_times(t_b_sync, t_max, numpoints, size)
        # Cancel if given interpolation points cannot be reached
        if not trajektory_ok:
            print("This trajectory cannot be run with the given conditions")
            return
        ## Calculation of the control trajectories
        # Initialisation of the times
        t_end = np.sum(t_max)
        
        T = np.arange(start=0, step=dt, stop=t_end)
        T = np.append(T, T[-1]+dt)
        totalsteps = len(T)
        # Initialisation of the trajectory
        Q_traj = np.zeros((size,totalsteps))
        dot_Q_traj = np.zeros((size,totalsteps))
        ddot_Q_traj = np.zeros((size,totalsteps))
        # Calculate velocities/acceleration and times
        [dot_q_I, t_dot_q_I, ddot_q_I, t_ddot_q_I] = \
            get_acc_vel(t_const2acc, t_acc2const, dot_Q_sync, ddot_Q_sync, dt/1000, numpoints, size)
        # Interpolation of the values and integration of the position
        for g in range(size):
            ddot_Q_traj[g,:] = np.interp(xp=t_ddot_q_I[g,:],fp=ddot_q_I[g,:],x=T)
            dot_Q_traj[g,:] = np.interp(xp=t_dot_q_I[g,:],fp=dot_q_I[g,:],x=T)
            Q_traj[g,:] = trapez_int(t_dot_q_I[g,:], dot_q_I[g,:], T, Q_via_points[0,g])
        ddot_Q_traj[:,0] = np.zeros(size)
        return[T, Q_traj, dot_Q_traj, ddot_Q_traj]
