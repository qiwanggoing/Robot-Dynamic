import numpy as np

def get_spline_coefficients(points: np.array, T_part: np.array) -> list:
    """Calculation of the spline coefficients for the individual intervals,
    Interpolation by cubic splines of the form a*t^3 + b*t^2 + c*t + d
    
    numpoints: Number of interpolation points

    Parameters
    ---------
    points: np.array((numpoints))
        Scalar interpolation points

    T_part: np.array((numpoints-1))
        Times for the intervals

    Returns
    --------
    [a, b, c, d] : list[np.array((numpoints-1))]
        Spline coefficients for the individual intervals
    """


    ###Workspace Start###
    numpoints = len(points)
    h = T_part

    # Ax=B
    # Setup matrix A
    A = np.zeros((numpoints - 2, numpoints - 2))
    for i in range(numpoints - 2):
        A[i, i] = 2 * (h[i] + h[i+1])
        if i > 0:
            A[i, i-1] = h[i]
        if i < numpoints - 3:
            A[i, i+1] = h[i+1]

    # Setup vector B
    B = np.zeros(numpoints - 2)
    for i in range(numpoints - 2):
        B[i] = 6 * ((points[i+2] - points[i+1]) / h[i+1] - (points[i+1] - points[i]) / h[i])

    # Solve for ddot_p
    ddot_p_internal = np.linalg.solve(A, B)
    
    ddot_p = np.zeros(numpoints)
    ddot_p[1:-1] = ddot_p_internal

    # Calculate coefficients
    a = np.zeros(numpoints - 1)
    b = np.zeros(numpoints - 1)
    c = np.zeros(numpoints - 1)
    d = np.zeros(numpoints - 1)

    for i in range(numpoints - 1):
        a[i] = (ddot_p[i+1] - ddot_p[i]) / (6 * h[i])
        b[i] = ddot_p[i] / 2
        c[i] = (points[i+1] - points[i]) / h[i] - (h[i] * (ddot_p[i+1] + 2 * ddot_p[i])) / 6
        d[i] = points[i]

    return [a, b, c, d]
    ###Workspace End###

def get_spline_scalar(points: np.array, T_part: np.array, T_list: list):
    """scalar spline interpolation with cubic splines of the form a*t^3 + b*t^2 + c*t + d
    
    - numpoints: Number of interpolation points
    - totalsteps: Number of time steps of the entire trajectory

    Parameters
    ---------
    points: np.array((numpoints))
        scalar interpolation points

    T_part: np.array((numpoints-1))
        Times for the intervals

    Returns
    --------
    [traj, dot_traj, ddot_traj]: list[np.array((totalsteps)), ...]
        scalar trajectory: position traj, velocity dot_traj, acceleration ddot_traj
    """

    totalsteps = sum(np.size(t) for t in T_list) #Number of time steps
    intervals = len(T_list) #Number of intervals

    #Calculation of the spline coefficients
    [a, b, c, d] = get_spline_coefficients(points, T_part)

    #Initialisation of the trajectory
    traj = np.zeros((totalsteps)) #Position
    dot_traj = np.zeros((totalsteps)) #Velocity
    ddot_traj = np.zeros((totalsteps)) #Acceleration

    #loop over all intervals to calculate the trajectory
    index = 0 #to save the current index of the time vector of the entire trajectory
    for i in range(intervals):
        ti = T_list[i][0]
        #Calculate the trajectory for the current interval
        for j in range(len(T_list[i])):
            t = T_list[i][j]
            
            ###Workspace Start###
            dt_interval = t - ti
            traj[index + j] = a[i] * dt_interval**3 + b[i] * dt_interval**2 + c[i] * dt_interval + d[i]
            dot_traj[index + j] = 3 * a[i] * dt_interval**2 + 2 * b[i] * dt_interval + c[i]
            ddot_traj[index + j] = 6 * a[i] * dt_interval + 2 * b[i]
            ###Workspace End###

        index += np.size(T_list[i]) #Save the index of the time vector

    return [traj, dot_traj, ddot_traj]
        
def spline(through_points: np.array, T_part: np.array, dt: float):
    """Planning of a trajectory with cubic spline interpolation between
    the interpolation points

    -numpoints: Number of interpolation points
    -dimension: Dimension of the interpolation points (e.g. positions in the workspace: dimension = 3)
    -totalsteps: Number of time steps of the entire trajectory

    Parameters
    ---------
    through_points: np.array((numpoints,dimension))
        interpolation points
    
    T_part: np.array((numpoints-1))
        Time for the individual intervals
    
    dt: float
        Cycle time

    Returns
    --------
    [T, Traj, dot_Traj, ddot_Traj]: list[np.array((totalsteps)), np.array((dimension,totalsteps)), ...]
        Trajectory: time vector T, positions Traj, velocities dot_Traj, accelerations ddot_Traj

    """  
    intervals = np.size(through_points,0) - 1 #Number of intervals
    size = np.size(through_points,1) #Number of coordinates

    #Initialisation 
    T = np.array([]) #Time vector of the entire trajectory
    T_list = [] #Time vectors of the intervals
    t_start = 0 #Initialisation of the start time
    intervalsteps_list = [] #Number of time steps of the intervals

    #Loop over all intervals to calculate the time vectors
    for i in range(intervals):
        t = np.arange(start = t_start, step=dt, stop=t_start + T_part[i]) #Time vector for interval
        #Save the times
        intervalsteps_list.append(len(t))
        T = np.concatenate((T,t))
        T_list.append(t)
        #Set the start time for the next interval
        t_start = t[-1]

    #Initialisation of the data matrices of the trajectory
    totalsteps = sum(intervalsteps_list) #Number of time steps
    Traj = np.empty((size,totalsteps)) #Positions
    dot_Traj = np.empty((size,totalsteps)) #Speeds
    ddot_Traj = np.empty((size,totalsteps)) #Accelerations

    #Loop over all dimensions of the interpolation points to calculate the trajectory
    for i in range(size):
        [traj, dot_traj, ddot_traj] = get_spline_scalar(through_points[:,i], T_part, T_list)
        Traj[i,:] = traj
        dot_Traj[i,:] = dot_traj
        ddot_Traj[i,:] = ddot_traj

    return [T, Traj, dot_Traj, ddot_Traj]
