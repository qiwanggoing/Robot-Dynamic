from enum import Enum
from numpy.linalg import inv
import numpy as np
from scipy.spatial.transform import Rotation as R

from .robot import Robot

IKOptions = Enum('IKOptions', ['COMFORT', 'LIMIT', 'BOTH'])

class IKSolverVelocity:
    """Solver for Velocity Inverse kinematics with the method \
    Resolved Motion Rate Control and Automatic Supervisory Control

    In the following size = number of DoFs of the robot

    Attributes
    ----------
    W : np.array((size,size))
        quadratic weighting matrix

    W_inv: np.array((size,size))
        inverse of the quadratic weighting matrix \\
        calculated once in the constructor \\
        *attention*: if you change W, you have to recompute W_inv

    k_position: float
        gain for the translation drift compensation

    k_orientation: float
        gain for the rotational drift compensation

    alpha: float
        weighting alpha for Automatic Supervisory Control

    dt: float
        time step in seconds

    opt: IKOptions
        only relevant for Automatic Supervisory Control
    """

    W : np.array
    W_inv: np.array
    k_position : float
    k_orientation: float
    alpha: float
    dt: float
    opt : IKOptions

    def __init__(self, W: np.array, k_position: float, dt: float, \
                 k_orientation: float, alpha: float = 1.0, opt=IKOptions.LIMIT) -> None:
        self.W = W
        self.W_inv = inv(W)
        self.k_position = k_position
        self.k_orientation = k_orientation
        self.alpha = alpha
        self.dt = dt
        self.opt = opt

    def H_LIMIT(self, rob: Robot) -> float:
        """
        Calculation of the quality function to avoid joint limits

        Parameters
        ----------
        rob : Robot
            robot object with current values for position

        Returns
        ---------
        H: float
            value of the quality function
        """


        ###Workspace Start###
        H = 0.0
        q_curr = rob.q
        q_min = rob.q_min
        q_max = rob.q_max

        for i in range(rob.size):
            H += (1.0 / (q_curr[i] - q_min[i])**2) + (1.0 / (q_max[i] - q_curr[i])**2)

        return H
        ###Workspace End###

    def grad_H_LIMIT(self, rob: Robot) -> np.array:
        """Calculation of the gradient of the quality function to avoid joint limits

        Parameters
        ----------
        rob : Robot
            robot object with current values for position

        Returns
        ---------
        grad_H: np.array((n))
            gradient of the quality function
        """


        ###Workspace Start###
        grad_H = np.zeros(rob.size)
        q_curr = rob.q
        q_min = rob.q_min
        q_max = rob.q_max

        # This assumes q_min and q_max are valid and have been loaded.
        # No checks for None or mismatched lengths as per user request.

        for i in range(rob.size):
            # The partial derivative of H with respect to q_curr_i
            grad_H[i] = -2.0 / ((q_curr[i] - q_min[i]) ** 3) + 2.0 / ((q_max[i] - q_curr[i]) ** 3)

        return grad_H
        ###Workspace End###

    def H_COMFORT(self, rob: Robot) -> float:
        """Calculation of the quality function for favoring a comfort joint angle position

        Parameters
        ----------
        rob : Robot
            robot object with current values for position

        Returns
        ---------
        H: float
            value of the quality function
        """


        ###Workspace Start###
        H = 0.0
        q_curr = rob.q
        q_comfort = rob.q_comfort

        for i in range(rob.size):
            H += 0.5 * (q_curr[i] - q_comfort[i])**2

        return H
        ###Workspace End###

    def grad_H_COMFORT(self, rob: Robot) -> np.array:
        """Calculation of the gradient of the quality function
        for favoring a comfort joint angle position

        Parameters
        ----------
        rob : Robot
            robot object with current values for position

        Returns
        ---------
        grad_H: np.array((n))
            gradient of the quality function
        """


        ###Workspace Start###
        q_curr = rob.q
        q_comfort = rob.q_comfort

        grad_H = q_curr - q_comfort

        return grad_H
        ###Workspace End###

    def calculate_ik_rmc(self, rob: Robot, v_TCP_d__0: np.array((3)), r_TCP_d__0: np.array((3)), \
                         omega_TCP_d__0:np.array((3)) = np.zeros((3)), A_N0_d:np.array((3,3)) = np.eye(3)) -> list:
        """Calculation of the Inverse Kinematics with the method Resolved Motion Rate Control

        Parameters
        ----------
        rob : Robot
            robot object with current values for position, velocity, and jacobi matrices

        v_TCP_d__0: np.array((3))
            desired velocity of the TCP represented in the fixed cosy 0

        r_TCP_d__0: np.array((3))
            desired position of the TCP represented in the fixed cosy 0

        omega_TCP_d__0: np.array((3))
            desired angular velocity of the TCP represented in the fixed cosy 0

        A_N0_d: np.array((3,3))
            desired orientation of the tool represented as a basis transform matrix from 0 to the cosy of the last body
        """


        ###Workspace Start###
        # jacobi matrix for the workspace
        J_TTCP__0 = rob.J_TTCP__0
        J_RTCP__0 = rob.J_RTCP__0
        J = np.vstack((J_TTCP__0, J_RTCP__0))
        # calculate pseudoinverse
        J_pseudo = self.W_inv @ J.T @ inv(J @ self.W_inv @ J.T)

        # calculate effectively desired workspace velocity
        # current position from DK
        curr_TCP_position = rob.r_TCP__0
        pos_error =  r_TCP_d__0 - curr_TCP_position
        # linear velocity eff
        linear_velocity_d_eff = v_TCP_d__0 + self.k_position * pos_error

        #angular velocity is derivative of orientation
        A_0N = (rob.links[rob.size - 1].A_i0).T
        # Relative rotation from current N to desired Nd
        A_0_Nd = A_N0_d.T
        A_N_Nd = A_0N.T @ A_0_Nd  #

        # Convert relative rotation matrix to axis-angle representation(rotation vector)
        rot_error_object = R.from_matrix(A_N_Nd)
        rot_error_N_Nd = rot_error_object.as_rotvec()
        # Transform rotation vector (axis-angle error) to world frame 0
        omega_error__0 = A_0N @ rot_error_N_Nd

        # Effective desired angular velocity
        omega_d_eff = omega_TCP_d__0 + self.k_orientation * omega_error__0

        #workspace_velocity_eff = ...
        workspace_velocity_eff = np.hstack([linear_velocity_d_eff, omega_d_eff])

        # joint velocity
        dot_q = J_pseudo @ workspace_velocity_eff

        # numerical integration to joint position
        q = rob.q + dot_q * self.dt
        ###Workspace End###
        H = 0.0

        return [q, dot_q, H]

    def calculate_ik_asc(self, rob: Robot, v_TCP_d__0: np.array((3)), r_TCP_d__0: np.array((3)), \
                         omega_TCP_d__0:np.array((3)) = np.zeros((3)), A_N0_d:np.array((3,3)) = np.eye(3)):
        """Calculation of the Inverse Kinematics with the method Resolved Automatic Supervisory Control

        Parameters
        ----------
        rob : Robot
            robot object with current values for position, velocity, and jacobi matrices

        v_TCP_d__0: np.array((3))
            desired velocity of the TCP represented in the fixed cosy 0

        r_TCP_d__0: np.array((3))
            desired position of the TCP represented in the fixed cosy 0

        omega_TCP_d__0: np.array((3))
            desired angular velocity of the TCP represented in the fixed cosy 0

        A_N0_d: np.array((3,3))
            desired orientation of the tool represented as a basis transform matrix from 0 to the cosy of the last body
        """


        ###Workspace Start###
        ##Auswaehlen der H-Funktion
        if self.opt == IKOptions.LIMIT:
            H = self.H_LIMIT(rob)
            grad_H = self.grad_H_LIMIT(rob)
        elif self.opt == IKOptions.COMFORT:
            grad_H = self.grad_H_COMFORT(rob)
            H = self.H_COMFORT(rob)
        else:
            grad_H = self.grad_H_LIMIT(rob) + self.grad_H_COMFORT(rob)
            H = self.H_COMFORT(rob) + self.H_LIMIT(rob)

        ##Zusammensetzen der Jakobimatrix für den gültigen Arbeitsraum
        # jacobi matrix for the workspace
        J_TTCP__0 = rob.J_TTCP__0
        J_RTCP__0 = rob.J_RTCP__0
        J = np.vstack((J_TTCP__0, J_RTCP__0))
        # calculate pseudoinverse
        J_pseudo = self.W_inv @ J.T @ inv(J @ self.W_inv @ J.T)

        # calculate effectively desired workspace velocity
        # current position from DK
        curr_TCP_position = rob.r_TCP__0
        pos_error = r_TCP_d__0 - curr_TCP_position
        # linear velocity eff
        linear_velocity_d_eff = v_TCP_d__0 + self.k_position * pos_error

        # angular velocity is derivative of orientation
        A_0N = (rob.links[rob.size - 1].A_i0).T
        # Relative rotation from current N to desired Nd
        A_0_Nd = A_N0_d.T
        A_N_Nd = A_0N.T @ A_0_Nd  #

        # Convert relative rotation matrix to axis-angle representation(rotation vector)
        rot_error_object = R.from_matrix(A_N_Nd)
        rot_error_N_Nd = rot_error_object.as_rotvec()
        # Transform rotation vector (axis-angle error) to world frame 0
        omega_error__0 = A_0N @ rot_error_N_Nd

        # Effective desired angular velocity
        omega_d_eff = omega_TCP_d__0 + self.k_orientation * omega_error__0

        # workspace_velocity_eff = ...
        workspace_velocity_eff = np.hstack([linear_velocity_d_eff, omega_d_eff])

        ##Berechnung der neuen Geschwindigkeit dot_q durch Automatic Supervisory Control
        # dot_q = J#·w_eff_d - alpha*N_w·W_inv·grad_H.T
        # E is an identity matrix need to match J# * J 's dimension
        E = np.eye((max(J_pseudo.shape[0], J.shape[1])))
        # N_w = E - J# * J
        N_w = E - J_pseudo @ J
        dot_q = J_pseudo @ workspace_velocity_eff.T - self.alpha * N_w @ self.W_inv @ grad_H.T

        ##Berechnung der Position durch numerische Integration
        q = rob.q + dot_q * self.dt

        return [q, dot_q, H]
        ###Workspace End###

class IKSolverPositionDLS:
    """Solver for Position Inverse kinematics with the Method Damped Least Squares

    Attributes:
    ---------
    maxiter: int
        maximum number of iterations for the line search
    
    maxerror: float
        maximum error

    damping_factor: float
        damping constant
        -> larger value: better stability near singularities, but worse convergence
    
    position_factor: float
        stepsize factor for translation (default: 1.0)

    rotation_factor: float
        stepsize factor for rotation (default: 1.0)
    
    Source:
    ------------
    Buss, Samuel R.
    "Introduction to inverse kinematics with jacobian transpose, 
    pseudoinverse and damped least squares methods."
    IEEE Journal of robotics and Automation 17.1-19 (2004): 16.
    """

    maxiter: int
    maxerror: float
    damping_factor: float
    position_factor: float
    rotation_factor: float

    def __init__(self, maxiter: int = 100, tol: float = 0.01, damping_factor : float = 0.01,
                 position_factor: float = 1.0, rotation_factor: float = 1.0) -> None:
        self.maxiter = maxiter
        self.maxerror = tol
        self.damping_factor = damping_factor
        self.position_factor = position_factor
        self.rotation_factor = rotation_factor
        
    def calculate_joint_angles(self, rob: Robot, r_TCP_d__0: np.array, \
                                A_N0_d: np.array, q_start: np.array) -> np.array:
        """Calculation of the joint angle from the desired position and orientation
        
        Parameters
        ---------
        rob: Robot
            robot object

        r_TCP_d__0: np.array((3))
            desired position of the TCP represented in the fixed cosy 0

        A_N0_d: np.array((3,3))
            desired orientation of the tool represented as a basis transform matrix from 0 to TCP
        
        q_start: np.array((size))
            initial guess for the joint configuration

        Returns
        -------
        q: np.array((size))
            joint configuration
        """
        
        # initializing
        q = np.copy(q_start) # copy start configuration, it gets changed in the loop
        counter = 0 # iteration counter

        while True:

            ###Workspace Start###
            # direct kinematics
            rob.calculate_positions(q)
            rob.calculate_jacobis()

            # jacobi matrix for the workspace
            J_TTCP__0 = rob.J_TTCP__0
            J_RTCP__0 = rob.J_RTCP__0
            J = np.vstack((J_TTCP__0, J_RTCP__0))

            # position error
            curr_TCP_position = rob.r_TCP__0
            pos_error = r_TCP_d__0 - curr_TCP_position

            # rotation error
            A_0N = (rob.links[rob.size - 1].A_i0).T
            A_0_Nd = A_N0_d.T
            A_N_Nd = A_0N.T @ A_0_Nd  #
            rot_error_object = R.from_matrix(A_N_Nd)
            rot_error_N_Nd = rot_error_object.as_rotvec()
            rot_error = A_0N @ rot_error_N_Nd

            # total error
            total_error = np.hstack((pos_error, rot_error))

            # identity matrix I
            I = np.eye((max(J.shape[0], J.T.shape[1])))
            # update joint configuration
            q += J.T @ inv(J @ J.T + (self.damping_factor ** 2) * I) @ total_error
            ###Workspace End###

            counter += 1
            # abort if the maximum number of iterations is reached
            if(counter>self.maxiter):
                print('\nwarning: IKSolverPositionDLS reached maximum number of iterations!\n')
                break
            # abort if the accuracy (maxerror) is reached
            if self.position_factor * np.linalg.norm(pos_error) \
                + self.rotation_factor * np.linalg.norm(rot_error) < self.maxerror:
                break

        return q