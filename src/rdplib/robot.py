import json
import numpy as np
import pathlib
from scipy.integrate import solve_ivp
import yaml

from . import math_helpers as mh

# define class to allow links to be attributes of links
class Link:
    pass

class Link:
    """one body of a serial robot with rotational joints


    Vectors
    ------------------
    - r position
    - v translational velocity
    - a translational acceleration
    - omega angular velocity
    - alpha angular acceleration

    Points in the indizes
    - i origin of the body
    - p origin of the predecessor
    - s origin of the successor
    - CoM center of mass of this body
    
    The indices after one underscore give the two points/bodies between which the vector is the difference. E.G., r_iCoM__i is the vector from i to CoM. A leading 0 is neglected, e.g., v_i__i is the translational velocity of i with respect to 0.

    The index after two underscores gives the cosy in which the vector is represented.
    E.G., r_iCoM__i is represented in the body coordinate system i


    Units
    -----
    - lengths are given in meter
    - angle are given in radians
    - times are given in seconds

    Attributes
    ----------
    dof: int
        number of this body
        - number of the DoF starting at the base
        - first body: dof=1
        - base: dof=0

    alpha_p: float
        angle alpha, DH parameter, time-independent

    a_p: float
        length a, DH parameter, time-independent

    d_i: float
        length d, DH parameter, time-independent

    r_pi__p: np.array((3))
        see module description for an explanation of vectors

    r_i__0: np.array((3))
        see module description for an explanation of vectors

    r_iCoM__i: np.array((3))
        see module description for an explanation of vectors

    A_ip: np.array((3,3))
        base transformation from predecessor cosy to this body cosy

    A_i0: np.array((3,3))
        base transformation from world fixed cosy 0 to this body cosy

    omega_pi__i: np.array((3))
        see module description for an explanation of vectors

    omega_i__i: np.array((3))
        see module description for an explanation of vectors

    v_pi__i: np.array((3))
        see module description for an explanation of vectors

    v_i__i: np.array((3))
        see module description for an explanation of vectors

    alpha_i__i: np.array((3))
        see module description for an explanation of vectors

    a_i__i: np.array((3))
        see module description for an explanation of vectors

    a_CoM__i: np.array((3))
        see module description for an explanation of vectors

    J_Ti__i: np.array((3,n))
        translational jacobi matrix of the origin i represented in the body cosy

    J_R__i: np.array((3,n))
        rotational jacobi matrix of this body represented in the body cosy

    Theta_i_i__i: np.array((3,3))
        mass moment of inertia with respect to the origin of the body cosy

    m: float
        mass in kilo grams

    dM: np.array((size,size))
        part of the total mass matrix in kilo grams

    tau_id: np.array((size))
        part of this link to the joint torques (used in inverse dynamics)
    """
    # number of this body
    dof = -1
    # fix DH parameters
    alpha_p = 0.0
    a_p = 0.0
    d_i = 0.0
    # position and orientation
    r_pi__p = np.zeros((3))
    r_i__0 = np.zeros((3))
    r_iCoM__i = np.zeros((3))
    A_ip = np.eye(3)
    A_i0 = np.eye(3)
    # velocities
    omega_i__i = np.zeros((3))
    v_i__i = np.zeros((3))
    # accelerations
    alpha_i__i = np.zeros((3))
    a_i__i = np.zeros((3))
    a_CoM__i = np.zeros((3))
    # Jacobi matrices
    J_R__i = np.array
    J_Ti__i = np.array
    # inertia parameters
    Theta_i_i__i = np.zeros((3,3))
    m = 0.0
    dM = np.array
    # for id
    tau_id = np.array

    def __init__(self, dof: int, size: int) -> None:
        """constructor

        Parameters
        ----------
        dof: int
            number of this body
            - number of the DoF starting at the base
            - first body: dof=1
            - base: dof=0

        size: int
            number of DoFs of the total robot

        Returns
        ----------
        None
        """
        self.dof = dof
        self.J_R__i = np.zeros((3, size))
        self.J_Ti__i = np.zeros((3, size))
        self.dM = np.zeros((size, size))
        self.tau_id = np.zeros((size))

    def set_dh_parameters(self, alpha_p: float, a_p: float, d_i: float) -> None:
        """initialization of the DH parameters

        Parameters
        ----------
        alpha_p: float
            alpha_p in radians

        a_p: float
            a_p in meter

        d_i: float
            d_i in meter

        Returns
        ----------
        None
        """
        self.alpha_p = alpha_p
        self.a_p = a_p
        self.d_i = d_i

    def set_inertia_parameters(self, mass: float, Theta_i_i__i: np.array, r_iCoM__i: np.array) -> None:
        """initialization of inertia parameters

        Parameters
        ----------
        mass: float
            mass in kilo grams

        Theta_i_i__i: np.array((3,3))
            mass moment of inertia with respect to the origin of the body cosy

        r_iCoM__i: np.array((3))
            positionvector from the origin of the body cosy to the center of mass

        Returns
        ----------
        None
        """
        self.m = mass
        self.Theta_i_i__i = Theta_i_i__i
        self.r_iCoM__i = r_iCoM__i

    def calculate_positions(self, prev: Link, q: float)->None:
        """calculation of the position and orientation of this link based on the position and orientation of the predecessor

        Parameters
        -------
        prev: Link
            predecessor

        q: float
            joint angle

        Returns
        ------
        None

        """

        ###Workspace Start###
        # vector from predecessor to this link
        # r_pi__p from p to i current link wrt coordinate p
        # self.r_pi__p = ...
        self.r_pi__p = np.array([
            self.a_p,
            -np.sin(self.alpha_p) * self.d_i,
            np.cos(self.alpha_p) * self.d_i,
        ])
        # base transformation from predecessor to this link
        # self.A_ip = ...
        self.A_pi = np.array([
            [np.cos(q), -np.sin(q), 0],
            [np.sin(q) * np.cos(self.alpha_p), np.cos(q) * np.cos(self.alpha_p), -np.sin(self.alpha_p)],
            [np.sin(q) * np.sin(self.alpha_p), np.cos(q) * np.sin(self.alpha_p), np.cos(self.alpha_p)]
        ])
        self.A_ip = self.A_pi.T
        ## absolute position and orientation
        # base transformation from cosy 0 to this link
        # A_i0 rotate from 0 world system to i current link, prev.A_i0 from 0 to p
        # self.A_i0 = ...
        self.A_i0 = self.A_ip @ prev.A_i0
        # position of this link
        # self.r_i__0 = ...       r_i__0 center of i wrt the world 0 in 0 (from 0 to i) , prev.r_i__0 from 0 to p
        self.r_i__0 = prev.r_i__0 + prev.A_i0.T @ self.r_pi__p
        ###Workspace End###

    def calculate_velocities(self, prev: Link, dot_q: float)-> None:
        """calculation of the velocity of this link based on the velocity of the predecessor

        Prerequisites
        -------
        - calculate_positions(args)

        Parameters
        -------
        prev: Link
            predecessor


        dot_q: float
            joint velocity omega around Zi

        Returns
        ------
        None

        """

        ###Workspace Start###
        # difference angular velocity this link minus predecessor
        # self.omega_pi__i = ...
        self.omega_pi__i = np.array([0.0, 0.0, dot_q])
        # absolute angular velocity
        # self.omega_i__i = ...
        self.omega_i__i = (self.A_ip @ prev.omega_i__i) + self.omega_pi__i
        # translational velocity
        # self.v_i__i = ...
        # Velocity of origin of i, as seen from and expressed in frame p:
        v_pi__i = prev.v_i__i + mh.tilde(prev.omega_i__i) @ self.r_pi__p
        # Transform this velocity to frame i:
        self.v_i__i = self.A_ip @ v_pi__i
        ###Workspace End###

    def calculate_accelerations(self, prev: Link, ddot_q: float) -> None:
        """calculation of the acceleration of this link based on the acceleration of the predecessor

        Prerequisites
        -------
        - calculate_positions(args)
        - calculate_velocities(args)

        Parameters
        -------
        prev: Link
            predecessor

        ddot_q: float
            joint acceleration

        Returns
        ------
        None

        """


        ###Workspace Start
        # self.alpha_i__i = ... absolute angular acceleration of i = previous +
        self.alpha_i__i = ((self.A_ip @ prev.alpha_i__i) + mh.tilde(self.A_ip @ prev.omega_i__i) @ self.omega_pi__i
                           + np.array([0.0, 0.0, ddot_q]))
        # absolute translational acceleration of i
        # self.a_i__i = ...
        self.a_i__i = self.A_ip @ (prev.a_i__i + mh.tilde(prev.alpha_i__i) @ self.r_pi__p
                                   + mh.tilde(prev.omega_i__i) @ (mh.tilde(prev.omega_i__i) @ self.r_pi__p))
        # translation acceleration of CoM
        # self.a_CoM__i = ... center of mass
        self.a_CoM__i = (self.a_i__i + mh.tilde(self.alpha_i__i) @ self.r_iCoM__i
                         + mh.tilde(self.omega_i__i) @ (mh.tilde(self.omega_i__i) @ self.r_iCoM__i))
        ###Workspace End###

    def calculate_jacobis(self, prev: Link) -> None:
        """calculation of the jacobis based on the jacobis of the predecessor

        Prerequisites
        --------
        - calculate_positions(args)

        Parameters
        -------
        prev: Link
            predecessor

        Returns
        -----
        None

        """


        ###Workspace Start###
        # rotational
        # self.J_R__i = ...
        self.J_R_rel_i = np.zeros(self.J_R__i.shape)
        if self.dof > 0:
            self.J_R_rel_i[2, self.dof - 1] = 1
        self.J_R__i = self.A_ip @ prev.J_R__i + self.J_R_rel_i
        # translational
        # self.J_Ti__i = ...
        self.J_T_rel_i = self.A_ip @ (mh.tilde(self.r_pi__p).T @ prev.J_R__i)
        self.J_Ti__i = self.A_ip @ prev.J_Ti__i + self.J_T_rel_i
        ###Workspace End###

    def calculate_mass_matrix(self)->None:
        """calculation of the contribution of this link to the total mass matrix

        Prerequisites
        --------
        - calculate_positions(args)
        - calculate_jacobis(args)

        Returns
        ------
        None

        """


        ###Workspace Start###
        J_T_i = self.J_Ti__i
        J_R_i = self.J_R__i
        Theta_i_i = self.Theta_i_i__i  # mass moment of inertia
        r_tilde = mh.tilde(self.r_iCoM__i)
        m_i = self.m

        term1 = J_T_i.T @ (m_i * J_T_i)
        term2 = m_i * (J_T_i.T @ r_tilde.T @ J_R_i)
        term3 = m_i * (J_R_i.T @ r_tilde @ J_T_i)
        term4 = J_R_i.T @ Theta_i_i @ J_R_i

        self.dM = term1 + term2 + term3 + term4
        ###Workspace End###

    def calculate_id(self, g__0: np.array((3))) -> None:
        """inverse dynamics, calculation of the joint torques

        Prerequisites
        -------
        - calculate_positions(args)
        - calculate_velocities(args)
        - calculate_accelerations(args)
        - calculate_jacobis()

        Parameters
        -------
        g__0: np.array((3))
            gravity vector in cosy 0

        Returns
        -------
        None

        """

        ###Workspace Start###
        # newton
        # F = ...
        g__i = self.A_i0 @ g__0
        F = self.m * self.a_CoM__i - self.m * g__i
        # euler
        # T = ...
        d_dt_Li = self.Theta_i_i__i @ self.alpha_i__i + mh.tilde(
            self.omega_i__i) @ self.Theta_i_i__i @ self.omega_i__i

        M_gravity = mh.tilde(self.r_iCoM__i) @ (self.m * g__i)
        T = d_dt_Li + self.m * mh.tilde(
            self.r_iCoM__i) @ self.a_i__i - M_gravity
        # projection into direction compatible with the constraints and add to tau_id
        # self.tau_id = ...
        self.tau_id = self.J_Ti__i.T @ F + self.J_R__i.T @ T
        ###Workspace End###

class Robot:
    """robot class for serial robots with revolute joints

    Attributes
    ----------
    size: int
        number of joints

    links: list[Link]
        list of links

    r_NTCP__N: np.array((3))
        vector from origin of the last cosy to the TCP represented in the last cosy

    g__0: np.array((3))
        gravity vector in cosy 0

    home_q: np.array((size))
        home position in joint space in radians, default: Nullvektor

    base: Link
        base of the robot

    r_TCP__0 : np.array((3))
        position of the TCP in cosy 0 represented in cosy 0 in meter

    v_TCP__0 : np.array((3))
        translational velocity of the TCP represented in cosy 0 in meter/seconds

    a_TCP__0 : np.array((3)): np.array((3))
        translation acceleration of the TCP reperesented in cosy 0 in meter/seconds²

    q: np.array((size))
        joint angles in radians

    dot_q: np.array((size))
        joint velocities in radians/seconds

    ddot_q: np.array((size))
        joint accelerations in radians/seconds²

    J_TTCP__0: np.array((3,size))
        translational jacobian of the TCP in cosy 0

    J_RTCP__0: np.array((3,size))
        rotational jacobian of the last link in cosy 0

    M: np.array((size,size))
        massmatrix

    tau_id: np.array((size))
        joint torques

    tau_drive: np.array((size))
        drive torques
    """
    size = 0
    links : list[Link]
    r_NTCP__N : np.array((3))
    g__0 = np.array((3))
    q_home = np.array
    q_max = np.array
    q_min = np.array
    q_comfort = np.array
    base : Link
    r_TCP__0 : np.array((3))
    v_TCP__0 : np.array((3))
    a_TCP__0 : np.array((3))
    q : np.array
    dot_q : np.array
    ddot_q : np.array
    J_RTCP__0 : np.array
    J_TTCP__0 : np.array
    M : np.array
    tau_id : np.array
    tau_antrieb : np.array

    def __init__(self, links: list[Link]) -> None:
        self.size = len(links)
        self.links = links
        self.r_NTCP__N = np.zeros((3))
        self.g__0 = np.zeros((3))
        self.base = Link(0, self.size)
        self.r_TCP__0 = np.zeros((3))
        self.v_TCP__0 = np.zeros((3))
        self.a_TCP__0 = np.zeros((3))
        self.q = np.zeros((self.size))
        self.dot_q = np.zeros((self.size))
        self.ddot_q = np.zeros((self.size))
        self.J_RTCP__0 = np.zeros((3,self.size))
        self.J_TTCP__0 = np.zeros((3,self.size))
        self.M = np.zeros((self.size,self.size))
        self.tau_id = np.zeros((self.size))
        self.tau_antrieb = np.zeros((self.size))
        self.q_home = np.zeros((self.size))
        self.q_max = np.pi * np.ones((self.size))
        self.q_min = -np.pi * np.ones((self.size))
        self.q_comfort = np.zeros((self.size))

    def calculate_positions(self, q: np.array) -> None:
        """direct kinematics (position)
        calculation of the position and orientation of all links and the TCP

        Parameters
        ----------
        q: np.array
            Gelenkwinkel in [rad]

        Returns
        -------
        None
        """

        for link in self.links:
            if link.dof == 1:
                link.calculate_positions(self.base, q[link.dof-1])
            else:
                link.calculate_positions(self.links[link.dof-2], q[link.dof-1])

        ###Workspace Start###
        # r_TCP__0 = r_i__0 + A_0i * r_NTCP__N link. is last link
        self.r_TCP__0 = link.r_i__0 + link.A_i0.T @ self.r_NTCP__N
        ###Workspace End###

        self.q = np.copy(q)

    def calculate_velocities(self, dot_q: np.array) -> None:
        """direct kinematics (velocity)
        calculation of the velocities of all links and the TCP

        Prerequisites
        ----------
        - calculate_positions(args)

        Parameters
        ----------
        dot_q: np.array
            joint velocities

        Returns
        -------
        None
        """


        ###Workspace Start###
        # calculation of the velocity of all links
        # ...
        for link in self.links:  # Using link_obj to avoid confusion if needed, but 'link' is fine too
            if link.dof == 1:
                link.calculate_velocities(self.base, dot_q[link.dof-1])
            else:
                link.calculate_velocities(self.links[link.dof-2], dot_q[link.dof-1])

        # calculation of the velocity of the TCP
        # self.v_TCP__0 = ...
        # TCP velocity in world frame: v_TCP = v_N + omega_N x r_N_TCP (all in world frame)
        self.v_TCP__0 = link.A_i0.T @ (link.v_i__i + mh.tilde(link.omega_i__i) @ self.r_NTCP__N)
        ###Workspace End###

        self.dot_q = np.copy(dot_q)

    def calculate_accelerations(self, ddot_q: np.array) -> None:
        """direct kinematics (acceleration)
        calculation of the accelerations of all links and the TCP

        Prerequisites
        ----------
        - calculate_positions(args)
        - calculate_velocities(args)

        Parameters
        ----------
        ddot_q: np.array
            Gelenkwinkelbeschleunigungen in [rad/s²]

        Returns
        -------
        None
        """


        ###Workspace Start###
        # calculation of the acceleration of all links
        # ...
        for link in self.links:
            if link.dof == 1:
                link.calculate_accelerations(self.base, ddot_q[link.dof - 1])
            else:
                link.calculate_accelerations(self.links[link.dof - 2], ddot_q[link.dof - 1])
        # calculation of the acceleration of the TCP
        # self.a_TCP__0 = ...
        # Calculate TCP acceleration in frame N
        # Formula: a_TCP__N = a_N__N + alpha_N__N x r_N_TCP__N + omega_N__N x (omega_N__N x r_N_TCP__N)
        a_PTCP_i = (link.a_i__i + mh.tilde(link.alpha_i__i) @ self.r_NTCP__N
                    + mh.tilde(link.omega_i__i) @ (mh.tilde(link.omega_i__i) @ self.r_NTCP__N))

        # Transform TCP acceleration from frame N to base frame 0
        # last_link.A_i0 is the rotation matrix from frame 0 to frame N (A_N0)
        self.a_TCP__0 = link.A_i0.T @ a_PTCP_i
        ###Workspace End###

        self.ddot_q = np.copy(ddot_q)

    def calculate_jacobis(self) -> None:
        """direct kinematics (jacobian)
        calculation of the jacobians of all links and the TCP

        Prerequisites
        ----------
        - calculate_positions(args)

        Returns
        -------
        None
        """


        ###Workspace Start###
        # calculation of the jacobians of all links
        # ...
        for link in self.links:
            if link.dof == 1:
                link.calculate_jacobis(self.base)
            else:
                link.calculate_jacobis(self.links[link.dof-2])
        # calculation of the jacobians of the TCP
        # self.J_RTCP__0 = ...
        self.J_RTCP__0 = link.A_i0.T @ link.J_R__i
        # self.J_TTCP__0 = ...
        self.J_TTCP__N = link.J_Ti__i + mh.tilde(self.r_NTCP__N).T @ link.J_R__i
        self.J_TTCP__0 = link.A_i0.T @ self.J_TTCP__N
        ###Workspace End###

    def calculate_mass_matrix(self) -> None:
        """calculation of the mass matrix

        Prerequisites
        ----------
        - calculate_positions(args)
        - calculate_jacobis()

        Returns
        -------
        None
        """


        ###workspace Start###
        # build total mass matrix M from the contribution of each link
        self.M = np.zeros((self.size, self.size))
        for link in self.links:
            link.calculate_mass_matrix()
            self.M += link.dM
        ###Workspace End###

    def calculate_id(self) -> None:
        """inverse dynamics
        calculation of the joint torques

        Prerequisites
        ----------
        - calculate_positions(args)
        - calculate_velocities(args)
        - calculate_accelerations(args)
        - calculate_jacobis()

        Returns
        -------
        None
        """

        ###Workspace Start###
        # add the contribution of each link to the total joint torques
        self.tau_id = np.zeros(self.size)

        for link in self.links:
            link.calculate_id(self.g__0)  # 传入重力矢量
            self.tau_id += link.tau_id
        ###Workspace End###

    def calculate_eom(self) -> np.array:
        """direct dynamics
        calculation of the joint accelerations from the joint torques

        Prerequisites
        ---------
        - calculate_positions(args)
        - calculate_velocities(args)
        - calculate_jacobis()

        Returns
        --------
        ddot_q: np.array((size))
            joint accelerations

        """


        ###Workspace Start###
        # calculatio h-vector
        # a...qddot
        self.ddot_q = np.zeros_like(self.ddot_q)
        self.calculate_accelerations(self.ddot_q)
        # h = ...
        self.calculate_id()
        h = np.copy(self.tau_id)
        # calculate mass matrix
        self.calculate_mass_matrix()

        # solve system of equations
        # M * ddot_q = Q_u - h
        # b = Q_u - h
        b = self.tau_antrieb - h

        # 使用 np.linalg.solve() 求解 ddot_q，这比手动求逆更高效、更稳定。
        ddot_q = np.linalg.solve(self.M, b)
        return ddot_q
        ###Workspace End###

    def calculate_tmstp_euler(self, dt: float) -> None:
        """calculation of the joint accelerations, velocities and positions
        through time integration with an explicit Euler method

        Parameters
        --------

        dt: float
            time step in seconds

        Returns
        -------
        None

        """


        ###Workspace Start###
        # calculation of the joint accelerations
        self.ddot_q = self.calculate_eom()
        # time integration for velocities and positions (explicite euler)
        self.q = self.q + self.dot_q * dt
        self.dot_q = self.dot_q + self.ddot_q * dt
        ###Workspace End###

        # direct kinematics with new positions and velocities
        self.calculate_positions(self.q)
        self.calculate_velocities(self.dot_q)
        self.calculate_jacobis()


    ###Workspace Start###
    def ode_func(self, t: float, y: np.array) -> np.array:
        """statespace formulation of the equations of motion for an ode solver
        (e.g. scipy.integrate.RK45 -> fun)
        ... """
        # 1. 从状态向量 y 中解包出 q 和 dot_q
        current_q = y[:self.size]
        current_dot_q = y[self.size:]

        # 2. 将机器人的内部状态更新为求解器传入的当前状态
        #    这是必须的，因为 calculate_eom 会使用 self.q 和 self.dot_q 等
        self.calculate_positions(current_q)
        self.calculate_velocities(current_dot_q)
        self.calculate_jacobis()

        # 3. 计算关节加速度 ddot_q
        #    calculate_eom 会使用 self.tau_antrieb，这个量在调用积分器之前已经被设置好了
        current_ddot_q = self.calculate_eom()

        # 4. 将结果打包成状态向量的导数 [dot_q, ddot_q] 并返回
        dydt = np.concatenate([current_dot_q, current_ddot_q])

        return dydt
    ###Workspace End###

    def calculate_tmstp_rk45(self, dt: float) -> None:
        """calculation of the joint accelerations, velocities and positions
        through time integration with the Runge-Kutta method (RK45)

        Parameters
        --------

        dt: float
            time step in seconds

        Returns
        -------
        None

        """


        ###Workspace Start###
        # get q, dot_q, ddot_q from time integration with RK45
        # 1. 将机器人当前状态 (q, dot_q) 打包成初始状态向量 y0
        y0 = np.concatenate([self.q, self.dot_q])

        # 2. 定义积分的时间范围（从当前时刻 0 到 dt 之后）
        t_span = [0, dt]

        # 3. 调用 scipy 的 solve_ivp 求解器
        #    它会使用我们上面定义的 self.ode_func 来计算系统动态
        sol = solve_ivp(fun=self.ode_func, t_span=t_span, y0=y0, method='RK45')

        # 4. 从求解结果 sol 中提取积分结束时（即 dt 之后）的状态
        y_new = sol.y[:, -1]
        q_new = y_new[:self.size]
        dot_q_new = y_new[self.size:]
        # direct kinematics with new positions and velocities
        # 5. 更新机器人的状态
        self.q = q_new
        self.dot_q = dot_q_new
        ###Workspace End###

        # 6. 用新的位置和速度更新机器人的运动学状态
        self.calculate_positions(self.q)
        self.calculate_velocities(self.dot_q)
        self.calculate_jacobis()
        ###Workspace End###


def robot_factory(path: str) -> Robot:
    """create a robot object based on a robot config file
    
    Supported formats
    --------
    - .yaml
    - .json

    Parameters
    ---------

    path: string
        path to the robot config file

    Returns
    --------

    new_robot: robot
        new robot object
    """

    #Lesen der Datei
    data = {}
    tmp = pathlib.Path(path).suffix
    match tmp:
        case '.json':
            with open (path,'r') as f:
                data = json.load(f)
        case '.yaml':
            with open (path,'r') as f:
                data = yaml.load(f,Loader=yaml.loader.SafeLoader)
        case _:
            print('config file type not valid')
            return None
    
    #Erstellen des Roboters
    return __parse_dict_to_robot(data)

def __eval_expression(expression: str) -> any:
    """Reliable evaluation of strings from the configuration data \\
    - Lengths are specified in meter by default
    - Angles are specified in radians by default
    - Angles can be formulated as a function of pi, e.g. alpha_p = pi/2
    - If degrees are to be used, this must be explicitly specified by \'deg\', e.g. alpha_p = 90 deg

    Parameters
    ---------
    expression: str
        expression to evaluate
    
    Returns
    --------
    evaluated: float | np.array
        resulting number
    """

    # replace \'def\' by pi/180
    expression = expression.replace('deg','* pi/180.0')
    # remove \'
    expression = expression.replace('\'','')
    # define alias for pi
    allowed_names = {'pi': np.pi}
    code = compile(expression, '<string>', 'eval')
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(f'Use of {name} not allowed')
    return np.array(eval(code, {'__builtins__': {}}, allowed_names))

def __set_attributes_to_object(object: any, obj_dict: dict, optional: list, required: list, \
                             disable_warnings: bool = True):
    """Assigns a value from a dictionary to the attribute of a class instance
    
    Parameters
    --------

    object: any
        object that gets the attributes
    
    obj_dict: dict
        dictionary with the attributes

    optional: list[str]
        list with the names of the optional attributes, if these are missing in the dictionary, a warning is issued

    required: list[str]
        list with the names of the required attributes; if these are missing in the dictionary, an error is displayed
    
    disable_warnings: bool
        warnings are deactivated
    """

    for param in required:
        expr = obj_dict.get(param)
        if expr != None:
            object.__setattr__(param,__eval_expression(str(expr)))
        else:
            raise ValueError(f'{param} is missing, but is a required parameter for a link')
    for param in optional:
        expr = obj_dict.get(param)
        if expr != None:
            object.__setattr__(param,__eval_expression(str(expr)))
        elif not disable_warnings:
            print(f'{param} is missing, using default value:\n {object.__getattribute__(param)} \n')

def __parse_dict_to_robot(rob_dict: dict) -> Robot:

    # create empty list for links
    link_dict = rob_dict.get('links')
    if link_dict == None:
        raise ValueError('robot without a list of links is not valid')
    links = list()

    # iteration over the links
    size = len(link_dict)
    for i in range(size):
        links.append(Link(i+1, size))
        # set necessary and optional parameters for a link
        required = ['alpha_p', 'a_p', 'd_i'] # necessary -> error if not in config file
        optional = ['m', 'Theta_i_i__i', 'r_iCoM__i'] # optional -> warning if not in confi file
        __set_attributes_to_object(links[i], link_dict[i], optional, required)
    
    # create robot object
    new_robot = Robot(links)
    # set necessary and optional parameters for a robot
    required = ['r_NTCP__N']
    optional = ['g__0']
    __set_attributes_to_object(new_robot, rob_dict, optional, required)
    
    # read angle configurations
    angle_configs = ['q_home', 'q_max', 'q_min', 'q_comfort']
    for name in angle_configs:
        angles = rob_dict.get(name)
        if angles != None:
            angles = __eval_expression(str(rob_dict[name]))
            if np.size(angles) < size:
                diff = size-np.size(angles)
                angles = np.concatenate((angles,np.zeros(diff)))
            else:
                angles = np.array(angles[0:size])
            new_robot.__setattr__(name, angles)
        else:
            print(f'{name} is missing, using default value:\n {new_robot.__getattribute__(name)}\n')

    return new_robot 
