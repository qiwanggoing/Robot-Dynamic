import numpy as np
import os
import sys
import unittest
sys.path.append(r"..")
import rdplib.robot as rb
# Python 3.10 is the minimum version required for this project


import direct_kinematics as dk




assert sys.version_info >= (3, 10)


class TestDK(unittest.TestCase):
    def test_velocity_calculation(self):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, 'robot_config_files', 'ABB_GoFa.yaml')
        robot = rb.robot_factory(config_path)

        ###Workspace Start###

        print('First test begins.')

        ###Workspace Start###
        num_approx_v_for = []
        num_approx_a_for = []
        v_list = []
        a_list = []

        for i in range(dk.robot.size):
            start = i * dk.N_joint
            end = start + dk.N_joint
            # Effective range for derivative calculation (excluding the first and last two points of each segment)
            s = start + 2
            e = end - 2

            num_approx_v_for.append((dk.r_TCP_traj[:, s - 2:e - 2] - 8 * dk.r_TCP_traj[:,
                                                                         s - 1:e - 1] + 8 * dk.r_TCP_traj[:,
                                                                                            s + 1:e + 1] - dk.r_TCP_traj[
                                                                                                           :,
                                                                                                           s + 2:e + 2]) / (
                                            12 * dk.dt))
            num_approx_a_for.append((dk.v_TCP_traj[:, s - 2:e - 2] - 8 * dk.v_TCP_traj[:,
                                                                         s - 1:e - 1] + 8 * dk.v_TCP_traj[:,
                                                                                            s + 1:e + 1] - dk.v_TCP_traj[
                                                                                                           :,
                                                                                                           s + 2:e + 2]) / (
                                            12 * dk.dt))
            v_list.append(dk.v_TCP_traj[:, s:e])
            a_list.append(dk.a_TCP_traj[:, s:e])
        num_approx_v = np.hstack(num_approx_v_for)
        num_approx_a = np.hstack(num_approx_a_for)
        v_truncated = np.hstack(v_list)
        a_truncated = np.hstack(a_list)
        np.testing.assert_allclose(num_approx_v, v_truncated, err_msg='Velocity test failed!', rtol=1e-2, atol=1e-4)
        np.testing.assert_allclose(num_approx_a, a_truncated, err_msg='Acceleration test failed!', rtol=1e-2, atol=1e-4)
        ###Workspace End###

        print('First test ends.')
        print("Velocity and acceleration have passed the tests!")  # 打印成功信息























        # 从 direct_kinematics.py 中借鉴参数设置
        q_start = np.copy(robot.q_home)  #
        q_end = q_start + np.pi / 4 * np.ones((robot.size))  #

        dt = 0.001  # time step
        T_ges = 4.0  # total time of the movement
        T_joint = T_ges / robot.size  # approximate time for the movement of one joint
        N_joint = int(T_joint / dt)  # steps for the movement of one joint
        deltaT = (N_joint - 1) * dt  # real time for the movement of one joint

        # 选择一个测试点：例如，第一个关节 (i=0) 运动到一半的时间 (k=N_joint//2)
        test_joint_idx = 1  # 测试第一个关节
        test_time_step_k = N_joint // 2 # 测试该关节运动轨迹的一半处

        # 初始化 dot_q 数组
        dot_q_snapshot = np.zeros(robot.size)

        current_q_for_setup = np.copy(q_start)
        if test_joint_idx > 0:
            current_q_for_setup[:test_joint_idx] = q_end[:test_joint_idx]

        q_snapshot = np.copy(current_q_for_setup)

        # 计算当前被驱动关节的 a 和 b 系数
        a_coeff = 2 / pow(deltaT, 3) * (q_start[test_joint_idx] - q_end[test_joint_idx])  #
        b_coeff = 3 / pow(deltaT, 2) * (q_end[test_joint_idx] - q_start[test_joint_idx])  #

        current_t = test_time_step_k * dt

        q_snapshot[test_joint_idx] = a_coeff * pow(current_t, 3) + b_coeff * pow(current_t, 2) + q_start[
            test_joint_idx]  #
        dot_q_snapshot[test_joint_idx] = 3 * a_coeff * pow(current_t, 2) + 2 * b_coeff * current_t  #
        # ddot_q_snapshot[test_joint_idx] = 6*a_coeff*current_t + 2*b_coeff #

        # 1. 计算位置 (必须在雅可比和速度计算之前)
        robot.calculate_positions(q_snapshot)

        # 2. 计算雅可比矩阵 (假设这个函数是正确的)
        robot.calculate_jacobis()
        J_TTCP__0_snapshot = robot.J_TTCP__0  # 获取计算得到的平移雅可比矩阵

        # 3. 通过雅可比矩阵和关节速度计算期望的TCP速度
        # v_TCP = J_TTCP * dot_q
        expected_v_TCP__0 = J_TTCP__0_snapshot @ dot_q_snapshot

        # 4. 计算通过速度递归得到的TCP速度
        robot.calculate_velocities(dot_q_snapshot)
        actual_v_TCP__0 = robot.v_TCP__0

        # 5. 比较两个结果
        print(f"\nTest State for joint {test_joint_idx} at time step k={test_time_step_k}:")
        print(f"q_snapshot: {q_snapshot}")
        print(f"dot_q_snapshot: {dot_q_snapshot}")
        print(f"J_TTCP__0 @ dot_q_snapshot (expected_v_TCP__0): {expected_v_TCP__0}")
        print(f"robot.v_TCP__0 (actual_v_TCP__0): {actual_v_TCP__0}")
        print(f"Difference: {actual_v_TCP__0 - expected_v_TCP__0}")

        np.testing.assert_allclose(actual_v_TCP__0, expected_v_TCP__0,
                                   err_msg='Velocity calculation (v_TCP__0) does not match Jacobian product (J*dq)!',
                                   rtol=1e-5,  # 相对容差
                                   atol=1e-7)  # 绝对容差

        print(
            f"\nUnit test 'test_velocity_calculation' in TestDK passed for {2 if robot.size > 0 else 1} checked states.")  # 打印成功信息
        ###Workspace End###



