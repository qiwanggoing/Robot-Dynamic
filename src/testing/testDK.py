import numpy as np
import os
import sys
import unittest

import rdplib.robot as rb

# Python 3.10 is the minimum version required for this project
assert sys.version_info >= (3, 10)


class TestDK(unittest.TestCase):
    def test_velocity_calculation(self):
        robot = rb.robot_factory(os.path.join('robot_config_files','ABB_GoFa.yaml'))

        ###Workspace Start###
        ### helpful functions could be the following
        ### robot.calculate_positions(???)
        ### robot.calculate_velocities(???)
        ### np.testing.assert_allclose(np.array([1, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 0, 0]), err_msg='Velocity test failed!', rtol=1e-2, atol=1e-4)
        ###Workspace End###
