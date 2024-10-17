"""
Gripper for Franka's Panda (has two fingers).
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class PandaGripperBase(GripperModel):
    """
    Gripper for Franka's Panda (has two fingers).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/panda_gripper_with_umi.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_collision", "finger1_pad_collision"],
            "right_finger": ["finger2_collision", "finger2_pad_collision"],
            "left_fingerpad": ["finger1_pad_collision"],
            "right_fingerpad": ["finger2_pad_collision"],
        }
    @property
    def _important_sites(self):
        """
        Sites used to aid visualization by human. (usually "grip_site" and "grip_cylinder")
        (and should be hidden from robots)

        Returns:
            dict:

                :`'grip_site'`: Name of grip actuation intersection location site
                :`'grip_cylinder'`: Name of grip actuation z-axis location site
                :`'ee'`: Name of end effector site
                :`'ee_x'`: Name of end effector site (x-axis)
                :`'ee_y'`: Name of end effector site (y-axis)
                :`'ee_z'`: Name of end effector site (z-axis)
        """
        return {
            "grip_site": "grip_site",
            "eef_site": "end_effector",
            "grip_cylinder": "grip_site_cylinder",
            "ee": "ee",
            "ee_x": "ee_x",
            "ee_y": "ee_y",
            "ee_z": "ee_z",
        }

class PandaGripper(PandaGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
