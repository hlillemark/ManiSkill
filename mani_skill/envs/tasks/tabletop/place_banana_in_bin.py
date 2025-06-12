from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

@register_env("PlaceBananaInBin-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PlaceBananaInBinEnv(BaseEnv):
    """A task where the robot must pick up a banana and place it into a wooden bin.
    
    The task involves:
    1. Grasping a banana from a random position on the table
    2. Moving it to a wooden bin
    3. Releasing it inside the bin
    
    Success requires the banana to be inside the bin and the robot to be static.
    """
    
    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]
    
    # Bin specifications 
    bin_wall_halfsize = [0.12, 0.008, 0.05]  # Length, thickness, height
    bin_bottom_halfsize = [0.12, 0.12, 0.008]  # Length, width, thickness

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Load robot-specific configurations
        cfg = PICK_CUBE_CONFIGS.get(robot_uids, PICK_CUBE_CONFIGS["panda"])
        for key, value in cfg.items():
            setattr(self, key, value)
            
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # Build table scene
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create wooden bin
        # If needed, download with wget https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/cse276f/rosewood_veneer1_diff_1k.png
        wood_material = sapien.render.RenderMaterial()
        wood_material.base_color_texture = sapien.render.RenderTexture2D("rosewood_veneer1_diff_1k.png")

        # Precalculate positions for bin components
        bottom_pos = [0, 0, self.bin_bottom_halfsize[2]]
        wall_height_center = self.bin_wall_halfsize[2] + self.bin_bottom_halfsize[2]

        # Wall positions
        wall_1_pos = [0, -self.bin_bottom_halfsize[1] + self.bin_wall_halfsize[1], wall_height_center]
        wall_2_pos = [0, self.bin_bottom_halfsize[1] - self.bin_wall_halfsize[1], wall_height_center]
        wall_3_pos = [-self.bin_bottom_halfsize[0] + self.bin_wall_halfsize[1], 0, wall_height_center]
        wall_4_pos = [self.bin_bottom_halfsize[0] - self.bin_wall_halfsize[1], 0, wall_height_center]

        # Side wall dimensions
        side_wall_halfsize = [self.bin_wall_halfsize[1], self.bin_bottom_halfsize[1], self.bin_wall_halfsize[2]]

        # Create bin
        builder = self.scene.create_actor_builder()

        # Add bottom
        builder.add_box_visual(
            pose=sapien.Pose(p=bottom_pos),
            half_size=self.bin_bottom_halfsize,
            material=wood_material
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=bottom_pos),
            half_size=self.bin_bottom_halfsize
        )
        
        # Add walls
        for pos, half_size in [
            (wall_1_pos, self.bin_wall_halfsize),
            (wall_2_pos, self.bin_wall_halfsize),
            (wall_3_pos, side_wall_halfsize),
            (wall_4_pos, side_wall_halfsize)
        ]:
            builder.add_box_visual(
                pose=sapien.Pose(p=pos),
                half_size=half_size,
                material=wood_material
            )
            builder.add_box_collision(
                pose=sapien.Pose(p=pos),
                half_size=half_size
            )

        builder.initial_pose = sapien.Pose(p=[0.1, 0.0, 0.02])
        self.bin = builder.build(name="bin")

        # Create banana
        banana_builder = actors.get_actor_builder(
            self.scene,
            id="ycb:011_banana",
        )
        banana_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.banana = banana_builder.build(name="banana")

    def _after_reconfigure(self, options: dict):
        # Get banana height for proper positioning
        collision_mesh = self.banana.get_first_collision_mesh()
        self.banana_z = -collision_mesh.bounding_box.bounds[0, 2]
        self.banana_z = common.to_tensor(self.banana_z, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Initialize positions
            bin_xyz = torch.zeros((b, 3))
            banana_xyz = torch.zeros((b, 3))
            
            # Fixed x positions
            bin_xyz[:, 0] = torch.rand((b)) * 0.1 + 0.05  # x in [0.05, 0.15]
            banana_xyz[:, 0] = torch.rand((b)) * 0.1 - 0.2  # x in [-0.2, -0.1]
            
            # Binary choice for y positioning
            choice = torch.randint(0, 2, (b,))
            
            # Position objects on opposite sides
            bin_xyz[:, 1] = torch.where(choice == 0, 
                                       torch.rand((b)) * 0.1 + 0.05,  # positive y [0.05, 0.15]
                                       torch.rand((b)) * 0.1 - 0.15)  # negative y [-0.15, -0.05]
            
            banana_xyz[:, 1] = torch.where(choice == 0,
                                          torch.rand((b)) * 0.1 - 0.15,  # negative y [-0.15, -0.05] 
                                          torch.rand((b)) * 0.1 + 0.05)  # positive y [0.05, 0.15]
            
            # Set z positions
            bin_xyz[:, 2] = self.bin_bottom_halfsize[2]
            banana_xyz[:, 2] = self.banana_z
            
            # Set poses with random z-axis rotation
            bin_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            banana_qs = random_quaternions(b, lock_x=True, lock_y=True)
            
            self.bin.set_pose(Pose.create_from_pq(bin_xyz, bin_qs))
            self.banana.set_pose(Pose.create_from_pq(banana_xyz, banana_qs))

    def _is_banana_in_bin(self):
        """Check if banana is inside the bin boundaries and not being grasped"""
        banana_pos = self.banana.pose.p
        bin_pos = self.bin.pose.p
        
        # Calculate bin inner boundaries
        bin_inner_x_half = self.bin_bottom_halfsize[0] - self.bin_wall_halfsize[1]
        bin_inner_y_half = self.bin_bottom_halfsize[1] - self.bin_wall_halfsize[1]
        
        # Check spatial constraints
        x_in_bin = torch.abs(banana_pos[:, 0] - bin_pos[:, 0]) <= bin_inner_x_half
        y_in_bin = torch.abs(banana_pos[:, 1] - bin_pos[:, 1]) <= bin_inner_y_half
        
        # Check height constraint
        bin_bottom_top = bin_pos[:, 2] + self.bin_bottom_halfsize[2]
        z_in_bin = (banana_pos[:, 2] >= bin_bottom_top - 0.01) & (banana_pos[:, 2] <= bin_bottom_top + 0.1)
        
        # Check if not grasped
        not_grasped = ~self.agent.is_grasping(self.banana)
        
        return x_in_bin & y_in_bin & z_in_bin & not_grasped

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            bin_pos=self.bin.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                banana_pose=self.banana.pose.raw_pose,
                tcp_to_banana_pos=self.banana.pose.p - self.agent.tcp_pose.p,
                banana_to_bin_pos=self.bin.pose.p - self.banana.pose.p,
            )
        return obs

    def evaluate(self):
        is_banana_placed = self._is_banana_in_bin()
        is_grasped = self.agent.is_grasping(self.banana)
        is_robot_static = self.agent.is_static(0.2)
        
        return {
            "success": is_banana_placed & is_robot_static,
            "is_banana_placed": is_banana_placed,
            "is_obj_placed": is_banana_placed,  # For compatibility
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Reaching reward
        tcp_to_banana_dist = torch.linalg.norm(
            self.banana.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_banana_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]

        # Calculate target positions
        banana_pos = self.banana.pose.p
        bin_pos = self.bin.pose.p
        
        target_height_above_bin = 0.15
        target_above_bin = bin_pos.clone()
        target_above_bin[:, 2] = bin_pos[:, 2] + target_height_above_bin
        
        target_in_bin = bin_pos.clone()
        target_in_bin[:, 2] = bin_pos[:, 2] + self.bin_bottom_halfsize[2] + 0.02
        
        # Calculate distances
        dist_to_above = torch.linalg.norm(target_above_bin - banana_pos, axis=1)
        dist_to_center = torch.linalg.norm(target_in_bin - banana_pos, axis=1)
        
        # Drop zone check
        drop_zone_radius = 0.08
        in_drop_zone = dist_to_above <= drop_zone_radius
        
        # Grasping reward (disabled in drop zone)
        reward += is_grasped * (~in_drop_zone).float()
        
        # Placement reward
        place_reward = 1 - torch.tanh(5 * dist_to_above)
        
        # Completion bonus
        in_bin_bonus = self._is_banana_in_bin().float() * 3.0
        
        # Placement reward logic
        placement_reward_condition = torch.where(in_drop_zone, 
                                                torch.ones_like(is_grasped),
                                                is_grasped)
        
        reward += place_reward * placement_reward_condition
        reward += in_bin_bonus
        
        # Release bonus
        release_bonus = torch.where(in_drop_zone & ~is_grasped, 
                                   0.5,
                                   torch.zeros_like(is_grasped.float()))
        reward += release_bonus

        # Static reward
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        # Success bonus
        reward[info["success"]] = 8
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

