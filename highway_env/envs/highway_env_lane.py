from typing import Dict, Text

import numpy as np
import random
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.regulation import RegulatedRoad
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import LineType,StraightLane
Observation = np.ndarray


class HighwayEnvLane(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 5,
                "vehicles_count": 100,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -0.9,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.2,  # The reward received when driving on the right-most lanes, linearly mapped to
                "middle_lane_reward" : 0.85, # Reward for maintaining speed in middle lane zero for other lanes.
                "high_speed_reward": 0.25,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0.5,  # The reward received at each lane change action.
                "reward_speed_range": [20, 65],
                "normalize_reward": True,
                "offroad_terminal": False,
                "speed_limits" : [55,35,30,30,20]
            }
        )
        return config

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=1000) -> None:
        """Create a road composed of straight adjacent lanes."""
        net = RoadNetwork()
        nodes_str = ("a", "b")
        default_lane_width = 4
        start = 0
        angle = 0
        for lane in range(self.config["lanes_count"]):
            origin = np.array([start, lane * default_lane_width])
            end = np.array([start + length, lane * default_lane_width])
            rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            origin = rotation @ origin
            end = rotation @ end
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == self.config["lanes_count"] - 1 else LineType.NONE]
            net.add_lane(*nodes_str, StraightLane(origin, end, line_types=line_types, speed_limit=self.config["speed_limits"][lane]))
        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road
        

    def _make_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []

        ego_vehicle = Vehicle.create_random(self.road,speed=self.config["speed_limits"][3],lane_id=3,spacing=self.config["ego_spacing"],)
        ego_vehicle = self.action_type.vehicle_class(self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed)
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        for others in other_per_controlled:
            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
    
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["middle_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        current_lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        # Define desired speed for certain lane (e.g., middle lane)
        lane_currently = self.vehicle.lane_index[2]
        desired_speed = self.config["speed_limits"][lane_currently]
        # Reward for maintaining desired speed
        speed_reward = 1 - abs(forward_speed - desired_speed) / desired_speed
        # Reward for staying in the middle lane
        # print(current_lane)
        if(current_lane ==2 or current_lane ==1):
            middle_lane_reward = self.config ["middle_lane_reward"]
            lane_change_reward = -0.1
        elif(current_lane == 0 or current_lane == 4):
            middle_lane_reward = -0.30
            lane_change_reward = self.config["lane_change_reward"]
        else:
            middle_lane_reward = -0.1
            lane_change_reward = 0.1
        # middle_lane_reward = abs(current_lane - 2) * 0.1
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": current_lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "speed_reward": speed_reward,
            "middle_lane_reward": middle_lane_reward,
            "lane_change_reward": lane_change_reward
        }
    
    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


