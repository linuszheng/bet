from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env import utils
from typing import Dict, Text
from gym.utils import seeding
from .settings import pv_stddev, _n_timesteps, motor_model
import gym
from scipy.stats import norm
import numpy as np


def get_err(a, b, stdev):
  return norm(a, stdev).logpdf(b)

class HighwayEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 8,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": _n_timesteps,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "simulation_frequency": 24,
            "policy_frequency": 8,
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()


    def _create_road(self) -> None:
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=1)

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _info(self, obs, action=None):
      return {}
      

    def _reward(self, action) -> float:
        return 0

    def _is_terminated(self) -> bool:
        return False

    def _is_truncated(self) -> bool:
        return self.time*self.config["policy_frequency"] >= self.config["duration"]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.last_action = [0., 0.]
        return super().reset()[0]
    
    def step(self, action):
        other_options = self.observation_type.observe()[-8:]
        opt1 = get_err(action, other_options[0:2], pv_stddev).sum()
        opt2 = get_err(action, other_options[2:4], pv_stddev).sum()
        opt3 = get_err(action, other_options[4:6], pv_stddev).sum()
        opt4 = get_err(action, other_options[6:8], pv_stddev).sum()
        opt_errors = np.array([opt1, opt2, opt3, opt4])
        action_i = np.argmax(opt_errors)
        action = other_options[(2*action_i) : (2*action_i+2)]
        res = super().step(action)
        self.last_action = action
        return (res[0], res[1], res[2] or res[3], res[4])

    def render(self, mode="human"):
        if not hasattr(self, "n_success"):
            self.n_success = 0.
        if self.time*self.config["policy_frequency"] >= self.config["duration"]-1:
            print(f"{not self.road.vehicles[0].crashed} {self.road.vehicles[0].on_road} {self.observation_type.get_lane()}")
            if (not self.road.vehicles[0].crashed) and (self.road.vehicles[0].on_road):
                self.n_success += 1.
            print(f"{self.n_success / 100.}")




gym.register(
    id="highway-v0",
    entry_point=HighwayEnv
)

