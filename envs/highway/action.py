from highway_env.envs.common.action import ActionType
from gym import spaces
from .settings import motor_model, numHA, pv_stddev
from highway_env.vehicle.kinematics import Vehicle
import numpy as np
from .observation_helper import process_obs

class MyAction(ActionType):


    def space(self):
        return spaces.Box(shape=(2,), low=np.array([-40, -40]), high=np.array([40, 40]), dtype=np.float32)

    @property
    def vehicle_class(self):
        return Vehicle

    def act(self, action: int) -> None:
        la = action

        la[0] += np.random.normal(0, pv_stddev[0])
        la[1] += np.random.normal(0, pv_stddev[1])
        # print(la)
        self.controlled_vehicle.act({
                "steering": la[0],
                "acceleration": la[1]
            })




def action_factory(env: 'AbstractEnv', config: dict) -> ActionType:
  assert(config["type"]=="myact")
  return MyAction(env, **config)

