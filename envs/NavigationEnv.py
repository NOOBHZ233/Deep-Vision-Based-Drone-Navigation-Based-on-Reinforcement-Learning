import numpy as np

from gym_pybullet_drones.envs.BaseRLEnv import BaseRLEnv
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class NavigationEnv(BaseRLEnv):
    """Single agent RL problem: hover and navigate to target using KIN+Depth observations."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 24,
                 gui=True,
                 record=False,
                 obs: ObservationType = ObservationType.KIND,
                 act: ActionType = ActionType.RPM
                 ):
        """Initialization of a single agent RL environment."""
        self.TARGET_POS = np.array([0, 0, 1])  
        self.EPISODE_LEN_SEC = 8               
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

        self.last_depth = [0.0 for _ in range(self.NUM_DRONES)]

    ################################################################################
    
    def _computeReward(self):
        """Computes reward based on distance to target."""
        state = self._getDroneStateVector(0)
        pos_error = np.linalg.norm(self.TARGET_POS - state[0:3])
        reward = max(0, 2 - pos_error**4)
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        """Episode ends if the drone reaches the target."""
        state = self._getDroneStateVector(0)
        pos_error = np.linalg.norm(self.TARGET_POS - state[0:3])
        return pos_error < 0.01 

    ################################################################################
    
    def _computeTruncated(self):
        """Episode is truncated if drone exceeds safe boundaries or timeout."""
        state = self._getDroneStateVector(0)
        out_of_bounds = (abs(state[0]) > 1.5 or
                         abs(state[1]) > 1.5 or
                         state[2] > 2.0)
        too_tilted = abs(state[7]) > 0.4 or abs(state[8]) > 0.4  # roll/pitch
        timeout = self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC
        return out_of_bounds or too_tilted or timeout

    ################################################################################
    
    def _computeInfo(self):
        """Returns info dictionary, currently dummy."""
        return {"distance_to_target": np.linalg.norm(self.TARGET_POS - self._getDroneStateVector(0)[0:3])}

    ################################################################################

    def _computeObs(self):
        """
        Overrides BaseRLEnv._computeObs() to ensure stable depth handling
        for ObservationType.KIND.
        """
        if self.OBS_TYPE == ObservationType.KIND:
            obs_13 = np.zeros((self.NUM_DRONES, 13))
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                    rgb, dep, seg = self._getDroneImages(i, segmentation=False)
                    if dep is not None and dep.size > 0:
                        depth_value = np.mean(dep)  
                    else:
                        depth_value = self.last_depth[i]
                else:
                    depth_value = self.last_depth[i]

                self.last_depth[i] = depth_value
                obs_13[i, :] = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16], depth_value])

            ret = obs_13.astype(np.float32)
            for buf in self.action_buffer:
                ret = np.hstack([ret, buf])
            return ret

        else:
            return super()._computeObs()
