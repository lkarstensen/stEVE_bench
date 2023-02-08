from enum import Enum
from typing import Any, Dict
import logging
import random
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding


from .arch_vmr94_util.aorticarch import AorticArch
from .arch_vmr94_util.simulation import Simulation
from .arch_vmr94_util.pathfinder import Pathfinder
from .arch_vmr94_util.imaging import Imaging


class ObservationType(Enum):
    IMAGE = 0
    TRACKING = 1


class ArchVMR94(gym.Env):
    def __init__(
        self,
        normalize_obs: bool = False,
        init_visual: bool = False,
        target_reached_threshold: float = 5,
        step_limit=150,
        normalize_action: bool = False,
        obs_type: ObservationType = ObservationType.TRACKING,
    ) -> None:
        self.logger = logging.getLogger(self.__module__)

        self.normalize_obs = normalize_obs
        self.target_reached_threshold = target_reached_threshold
        self.step_limit = step_limit
        self.obs_type = obs_type

        self.vesseltree = AorticArch()
        self.intervention = Simulation(
            self.vesseltree,
            init_visual=init_visual,
            normalize_action=normalize_action,
            target_size=target_reached_threshold,
        )
        self._init_potential_targets()
        self.pathfinder = Pathfinder(self.vesseltree, self.intervention)
        self.imaging = Imaging(self.intervention, (800, 2000))
        self._dist_between_tracking_obs = 1
        self.n_tracking_points = 2

        self.target = None
        self._np_random, _ = seeding.np_random(random.randint(0, 99999999999))
        self._tracking_memory = np.empty(())
        self._last_path_length = 0.0
        self._step_counter = 0
        self._target_reached = False
        if obs_type == ObservationType.TRACKING:
            self.observation_space = self._get_tracking_observation_space(normalize_obs)
        else:
            self.observation_space = self._get_image_observation_space(normalize_obs)
        self.action_space = self.intervention.action_space

    def step(
        self, action: np.ndarray
    ) -> tuple[Dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.array(action)
        self.intervention.step(action)
        target = self.target
        position = self.intervention.tracking[0]
        dist = np.linalg.norm(target - position)
        self._target_reached = dist < self.target_reached_threshold
        if self.obs_type == ObservationType.TRACKING:
            self._tracking_memory[1] = self._tracking_memory[0]
            self._tracking_memory[0] = self.intervention.tracking.astype(np.float32)
            obs = self._get_tracking_observation()
        else:
            obs = self._get_image_observation()
        reward = self._get_reward()
        terminal = self._get_terminal()
        truncation = self._get_truncation()
        self._step_counter += 1
        return obs, reward, terminal, truncation, {"success": self._target_reached}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Dict[str, np.ndarray]:
        super().reset(seed=seed)
        self.target = self._np_random.choice(self.potential_targets)
        self.intervention.reset(self.target)
        if self.obs_type == ObservationType.TRACKING:
            self._tracking_memory = np.array(
                [self.intervention.tracking, self.intervention.tracking],
                dtype=np.float32,
            )
            obs = self._get_tracking_observation()
        else:
            obs = self._get_image_observation()

        self._last_path_length = self.pathfinder.get_path_length(
            self.intervention.tracking[0], self.target
        )
        self._step_counter = 0
        self._target_reached = False

        return obs

    def close(self):
        self.intervention.close()

    def render(self) -> np.ndarray:
        return self.intervention.render()

    def _init_potential_targets(self):
        self.potential_targets = np.empty((0, 3))
        for branch in ["btrunk", "carotid", "rt_carotid", "subclavian"]:
            points = self.vesseltree[branch].coordinates
            dist_to_aorta = self.vesseltree["aorta"].dist_to_branch(points)
            to_omit = np.argwhere(dist_to_aorta < 17)
            points = np.delete(points, to_omit, axis=0)
            self.potential_targets = np.vstack((self.potential_targets, points))

    def _get_image_observation(self) -> Dict[str, np.ndarray]:
        image = self.imaging.get_image()
        image = np.asarray(image, dtype=np.uint8)
        insertion_point = np.delete(self.vesseltree.insertion.position, 1, axis=-1)
        target = np.delete(self.target, 1, axis=-1) - insertion_point
        if self.normalize_obs:
            image = self._normalize(image, self.imaging.low, self.imaging.high)
            target = self._normalize(
                target,
                np.min(self.potential_targets, axis=0),
                np.max(self.potential_targets, axis=0),
            )
        return {"image": image, "target": target}

    def _get_tracking_observation(self) -> Dict[str, np.ndarray]:
        insertion_point = np.delete(self.vesseltree.insertion.position, 1, axis=-1)
        memory = np.delete(self._tracking_memory, 1, axis=-1) - insertion_point

        tracking_state_0 = self._get_tracking_obs_one_timestep(memory[0])
        tracking_state_1 = self._get_tracking_obs_one_timestep(memory[1])

        tracking = np.array(
            [tracking_state_0, tracking_state_1],
            dtype=np.float32,
        )
        target = np.delete(self.target, 1, axis=-1) - insertion_point
        last_action = self.intervention.last_action
        if self.normalize_obs:
            tracking = self._normalize(
                tracking,
                self._tracking_obs_low,
                self._tracking_obs_high,
            )
            target = self._normalize(
                target,
                np.min(self.potential_targets, axis=0),
                np.max(self.potential_targets, axis=0),
            )
            last_action = self._normalize(
                last_action,
                self.intervention.action_space.low,
                self.intervention.action_space.high,
            )
        return {"tracking": tracking, "target": target, "action": last_action}

    def _get_tracking_obs_one_timestep(self, tracking):
        tracking_state = [tracking[0]]
        acc_dist = 0.0
        for point, next_point in zip(tracking[:-1], tracking[1:]):
            if len(tracking_state) >= self.n_tracking_points or np.all(
                point == next_point
            ):
                break
            length = np.linalg.norm(next_point - point)
            dist_to_point = self._dist_between_tracking_obs - acc_dist
            acc_dist += length
            while (
                acc_dist >= self._dist_between_tracking_obs
                and len(tracking_state) < self.n_tracking_points
            ):
                unit_vector = (next_point - point) / length
                tracking_point = point + unit_vector * dist_to_point
                tracking_state.append(tracking_point)
                acc_dist -= self._dist_between_tracking_obs
        while len(tracking_state) < self.n_tracking_points:
            tracking_state.append(tracking_state[-1])
        tracking_state = np.array(tracking_state, dtype=np.float32)
        tracking_state[1:] = tracking_state[1:] - tracking_state[:-1]
        return tracking_state

    def _get_reward(self) -> float:
        position = self.intervention.tracking[0]
        path_length = self.pathfinder.get_path_length(position, self.target)
        path_length_delta = self._last_path_length - path_length
        self._last_path_length = path_length
        reward = -0.005 + float(self._target_reached) - 0.001 * path_length_delta
        return reward

    def _get_terminal(self) -> bool:
        return self._target_reached

    def _get_truncation(self) -> bool:
        return self._step_counter >= self.step_limit

    @staticmethod
    def _normalize(values: np.ndarray, low: np.ndarray, high: np.ndarray):
        return np.array(2 * ((values - low) / (high - low)) - 1, dtype=np.float32)

    def _get_tracking_observation_space(self, normalize_obs):
        insertion_point = self.vesseltree.insertion.position
        tracking_low = self.vesseltree.coordinate_space.low - insertion_point
        tracking_high = self.vesseltree.coordinate_space.high - insertion_point
        tracking_obs_low = np.zeros((2, 2, 2))
        tracking_obs_high = np.zeros((2, 2, 2))
        tracking_obs_low[:, 0] = np.delete(tracking_low, 1, axis=-1)
        tracking_obs_high[:, 0] = np.delete(tracking_high, 1, axis=-1)
        tracking_obs_low[:, 1] = -np.ones((2,)) * self._dist_between_tracking_obs
        tracking_obs_high[:, 1] = np.ones((2,)) * self._dist_between_tracking_obs
        self._tracking_obs_low = tracking_obs_low
        self._tracking_obs_high = tracking_obs_high
        if normalize_obs:
            tracking_obs_low = self._normalize(
                tracking_obs_low, tracking_obs_low, tracking_obs_high
            )
            tracking_obs_high = self._normalize(
                tracking_obs_high, tracking_obs_low, tracking_obs_high
            )
        tracking_obs_space = gym.spaces.Box(
            low=tracking_obs_low.astype(np.float32),
            high=tracking_obs_high.astype(np.float32),
        )
        target_low = np.min(self.potential_targets, axis=0)
        target_high = np.max(self.potential_targets, axis=0)
        if normalize_obs:
            target_low = self._normalize(target_low, target_low, target_high)
            target_high = self._normalize(target_high, target_low, target_high)
        target_obs_space = gym.spaces.Box(
            low=target_low.astype(np.float32), high=target_high.astype(np.float32)
        )
        action_obs_low = self.intervention.action_space.low
        action_obs_high = self.intervention.action_space.high
        if normalize_obs:
            action_obs_low = self._normalize(
                action_obs_low, action_obs_low, action_obs_high
            )
            action_obs_high = self._normalize(
                action_obs_high, action_obs_low, action_obs_high
            )
        action_obs_space = gym.spaces.Box(
            low=action_obs_low.astype(np.float32),
            high=action_obs_high.astype(np.float32),
        )
        observation_space = gym.spaces.Dict(
            {
                "tracking": tracking_obs_space,
                "target": target_obs_space,
                "action": action_obs_space,
            }
        )

        return observation_space

    def _get_image_observation_space(self, normalize_obs: bool):
        image_shape = self.imaging.image_size
        image_low = np.ones(image_shape, dtype=np.int8) * self.imaging.low
        image_high = np.ones(image_shape, dtype=np.int8) * self.imaging.high
        if normalize_obs:
            low, high = image_low, image_high
            image_low = self._normalize(image_low, low, high)
            image_high = self._normalize(image_high, low, high)
            image_obs_space = gym.spaces.Box(image_low, image_high)
        else:
            image_obs_space = gym.spaces.Box(image_low, image_high, dtype=np.int8)

        target_low = np.min(self.potential_targets, axis=0)
        target_high = np.max(self.potential_targets, axis=0)
        if normalize_obs:
            low, high = target_low, target_high
            target_low = self._normalize(target_low, low, high)
            target_high = self._normalize(target_high, low, high)
        target_obs_space = gym.spaces.Box(
            low=target_low.astype(np.float32), high=target_high.astype(np.float32)
        )
        observation_space = gym.spaces.Dict(
            {
                "image": image_obs_space,
                "target": target_obs_space,
            }
        )

        return observation_space
