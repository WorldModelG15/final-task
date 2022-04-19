import gym
from gym import spaces
import numpy as np

from gym_duckietown.simulator import Simulator


class MotionBlurWrapper(Simulator):
    def __init__(self, env=None):
        Simulator.__init__(self)
        self.env = env
        self.frame_skip = 3
        self.env.delta_time = self.env.delta_time / self.frame_skip

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        # Actions could be a Python list
        action = np.array(action)
        motion_blur_window = []
        for _ in range(self.frame_skip):
            obs = self.env.render_obs()
            motion_blur_window.append(obs)
            self.env.update_physics(action)

        # Generate the current camera image

        obs = self.env.render_obs()
        motion_blur_window.append(obs)
        obs = np.average(motion_blur_window, axis=0, weights=[0.8, 0.15, 0.04, 0.01])

        misc = self.env.get_agent_info()

        d = self.env._compute_done_reward()
        misc["Simulator"]["msg"] = d.done_why

        return obs, d.reward, d.done, misc


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space._shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype,
        )
        self.shape = shape

    # (480,640,3) -> (120,160,3)
    def observation(self, observation):
        from PIL import Image

        shape = (self.shape[1], self.shape[0])
        return np.array(Image.fromarray(observation).resize(shape))


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype,
        )

    # (120,160,3) -> (3,120,160)
    def observation(self, observation):
        return observation.transpose(2, 0, 1)


# 2の報酬設計でloop_pedestriansだとそれなりに進んでいる（だいたいアヒルと衝突して終了）
class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        # 1.default
        # if reward == -1000:
        #     reward = -10
        # elif reward > 0:
        #     reward += 10
        # else:
        #     reward += 4

        # 2.original_1
        # if reward == -1000:
        #     return reward
        # if reward < 0:
        #     reward = 0
        if reward == -1000:
            reward = -1
        if reward < 0:
            reward *= 0.1

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_


# early stop learning
class OriginalWrapper(gym.Wrapper):
    def __init__(self, env):
        super(OriginalWrapper, self).__init__(env)
        self.env = env
        self.total_steps = 0
        self.total_reward = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.total_reward += reward
        self.total_steps += 1
        if self.total_steps >= 3000:
            done = True
        # バック走行ペナルティ
        if action[0] < 0 and action[1] < 0:
            reward = -1.0 #-1
        # 前進に報酬
        if action[0] > 0 and action[1] > 0:
            reward = 1.0
        return next_state, reward, done, info

    def reset(self):
        self.total_steps = 0
        self.total_reward = 0
        return self.env.reset()
