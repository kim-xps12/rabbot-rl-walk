import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

class QuadrupedAirEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # --- Load your MuJoCo model ---
        # Get the directory of this file and construct path to XML
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "../model/quadruped_air.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep

        # --- Action space ---
        # あなたのロボットはモーター8個 → ctrl が8次元
        self.action_space = spaces.Box(
            low=np.full(8, -1.0),
            high=np.full(8, 1.0),
            dtype=np.float32
        )

        # --- Observation space ---
        # 胴体姿勢＋関節角度・速度などを返す
        obs_high = np.inf * np.ones(20)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

    def _get_obs(self):
        # 胴体の角度（クォータニオン）、関節角、関節速度など
        torso_quat = self.data.qpos[3:7]
        joint_angles = self.data.qpos[7:15]
        joint_vels = self.data.qvel[6:14]

        return np.concatenate([torso_quat, joint_angles, joint_vels])

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)

        # スタート位置
        self.data.qpos[2] = 0.45

        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # ctrl に RL の行動を入力
        self.data.ctrl[:] = action

        # 1ステップ進める
        mujoco.mj_step(self.model, self.data)

        # 観測
        obs = self._get_obs()

        # --- Reward 設計 ---
        forward_reward = self.data.qvel[0]     # x方向速度
        height_penalty = -abs(0.45 - self.data.qpos[2]) * 1.0
        tilt_penalty = -np.linalg.norm(self.data.qpos[3:7] - np.array([1,0,0,0])) * 2.0

        reward = forward_reward + height_penalty + tilt_penalty

        # --- Termination ---
        terminated = False
        if self.data.qpos[2] < 0.2:     # 転んだら終了
            terminated = True

        truncated = False

        return obs, reward, terminated, truncated, {}

