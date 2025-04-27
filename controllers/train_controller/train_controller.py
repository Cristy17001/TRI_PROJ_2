from controller import Supervisor
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from RobotEnv import RobotEnv

TIME_STEP = 64

def make_env():
    # Each env will grab its own Supervisor() internally
    return RobotEnv()

if __name__ == "__main__":
    # 1) Wrap your single‐env in a DummyVecEnv
    vec_env = DummyVecEnv([make_env])
    # 2) Create and train the model
    model = PPO('MlpPolicy', vec_env, verbose=1, device='cpu')
    model.learn(total_timesteps=10_000_000)
    # 3) Save the trained policy
    model.save("ppo_epuck")
