import highway_env
import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import RecordVideo

def test_ppo():
    env = gym.make("highway-lane-v0", render_mode = "rgb_array")
    # Load and test saved model
    model = PPO.load("highway_ppo/model")    
    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()



if __name__ == "__main__":
  test_ppo()