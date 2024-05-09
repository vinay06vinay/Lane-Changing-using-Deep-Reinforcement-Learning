import highway_env
import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo

def test_dqn():
    env = gym.make('highway-lane-v0', render_mode="rgb_array")
    # Load and test saved model
    model = DQN.load("highway_dqn/model/dqn")
    print(env.get_available_actions())
    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()



if __name__ == "__main__":
  test_dqn()