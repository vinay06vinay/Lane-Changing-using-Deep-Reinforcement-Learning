import highway_env
# from highway_env import register_highway_envs
# register_highway_envs()
import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
import argparse

'''
A DQN Agent from stable baseline is used to train the environment created highway lane v0 with hyperparameter tuning in place.
'''
def dqn_train(args):
    timesteps = args.time_steps
    learning_rate = args.learning_rate
    env = gym.make('highway-lane-v0', render_mode="rgb_array")
    observation, info = env.reset()
    model = DQN('MlpPolicy', env, policy_kwargs = dict(net_arch=[256, 256]),
                learning_rate = learning_rate, buffer_size = 1500, learning_starts = 200,
                batch_size = 128, gamma = 0.7, train_freq = 5, gradient_steps = 1,
                target_update_interval = 50, verbose = 1, tensorboard_log = 'tensorboard_logs/')
    model.learn(timesteps)
    model.save('highway_dqn/model/dqn')
    del model
    # action = env.action_space.sample()
    # observation, reward, terminated, truncated, info = env.step(action)
    # env.render()
    # for _ in range(100):
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     env.render()
    #     if terminated or truncated:
    #         observation, info = env.reset()
    # plt.imshow(env.render())
    # plt.show()
    # env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN model for highway Lane Changing Environment environment')
    parser.add_argument('--time-steps', type=int, default=300, help='Number of training time steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for DQN')
    args = parser.parse_args()

    dqn_train(args)
