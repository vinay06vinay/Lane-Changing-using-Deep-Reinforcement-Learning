import highway_env
import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse


'''
A Proximal Policy Optimization Algorithm is used to train the Highway Lane Changing environment.
'''
def ppo_train(args):
    timesteps = args.time_steps
    learning_rate = args.learning_rate
    n_cpu = 6
    batch_size = 32
    env = make_vec_env('highway-lane-v0', n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=20,
            learning_rate=learning_rate,
            gamma=0.8,
            verbose=2,
            tensorboard_log="highway_ppo/",
        )
    model.learn(total_timesteps=timesteps)
    # Save the agent
    model.save("highway_ppo/model")
    del model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN model for highway Lane Changing Environment environment')
    parser.add_argument('--time-steps', type=int, default=300, help='Number of training time steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for DQN')
    args = parser.parse_args()
    ppo_train(args)
