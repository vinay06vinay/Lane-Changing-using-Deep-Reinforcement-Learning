# A Lane Changing Highway Agent Using Deep Reinforcement Learning

This repository contains two different Reinforcement Learning Algorithms such as DQN (Deep Q-Learning) and PPO (Proximal Policy Optimization) algorithms trained on a Custom created highway environment in reference to [1]. 

*System Requirements: Ubuntu 22/ 20, Anaconda Distribution*

### Dependenices Installation:
1. Install Anaconda on your Ubuntu system if it already not installed. Refer to the official website [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/linux/). Open a Terminal Window and run below commands after modifying the installer version
```bash
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 #Contiue with next commands if some of the libraries fail to install
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh #Get the version from Anaconda Website
bash ~/Anaconda3-2024.02-1-Linux-x86_64.sh
```
2. Complete the Installation of Anaconda further by visiting the website and activate conda by below command
```bash
conda init
```
3. Clone the below repository 
```bash
cd ..
git clone https://github.com/vinay06vinay/Lane-Changing-using-Deep-Reinforcement-Learning.git
```
4. Create a new Conda environment with below dependencies. For Pytorch dependencies based on your system refer to Website [Pytorch](https://pytorch.org/get-started/locally/)
```bash
conda create --name enpm690
conda activate enpm690
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
cd Lane-Changing-using-Deep-Reinforcement-Learning
pip install highway-env
pip install stable_baselines3
pip install gymnasium
```
5. Make sure tensorboard is installed

### Instructions to Train and Test the Code:

#### DQN Agent Training and Testing

1. To train the DQN Agent run the below command with learning rate and timesteps as per convenience
```bash
 python3 highway_lane_train_dqn.py --time-steps 1000 --learning-rate 1e-4
```
2. The folder "highway_dqn/model" contains Weights for already tested DQN model. You can test it using below command
```bash
python3 highway_lane_test_dqn.py
```
**Results**: You can see the agent running on the customly created highway environment. The logs can be found in "tensorboard_logs/" if you have tensorboard setup

#### PPO Agent Training and Testing

1. To train the PPO Agent run the below command with learning rate and timesteps as per convenience
```bash
python3 highway_lane_train_ppo.py --time-steps 1000 --learning-rate 1e-4
```
2. The folder "highway_ppo/" contains Weights for already tested DQN model. You can test it using below command
```bash
python3 highway_lane_test_ppo.py
```

**Results**: You can see the agent running on the customly created highway environment. The logs can be found in "highway_ppo/" if you have tensorboard setup


## References & Credits

[1] Highway Env- https://github.com/Farama-Foundation/HighwayEnv/tree/master
