from argparse import Action
import os
import torch
import gym
from gym.utils.play import play, PlayPlot
from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import pygame




env = make_vec_env("CarRacing-v0", n_envs=4)
# normal reset, this changes the colour scheme by default
'''Action Space
If continuous: There are 3 actions: steering (-1 is full left, +1 is full right), gas, and breaking. If discrete: There are 5 actions: do nothing, steer left, steer right, gas, brake.
'''


Action_Space = Box(low=np.array([-1., 0., 0.]), high=1.0, shape=(3,), dtype=np.float32)

'''Observation Space
State consists of 96x96 pixels.:'''
Observation_Space =  Box(low=0, high=255, shape=(96, 96, 3), dtype= np.uint8)

# Model Learning Hyperparams
factor = 0.9
stop_factor_lr = 1e-3
lr = 0.1

def lr_call(step):
    global lr, stop_factor_lr, factor
    lr = max(stop_factor_lr, lr * factor)
    return lr


model = PPO (policy= "CnnPolicy",#observation_space=Observation_Space,observation_space=Observation_Space,action_space=Action_Space,
            #   policy_kwargs=dict(lr_schedule=lr_call, activation_fn=torch.nn.modules.activation.Tanh),
             env=env, verbose=1, learning_rate=lr_call, n_steps=1024, batch_size=128, n_epochs=10, device="cuda", tensorboard_log="CR_Racing_Model_1")

model.learn(total_timesteps=29500)


model.save('./Models/PPO_CarRacing_19500T')

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()