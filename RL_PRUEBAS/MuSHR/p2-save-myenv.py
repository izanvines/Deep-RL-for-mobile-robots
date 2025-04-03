import gym
from stable_baselines3 import PPO
import os

models_dir = "models_AMR_290325_0/PPO"
logdir = "logs_AMR_290325"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

#env = gym.make("Rover")
env = gym.make("Rover", render_mode="human")
env.reset()


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 5000
for i in range(1, 1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_AMR_290325_0")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()