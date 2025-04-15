import os
import gym, gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
import torch

# --- SEMILLA ---
SEED = 42  # o el número que prefieras
np.random.seed(SEED)
torch.manual_seed(SEED)

# === CONFIG ===
models_dir = "models_AMR_150425_3/PPO"
logdir = "logs_AMR_150425"
normalize_path = "models_AMR_150425_3/vec_normalize.pkl"
timesteps = 5000
n_iterations = 1000

# Crear directorios si no existen
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# === CREAR Y ENVOLVER ENTORNO ===
def make_env():
    #return gym.make("MuSHREnv-v0")
    #return gym.make("MuSHREnv-v0", render_mode="human") #-> con entorno gráfico
    return gymnasium.make("MuSHREnv-v0")
    #return gymnasium.make("MuSHREnv-v0", render_mode="human") #-> con entorno gráfico

vec_env = make_vec_env(make_env, n_envs=1, seed=SEED) #Por defeecto usa gymnasium -> ya se ha añadido el entorno a gymnasium para poder usarlo (menos warnings)

# Normalizar observaciones y recompensas
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=5.0)

"""
# === COMPROBACIÓN DE NORMALIZACIÓN  ===
obs_norm = env.reset()
obs_raw = env.get_original_obs()
obs_recalc = env.normalize_obs(obs_raw)

print("Observación normalizada del reset:", obs_norm)
print("Observación sin normalizar (original):", obs_raw)
print("Observación original transformada (con normalize_obs):", obs_recalc)
"""

# === CREAR MODELO ===
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=logdir)

# === ENTRENAMIENTO ITERATIVO ===
for i in range(1, n_iterations + 1):
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO_AMR_150425_3")
    model.save(f"{models_dir}/{timesteps * i}")
    vec_env.save(normalize_path)

vec_env.close()
print("Entrenamiento finalizado.")