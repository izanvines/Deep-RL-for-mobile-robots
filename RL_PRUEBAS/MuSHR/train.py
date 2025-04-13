import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# === CONFIG ===
models_dir = "models_AMR_130425_0/PPO"
logdir = "logs_AMR_130425"
normalize_path = "models_AMR_130425_0/vec_normalize.pkl"
timesteps = 5000
n_iterations = 1000

# Crear directorios si no existen
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# === CREAR Y ENVOLVER ENTORNO ===
def make_env():
    return gym.make("MuSHREnv-v0", render_mode="human")  # None => entrenamiento sin renderizar

dummy_env = DummyVecEnv([make_env])
env = VecNormalize(dummy_env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=5.0)

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
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

# === ENTRENAMIENTO ITERATIVO ===
for i in range(1, n_iterations + 1):
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO_AMR_130425_0")
    model.save(f"{models_dir}/{timesteps * i}")
    env.save(normalize_path)

env.close()
print("Entrenamiento finalizado.")