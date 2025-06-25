from controller import Supervisor
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from RobotEnv import RobotEnv
import torch
import sys
import time
import os

print("Usando CPU para treinamento")
device = "cpu"
TIME_STEP = 64

def make_env():
    return RobotEnv(verbose=False)

if __name__ == "__main__":
    vec_env = DummyVecEnv([make_env])
    n_steps = 1024
    batch_size = 64
    model_path = "ppo_epuck.zip"

    try:
        if os.path.exists(model_path):
            print(f"[LOAD] Tentando carregar modelo salvo: {model_path}", file=sys.stderr)
            model = PPO.load(model_path, env=vec_env, device=device)
            model.n_steps = n_steps
            model.batch_size = batch_size
            print(f"[SUCCESS] Modelo carregado com sucesso!", file=sys.stderr)
        else:
            print(f"[NEW] Criando novo modelo (device={device})", file=sys.stderr)
            model = PPO(
                policy="MlpPolicy",
                env=vec_env,
                verbose=1,
                device=device,
                n_steps=n_steps,
                batch_size=batch_size,
                learning_rate=3e-4,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                clip_range=0.2,
                max_grad_norm=0.5
            )
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print(f"[INFO] Modelo incompatível detectado. Renomeando modelo antigo...", file=sys.stderr)
        if os.path.exists(model_path):
            backup_name = "ppo_epuck_old_" + time.strftime("%Y%m%d_%H%M%S") + ".zip"
            os.rename(model_path, backup_name)
            print(f"[INFO] Modelo anterior renomeado para {backup_name}", file=sys.stderr)
        print(f"[NEW] Criando novo modelo com espaço de observação expandido", file=sys.stderr)
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            device=device,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=3e-4,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            max_grad_norm=0.5,
            vf_coef=0.25
        )

    vec_env.reset()
    total_timesteps = 10_000_000

    save_interval_minutes = 10
    save_interval = int((save_interval_minutes * 60 * 1000) / TIME_STEP)
    save_interval = (save_interval // n_steps) * n_steps
    if save_interval < n_steps:
        save_interval = n_steps

    print(f"Iniciando treinamento na CPU.")
    print(f"Parâmetros PPO: n_steps={n_steps}, batch_size={batch_size}")
    print(f"Salvando a cada {save_interval} passos (~{save_interval_minutes} min simulados)")

    start_time = time.time()
    total_steps_done = 0

    try:
        while total_steps_done < total_timesteps:
            loop_start_time = time.time()
            steps_this_round = min(save_interval, total_timesteps - total_steps_done)
            if steps_this_round <= 0:
                break
            model.learn(
                total_timesteps=steps_this_round, 
                reset_num_timesteps=False
            )
            total_steps_done += steps_this_round
            loop_duration = time.time() - loop_start_time
            model.save("ppo_epuck")
            simulated_minutes = total_steps_done * TIME_STEP / 1000 / 60
            steps_per_second = steps_this_round / max(loop_duration, 0.001)
            eta_hours = (total_timesteps - total_steps_done) / steps_per_second / 3600
            print(f"[MODELO] Progresso salvo: {total_steps_done} passos ({simulated_minutes:.1f} min simulados)"
                f" - {steps_per_second:.1f} passos/s - ETA: {eta_hours:.1f}h", file=sys.stderr)
            print(f"Treinando... {total_steps_done} passos concluídos")

        total_duration = time.time() - start_time
        print(f"\nTreinamento concluído em {total_duration/3600:.2f} horas.")
        print(f"Modelo final salvo em: {os.path.abspath(model_path)}")
        
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário. Salvando modelo...", file=sys.stderr)
        model.save("ppo_epuck")
        print(f"Modelo salvo em: {os.path.abspath(model_path)}", file=sys.stderr)
        print("Treinamento interrompido pelo usuário.")
        
    except Exception as e:
        print(f"\nERRO: {e}", file=sys.stderr)
        print("Tentando salvar o modelo antes de encerrar...", file=sys.stderr)
        try:
            model.save("ppo_epuck")
            print(f"Modelo salvo em: {os.path.abspath(model_path)}", file=sys.stderr)
        except:
            print("Não foi possível salvar o modelo.", file=sys.stderr)
