from controller import Supervisor
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from RobotEnv import RobotEnv
import sys
import os

device = "cpu"
print("Usando CPU para inferência")

TIME_STEP = 64

def make_env():
    return RobotEnv(verbose=False)

if __name__ == "__main__":
    vec_env = DummyVecEnv([make_env])

    model_path = "ppo_epuck.zip"

    if os.path.exists(model_path):
        print(f"[LOAD] Carregando modelo salvo de {model_path}")
        model = PPO.load(model_path, env=vec_env, device=device)
    else:
        print(f"[ERROR] Modelo '{model_path}' não encontrado. Terminar.", file=sys.stderr)
        sys.exit(1)

    obs = vec_env.reset()
    
    max_episodes = 10000
    episode_count = 0
    step_count = 0
    initial_distance = None
    episode_metrics = []

    while episode_count < max_episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        step_count += 1

        if initial_distance is None and "initial_distance" in info[0]:
            initial_distance = info[0]["initial_distance"]

        if done:
            final_distance = info[0].get("final_distance", None)
            initial_distance = initial_distance if initial_distance is not None else 1.0

            metrics = {
                "reached_goal": info[0].get("reached_goal", False),
                "collision_wall": info[0].get("collision_wall", False),
                "collision_rival": info[0].get("collision_rival", False),
                "timeout": info[0].get("timeout", False),
                "final_progress": info[0].get("final_progress", 0.0),
                "total_reward": reward[0],
                "final_distance": final_distance if final_distance is not None else 0.0,
                "steps": step_count,
                "avg_approach_speed": initial_distance / step_count if step_count > 0 else 0.0,
            }

            if metrics["collision_wall"] or metrics["collision_rival"]:
                metrics["distance_at_failure"] = final_distance if final_distance is not None else 0.0
            else:
                metrics["distance_at_failure"] = None

            episode_metrics.append(metrics)

            print(f"Episode {episode_count+1} finished:")
            print(f"Goal: {metrics['reached_goal']} | Wall: {metrics['collision_wall']} | Rival: {metrics['collision_rival']} | Timeout: {metrics['timeout']}")
            print(f"Progress: {metrics['final_progress']*100:.1f}% | Reward: {metrics['total_reward']:.2f}")
            print(f"Steps: {metrics['steps']} | Distância final: {metrics['final_distance']:.3f} | Velocidade média aproximação: {metrics['avg_approach_speed']:.4f}")
            if metrics["distance_at_failure"] is not None:
                print(f"Distância à estrela na falha: {metrics['distance_at_failure']:.3f}")
            print("─────────────────────────────")

            episode_count += 1
            step_count = 0
            initial_distance = None

    mean_reached_goal = sum(m["reached_goal"] for m in episode_metrics) / len(episode_metrics)
    mean_collision_wall = sum(m["collision_wall"] for m in episode_metrics) / len(episode_metrics)
    mean_collision_rival = sum(m["collision_rival"] for m in episode_metrics) / len(episode_metrics)
    mean_final_progress = sum(m["final_progress"] for m in episode_metrics) / len(episode_metrics)
    mean_steps = sum(m["steps"] for m in episode_metrics) / len(episode_metrics)
    mean_avg_approach_speed = sum(m["avg_approach_speed"] for m in episode_metrics) / len(episode_metrics)

    total_distance_failure = sum(m["distance_at_failure"] for m in episode_metrics if m["distance_at_failure"] is not None)
    num_failures = sum(1 for m in episode_metrics if m["distance_at_failure"] is not None)
    mean_distance_failure = total_distance_failure / num_failures if num_failures > 0 else 0

    print("\n=== Métricas Agregadas após rodar todos os episódios ===")
    print(f"Média de sucesso (alcance do objetivo): {mean_reached_goal:.3f}")
    print(f"Média colisão com parede: {mean_collision_wall:.3f}")
    print(f"Média colisão com rival: {mean_collision_rival:.3f}")
    print(f"Média progresso final: {mean_final_progress*100:.2f}%")
    print(f"Média passos por episódio: {mean_steps:.1f}")
    print(f"Média velocidade média de aproximação: {mean_avg_approach_speed:.4f}")
    print(f"Média distância à estrela nas falhas (colisões): {mean_distance_failure:.3f}")
