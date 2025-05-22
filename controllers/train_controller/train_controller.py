from controller import Supervisor
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from RobotEnv import RobotEnv
import torch
import sys
import time
import os

# Configuração fixa para CPU (removida verificação de GPU)
print("Usando CPU para treinamento")
device = "cpu"

TIME_STEP = 64

def make_env():
    return RobotEnv(verbose=False)

if __name__ == "__main__":
    # Configuração simples com DummyVecEnv
    vec_env = DummyVecEnv([make_env])
    
    # Parâmetros mais adequados para ambiente de controle de robô
    n_steps = 1024  # Valor mais baixo para facilitar o preenchimento
    batch_size = 64
    
    # Verificar modelo existente
    model_path = "ppo_epuck.zip"

    try:
        # Tenta carregar o modelo - se o espaço de observação não corresponder, captura a exceção
        if os.path.exists(model_path):
            print(f"[LOAD] Tentando carregar modelo salvo: {model_path}", file=sys.stderr)
            model = PPO.load(model_path, env=vec_env, device=device)
            model.n_steps = n_steps
            model.batch_size = batch_size

            # Recria o rollout_buffer para garantir inicialização correta
            from stable_baselines3.common.buffers import RolloutBuffer
            model.rollout_buffer = RolloutBuffer(
                model.n_steps,
                model.observation_space,
                model.action_space,
                model.device,
                gamma=model.gamma,
                gae_lambda=model.gae_lambda,
            )
            print(f"[SUCCESS] Modelo carregado com sucesso!", file=sys.stderr)
        else:
            # Se o modelo não existe, cria um novo
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
                clip_range=0.2
            )
    except ValueError as e:
        # Captura especificamente erros de incompatibilidade de espaço
        print(f"[ERROR] {e}", file=sys.stderr)
        print(f"[INFO] Modelo incompatível detectado. Renomeando modelo antigo...", file=sys.stderr)
        
        # Renomear o modelo antigo
        if os.path.exists(model_path):
            backup_name = "ppo_epuck_old_" + time.strftime("%Y%m%d_%H%M%S") + ".zip"
            os.rename(model_path, backup_name)
            print(f"[INFO] Modelo anterior renomeado para {backup_name}", file=sys.stderr)
        
        # Criar novo modelo
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
            clip_range=0.2
        )

    # Resetar o ambiente explicitamente
    vec_env.reset()
    
    # Parâmetros de treinamento
    total_timesteps = 10_000_000
    
    # Calcular intervalo de salvamento para 10 minutos de tempo simulado
    # TIME_STEP = 64ms, então 10 minutos = 600 segundos = 600.000ms
    # Número de passos para 10 minutos = 600.000 / 64
    save_interval_minutes = 10
    save_interval = int((save_interval_minutes * 60 * 1000) / TIME_STEP)
    # Ajustar para ser múltiplo de n_steps
    save_interval = (save_interval // n_steps) * n_steps
    if save_interval < n_steps:
        save_interval = n_steps
    
    print(f"Iniciando treinamento na CPU.")
    print(f"Parâmetros PPO: n_steps={n_steps}, batch_size={batch_size}")
    print(f"Salvando a cada {save_interval} passos (~{save_interval_minutes} min simulados)")

    start_time = time.time()
    total_steps_done = 0

    try:
        # Loop de treinamento com salvamento periódico
        while total_steps_done < total_timesteps:
            loop_start_time = time.time()
            
            # Prevenir casos especiais de valores muito pequenos
            steps_this_round = min(save_interval, total_timesteps - total_steps_done)
            if steps_this_round <= 0:
                break
                
            # Treinar por um intervalo
            model.learn(
                total_timesteps=steps_this_round, 
                reset_num_timesteps=False
            )
            
            total_steps_done += steps_this_round
            loop_duration = time.time() - loop_start_time
            
            # Salvar progresso
            model.save("ppo_epuck")
            
            # Estatísticas (apenas para o VSCode - via stderr)
            simulated_minutes = total_steps_done * TIME_STEP / 1000 / 60
            steps_per_second = steps_this_round / max(loop_duration, 0.001)
            eta_hours = (total_timesteps - total_steps_done) / steps_per_second / 3600
            
            print(f"[MODELO] Progresso salvo: {total_steps_done} passos ({simulated_minutes:.1f} min simulados)"
                f" - {steps_per_second:.1f} passos/s - ETA: {eta_hours:.1f}h", file=sys.stderr)
            
            # Versão simples para o console do Webots (sem stderr)
            print(f"Treinando... {total_steps_done} passos concluídos")

        # Estatísticas finais
        total_duration = time.time() - start_time
        print(f"\nTreinamento concluído em {total_duration/3600:.2f} horas.")
        print(f"Modelo final salvo em: {os.path.abspath(model_path)}")
        
    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário. Salvando modelo...", file=sys.stderr)
        model.save("ppo_epuck")  # Salva no arquivo padrão
        print(f"Modelo salvo em: {os.path.abspath(model_path)}", file=sys.stderr)
        print("Treinamento interrompido pelo usuário.")  # Para o Webots
        
    except Exception as e:
        print(f"\nERRO: {e}", file=sys.stderr)
        print("Tentando salvar o modelo antes de encerrar...", file=sys.stderr)
        try:
            model.save("ppo_epuck")  # Usar o arquivo padrão como solicitado
            print(f"Modelo salvo em: {os.path.abspath(model_path)}", file=sys.stderr)
        except:
            print("Não foi possível salvar o modelo.", file=sys.stderr)
