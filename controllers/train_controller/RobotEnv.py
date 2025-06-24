import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import Supervisor
import sys

# Constantes globais mais focadas
TIME_STEP = 64
MAX_SPEED = 4.0
COLLISION_THRESHOLD = 0.08
MIN_INITIAL_DISTANCE = 0.3
SPAWN_RANGE = 0.9
MAX_ACCEL = 1.0

# Constantes de recompensa PADRONIZADAS por importância (escala 0-10)
# CRÍTICAS (8-10): Objetivos principais e comportamentos destrutivos
GOAL_REWARD = 200.0              # [10] Alcançar a estrela (objetivo principal) - AUMENTADO
WALL_PENALTY = -50.0           # [9] Colisão com parede (evitar a todo custo) - AUMENTADO
ROBOT_COLLISION_PENALTY = -15.0 # [7] Colisão com rival (séria mas recuperável)

# IMPORTANTES (5-7): Progresso e competitividade
PROGRESS_REWARD_BASE = 2.0      # [6] Recompensa base por progresso
OVERTAKING_REWARD = 1.0         # [6] Ultrapassagem do rival
LEADING_BONUS = 0.8             # [5] Bônus por liderar a corrida

# COMPORTAMENTAIS (3-5): Estabilidade e eficiência
CONSISTENT_MOVEMENT_REWARD = 0.5  # [4] Movimento consistente >2s
ORIENTATION_REWARD = 0.3        # [3] Orientação correta
APPROACH_REWARD = 0.2           # [3] Aproximação estratégica do rival

# PENALIZAÇÕES LEVES (1-3): Correção de comportamento
SIGN_CHANGE_PENALTY = -0.15     # [3] Mudança de sinal dos motores
PROXIMITY_PENALTY = -0.10       # [2] Proximidade excessiva ao rival
WHEEL_INEFFICIENCY_PENALTY = -0.3  # [2] Movimento ineficiente
STEP_PENALTY = -0.005           # [1] Penalização por tempo (reduzida)

MAX_LINEAR_SPEED = 0.08      # Aumentado de 0.05 para 0.08 (60% mais rápido)
MAX_ANGULAR_SPEED = 8.0      # Aumentado de 6.28 para 8.0 (mais agilidade)
WHEEL_RADIUS = 0.0205        # Raio da roda do e-puck (m)
AXLE_LENGTH = 0.052          # Distância entre rodas (m)


class RobotEnv(gym.Env):
    def __init__(self, verbose=True):
        super(RobotEnv, self).__init__()
        
        self.verbose = verbose
        self.supervisor = Supervisor()
        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.gps = self.supervisor.getDevice("gps")
        self.imu = self.supervisor.getDevice("inertial unit")
        
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.gps.enable(TIME_STEP)
        self.imu.enable(TIME_STEP)

        self.robot_node = self.supervisor.getSelf()
        self.robot_translation = self.robot_node.getField("translation")
        self.star_node = self.supervisor.getFromDef("Star")
        self.star_translation = self.star_node.getField("translation")
        
        # Adicionar referência ao robô rival
        self.rival_robot_node = self.supervisor.getFromDef("RivalRobot")
        if self.rival_robot_node is None:
            print("AVISO: RivalRobot não encontrado!")
        
        self.prev_dist = None  # To track starting distance from target
        
        # Modificar o espaço de observação para incluir a posição do robô rival
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]),
                               high=np.array([1.0, 1.0]),
                               dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.previous_distance = None
        
        # Variáveis para rastreamento de comportamento dos motores
        self.motor_command_history = []  # Histórico de comandos dos motores
        self.consistent_movement_start = None  # Início do movimento consistente
        self.consistent_movement_duration = 0  # Duração do movimento consistente
        self.prev_left_speed = 0.0  # Velocidade anterior do motor esquerdo
        self.prev_right_speed = 0.0  # Velocidade anterior do motor direito
        self.sign_change_penalty_accumulator = 0.0  # Acumulador de penalizações por mudança de sinal
        
        self.reset()
    
    def log(self, message):
        """Helper method to print messages only when verbose is True"""
        if self.verbose:
            print(message)
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Spawn aleatório mais eficiente
        while True:
            robot_x = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            robot_y = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            star_x = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            star_y = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            
            # Cálculo único de distância
            dx, dy = star_x - robot_x, star_y - robot_y
            initial_distance = np.sqrt(dx*dx + dy*dy)
            
            if initial_distance >= MIN_INITIAL_DISTANCE:
                break

        self.robot_translation.setSFVec3f([robot_x, robot_y, 0])
        self.robot_node.resetPhysics()
        
        self.star_translation.setSFVec3f([star_x, star_y, 0.03])
        self.star_node.resetPhysics()
        
        # Calcula direção do robô principal para a estrela
        direction = [
            star_x - robot_x,
            star_y - robot_y,
            0
        ]
        norm = (direction[0]**2 + direction[1]**2 + direction[2]**2) ** 0.5
        if norm != 0:
            direction = [d / norm for d in direction]

        rival_spawn_pos = [
            star_x - direction[0] * 0.1,
            star_y - direction[1] * 0.1,
            0
        ]

        # Define a posição do rival
        self.rival_robot_node.getField("translation").setSFVec3f(rival_spawn_pos)
        self.rival_robot_node.resetPhysics()

        # Cálculo otimizado de time_limit (DUPLICADO para dar mais tempo)
        self.initial_distance = initial_distance
        half_max_linear_speed = 0.5 * MAX_SPEED * 0.02
        self.time_limit = int(8 * (initial_distance / half_max_linear_speed) * 1000 / TIME_STEP)  # Duplicado de 4 para 8
        self.step_count = 0
        
        self.log(f"[RESET] Robô: ({robot_x:.2f}, {robot_y:.2f}), Estrela: ({star_x:.2f}, {star_y:.2f}), "
              f"Dist: {initial_distance:.2f}, Tempo: {self.time_limit} (2x mais tempo)")
        
        self.previous_distance = initial_distance
        
        # Reset das variáveis de comportamento dos motores
        self.motor_command_history = []
        self.consistent_movement_start = None
        self.consistent_movement_duration = 0
        self.prev_left_speed = 0.0
        self.prev_right_speed = 0.0
        self.sign_change_penalty_accumulator = 0.0
    
        # Avançar simulação antes de começar
        for _ in range(10):
            self.supervisor.step(TIME_STEP)
        
        return self._get_obs(), {}

    def _get_obs(self):
        pos = self.gps.getValues()
        yaw = self.imu.getRollPitchYaw()[2]  # Mais eficiente que indexar toda array
        normalized_yaw = yaw / np.pi
        
        star_pos = self.star_node.getPosition()
        
        # Cálculo direto e eficiente de distância
        dx, dy = pos[0] - star_pos[0], pos[1] - star_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Obter posição do robô rival
        rival_pos = self.rival_robot_node.getPosition() if self.rival_robot_node else [0, 0, 0]
        
        # Retornar observação ampliada com posição do robô rival
        return np.array([
            pos[0], pos[1], 
            normalized_yaw, 
            dist, 
            star_pos[0], star_pos[1], 
            rival_pos[0], rival_pos[1]
        ], dtype=np.float32)


    def _calculate_progress_reward(self, distance_reduction, pos_x, pos_y, rival_x, rival_y, star_x, star_y):
        """Recompensa PADRONIZADA baseada em progresso - DUPLICADA quando mais perto que o rival."""
        base_reward = 0.0
        
        if distance_reduction > 0.001:
            # Recompensa proporcional usando constante padronizada
            base_reward = min(distance_reduction * PROGRESS_REWARD_BASE * 50.0, PROGRESS_REWARD_BASE)
        elif abs(distance_reduction) <= 0.001:
            # Sem penalização por estagnação breve
            base_reward = 0.0
        else:
            # Penalização leve por retrocesso
            base_reward = max(distance_reduction * PROGRESS_REWARD_BASE * 25.0, -PROGRESS_REWARD_BASE * 0.5)
        
        # Calcular distâncias para verificar se deve duplicar
        robot_to_star = np.sqrt((pos_x - star_x)**2 + (pos_y - star_y)**2)
        rival_to_star = np.sqrt((rival_x - star_x)**2 + (rival_y - star_y)**2)
        
        # DUPLICAR recompensa se robot está mais perto da estrela que o rival
        if robot_to_star < rival_to_star and base_reward > 0:
            # Calcular quão mais próximo o robot está comparado ao rival
            distance_advantage = rival_to_star - robot_to_star
            
            # Duplicar sempre, mas dar um bônus extra se a vantagem for significativa
            doubled_reward = base_reward * 2.0
            if distance_advantage > 0.1:  # Vantagem significativa (10cm ou mais)
                doubled_reward += LEADING_BONUS  # Bônus padronizado por liderança clara
                self.log(f"[✓✓✓] TRIPLO Progresso (liderança clara): +{doubled_reward:.3f}")
            else:
                self.log(f"[✓✓] DUPLO Progresso (mais perto que rival): +{doubled_reward:.3f}")
            return doubled_reward
        elif base_reward > 0:
            self.log(f"[✓] Progresso: +{base_reward:.3f}")
            return base_reward
        elif base_reward < 0:
            self.log(f"[✗] Retrocesso: {base_reward:.3f}")
            return base_reward
        else:
            return 0.0

    def _calculate_wheel_penalty(self, left_speed, right_speed):
        """Penalização PADRONIZADA por movimento ineficiente."""
        wheel_diff = abs(left_speed - right_speed)
        avg_speed = (abs(left_speed) + abs(right_speed)) / 2
        
        # Penalizar apenas comportamentos claramente ruins
        if avg_speed < 0.1 and wheel_diff > 2.0:
            # Robô girando no lugar sem avançar
            self.log(f"[!] Girando sem avançar: {WHEEL_INEFFICIENCY_PENALTY * 2:.3f}")
            return WHEEL_INEFFICIENCY_PENALTY * 2
        elif wheel_diff > 4.0:
            # Oscilação excessiva
            self.log(f"[!] Oscilação excessiva: {WHEEL_INEFFICIENCY_PENALTY:.3f}")
            return WHEEL_INEFFICIENCY_PENALTY
            
        return 0.0

    def _calculate_rival_proximity_penalty(self, pos_x, pos_y, rival_x, rival_y):
        """Penalização PADRONIZADA por proximidade ao robô rival."""
        dx, dy = pos_x - rival_x, pos_y - rival_y
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.001:
            dist = 0.001  # evita divisão por zero

        # Penalização gradual apenas quando muito perto (reduzida para incentivar aproximação)
        if dist < 0.12:  # Zona de perigo reduzida de 0.15 para 0.12
            penalty = PROXIMITY_PENALTY * (0.12 - dist) / 0.12  # Penalização padronizada
            self.log(f"[!] Muito perto rival ({dist:.3f}m): {penalty:.3f}")
            return penalty
        return 0.0

    def _calculate_overtaking_reward(self, pos_x, pos_y, rival_x, rival_y, star_x, star_y):
        """Recompensa PADRONIZADA por navegar próximo ao rival em direção ao objetivo (overtaking)."""
        # Distância do robot ao rival
        rival_dist = np.sqrt((pos_x - rival_x)**2 + (pos_y - rival_y)**2)
        
        # Distância do robot à estrela
        robot_to_star = np.sqrt((pos_x - star_x)**2 + (pos_y - star_y)**2)
        
        # Distância do rival à estrela
        rival_to_star = np.sqrt((rival_x - star_x)**2 + (rival_y - star_y)**2)
        
        # Se o robot está mais perto da estrela que o rival e numa zona próxima ao rival
        if robot_to_star < rival_to_star and 0.12 < rival_dist < 0.25:
            self.log(f"[★] Overtaking rival! Reward: +{OVERTAKING_REWARD:.2f}")
            return OVERTAKING_REWARD
        
        # Recompensa menor por estar numa posição de aproximação estratégica
        elif 0.15 < rival_dist < 0.3:
            return APPROACH_REWARD
            
        return 0.0

    def _calculate_orientation_reward(self, pos_x, pos_y, yaw, star_x, star_y):
        """Recompensa PADRONIZADA por estar orientado na direção do objetivo."""
        # Calcular ângulo para a estrela
        dx, dy = star_x - pos_x, star_y - pos_y
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return 0.0  # Já no objetivo
            
        target_angle = np.arctan2(dy, dx)
        
        # Diferença angular (considerando wrap-around)
        angle_diff = abs(yaw - target_angle)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        
        # Recompensa quando bem orientado
        if angle_diff < np.pi/6:  # 30 graus
            reward = ORIENTATION_REWARD * (1 - angle_diff / (np.pi/6))
            return reward
        return 0.0

    def _calculate_consistent_movement_reward(self, left_speed, right_speed):
        """Recompensa PADRONIZADA por movimento consistente (mesmo comando nos 2 motores por >2s)."""
        # Verificar se os motores estão com comandos similares (tolerância de 5%)
        speed_avg = (abs(left_speed) + abs(right_speed)) / 2
        speed_diff = abs(left_speed - right_speed)
        
        # Considerar movimento consistente se a diferença for pequena e há movimento
        is_consistent = (speed_diff < speed_avg * 0.05) and (speed_avg > 0.1)
        
        current_time = self.step_count * TIME_STEP / 1000.0  # Converter para segundos
        
        if is_consistent:
            if self.consistent_movement_start is None:
                self.consistent_movement_start = current_time
                self.consistent_movement_duration = 0
            else:
                self.consistent_movement_duration = current_time - self.consistent_movement_start
                
            # Recompensa após 2 segundos de movimento consistente
            if self.consistent_movement_duration >= 2.0:
                # Recompensa proporcional à duração (máximo padronizado)
                reward = min(CONSISTENT_MOVEMENT_REWARD * 0.4 + (self.consistent_movement_duration - 2.0) * 0.05, CONSISTENT_MOVEMENT_REWARD)
                self.log(f"[⚡] Movimento consistente ({self.consistent_movement_duration:.1f}s): +{reward:.3f}")
                return reward
        else:
            # Reset do contador se movimento não for mais consistente
            self.consistent_movement_start = None
            self.consistent_movement_duration = 0
            
        return 0.0

    def _calculate_sign_change_penalty(self, left_speed, right_speed):
        """Penalização PADRONIZADA por mudanças frequentes de sinal nos motores."""
        penalty = 0.0
        
        # Verificar mudança de sinal no motor esquerdo
        if (self.prev_left_speed > 0.1 and left_speed < -0.1) or \
           (self.prev_left_speed < -0.1 and left_speed > 0.1):
            penalty += SIGN_CHANGE_PENALTY
            self.log(f"[⚠] Mudança sinal motor esquerdo: {SIGN_CHANGE_PENALTY:.3f}")
            
        # Verificar mudança de sinal no motor direito
        if (self.prev_right_speed > 0.1 and right_speed < -0.1) or \
           (self.prev_right_speed < -0.1 and right_speed > 0.1):
            penalty += SIGN_CHANGE_PENALTY
            self.log(f"[⚠] Mudança sinal motor direito: {SIGN_CHANGE_PENALTY:.3f}")
            
        # Penalização extra se ambos os motores mudaram de sinal simultaneamente
        if penalty <= (SIGN_CHANGE_PENALTY * 1.8):  # Ambos mudaram (com tolerância)
            penalty += SIGN_CHANGE_PENALTY * 1.3  # Penalização adicional padronizada
            self.log(f"[⚠⚠] Mudança sinal simultânea: {SIGN_CHANGE_PENALTY * 1.3:.3f} extra")
            
        # Atualizar velocidades anteriores
        self.prev_left_speed = left_speed
        self.prev_right_speed = right_speed
        
        return penalty

    def step(self, action):
        # Obtem v e ω da ação normalizada
        v = action[0] * MAX_LINEAR_SPEED
        omega = action[1] * MAX_ANGULAR_SPEED

        # Converte para velocidades das rodas
        target_left_speed = (v - omega * AXLE_LENGTH / 2) / WHEEL_RADIUS
        target_right_speed = (v + omega * AXLE_LENGTH / 2) / WHEEL_RADIUS

        # Limitar aceleração
        current_left_speed = self.left_motor.getVelocity()
        current_right_speed = self.right_motor.getVelocity()

        delta_left = target_left_speed - current_left_speed
        delta_right = target_right_speed - current_right_speed

        left_speed = (
            current_left_speed + np.clip(delta_left, -MAX_ACCEL, MAX_ACCEL)
        )
        right_speed = (
            current_right_speed + np.clip(delta_right, -MAX_ACCEL, MAX_ACCEL)
        )

        # Clamp velocidades finais
        left_speed = np.clip(left_speed, -MAX_SPEED, MAX_SPEED)
        right_speed = np.clip(right_speed, -MAX_SPEED, MAX_SPEED)

        # Aplicar velocidades
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        self.supervisor.step(TIME_STEP)

        # Obter observações
        obs = self._get_obs()
        pos_x, pos_y, normalized_yaw, dist, star_x, star_y, rival_x, rival_y = obs

        # Recompensa base: pequena penalização por tempo para incentivar eficiência
        reward = STEP_PENALTY
        terminated = False
        truncated = False

        # Cálculo do progresso
        distance_reduction = self.previous_distance - dist
        self.previous_distance = dist

        # Sistema de recompensas PADRONIZADO por importância
        reward += self._calculate_progress_reward(distance_reduction, pos_x, pos_y, rival_x, rival_y, star_x, star_y)
        reward += self._calculate_wheel_penalty(left_speed, right_speed)
        # REMOVIDO: reward += self._calculate_rival_proximity_penalty(pos_x, pos_y, rival_x, rival_y)
        reward += self._calculate_orientation_reward(pos_x, pos_y, normalized_yaw * np.pi, star_x, star_y)
        reward += self._calculate_overtaking_reward(pos_x, pos_y, rival_x, rival_y, star_x, star_y)
        reward += self._calculate_consistent_movement_reward(left_speed, right_speed)
        reward += self._calculate_sign_change_penalty(left_speed, right_speed)

        # Recompensas por marcos de proximidade (padronizadas)
        progress_ratio = 1 - (dist / self.initial_distance)
        if progress_ratio > 0.8:
            reward += LEADING_BONUS * 0.625  # Muito perto do objetivo (0.5 padronizado)
        elif progress_ratio > 0.6:
            reward += LEADING_BONUS * 0.375  # Razoavelmente perto (0.3 padronizado)
        elif progress_ratio > 0.4:
            reward += LEADING_BONUS * 0.125  # Fazendo progresso (0.1 padronizado)

        # Incrementar contador de passos para rastreamento temporal
        self.step_count += 1

        # Check de objetivo
        if dist < COLLISION_THRESHOLD:
            reward += GOAL_REWARD
            terminated = True
            self.log("[✓] Objetivo alcançado!")

        # Check de colisão com parede
        if abs(pos_x) >= 0.95 or abs(pos_y) >= 0.95:
            reward += WALL_PENALTY
            terminated = True
            self.log("[✗] Colisão com parede!")

        # Check de colisão com rival - distância ajustada para dar mais espaço para manobras
        if self.rival_robot_node:
            rival_dist = np.linalg.norm([pos_x - rival_x, pos_y - rival_y])
            if rival_dist < 0.08:  # Reduzido de 0.09 para 0.08 (permite mais proximidade)
                reward += ROBOT_COLLISION_PENALTY
                terminated = True
                self.log("[✗] Colisão com robô rival!")

        # Limite de tempo - recompensa normalizada baseada em progresso final
        self.step_count += 1
        if self.step_count >= self.time_limit:
            truncated = True
            progress = 1 - (dist / self.initial_distance)
            if progress > 0.7:  # Excelente progresso
                reward += 3.0
                self.log(f"[!] Tempo esgotado - Excelente progresso: {progress * 100:.1f}%")
            elif progress > 0.5:  # Bom progresso
                reward += 1.5
                self.log(f"[!] Tempo esgotado - Bom progresso: {progress * 100:.1f}%")
            elif progress > 0.3:  # Progresso moderado
                reward += 0.5
                self.log(f"[!] Tempo esgotado - Progresso moderado: {progress * 100:.1f}%")
            else:
                reward -= 1.0  # Progresso insuficiente
                self.log(f"[!] Tempo esgotado - Progresso insuficiente: {progress * 100:.1f}%")

        # Log periódico com informações úteis
        if self.step_count % 50 == 0:  # Menos frequente para reduzir ruído
            progress_pct = (1 - dist / self.initial_distance) * 100
            self.log(f"[i] Step {self.step_count}: Dist={dist:.3f}, Reward={reward:.3f}, Progress={progress_pct:.1f}%")

        return obs, reward, terminated, truncated, {}


