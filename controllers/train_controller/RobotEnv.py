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

# Constantes de recompensa simplificadas
GOAL_REWARD = 500.0
WALL_PENALTY = -100.0
STEP_PENALTY = -0.05  # Penalização leve por passo para incentivar eficiência

class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        
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
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.previous_distance = None
        self.reset()
    
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

        self.robot_translation.setSFVec3f([robot_x, robot_y, 0.03])
        self.star_translation.setSFVec3f([star_x, star_y, 0.03])

        # Cálculo otimizado de time_limit
        self.initial_distance = initial_distance
        half_max_linear_speed = 0.5 * MAX_SPEED * 0.02
        self.time_limit = int(4 * (initial_distance / half_max_linear_speed) * 1000 / TIME_STEP)
        self.step_count = 0
        
        print(f"[RESET] Robô: ({robot_x:.2f}, {robot_y:.2f}), Estrela: ({star_x:.2f}, {star_y:.2f}), "
              f"Dist: {initial_distance:.2f}, Tempo: {self.time_limit}")
        
        self.previous_distance = initial_distance
        
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
        
        return np.array([pos[0], pos[1], normalized_yaw, dist, star_pos[0], star_pos[1]], dtype=np.float32)

    def step(self, action):
        # Calcular velocidades alvo
        target_left_speed = action[0] * MAX_SPEED
        target_right_speed = action[1] * MAX_SPEED
        
        # Limitar aceleração
        current_left_speed = self.left_motor.getVelocity()
        current_right_speed = self.right_motor.getVelocity()
        
        # Aplicar aceleração limitada (cálculos simplificados)
        delta_left = target_left_speed - current_left_speed
        delta_right = target_right_speed - current_right_speed
        
        left_speed = current_left_speed
        if delta_left > MAX_ACCEL:
            left_speed += MAX_ACCEL
        elif delta_left < -MAX_ACCEL:
            left_speed -= MAX_ACCEL
        else:
            left_speed = target_left_speed
            
        right_speed = current_right_speed
        if delta_right > MAX_ACCEL:
            right_speed += MAX_ACCEL
        elif delta_right < -MAX_ACCEL:
            right_speed -= MAX_ACCEL
        else:
            right_speed = target_right_speed
        
        # Aplicar velocidades e avançar simulação
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        self.supervisor.step(TIME_STEP)

        # Obter observação após ação
        obs = self._get_obs()
        pos_x, pos_y, normalized_yaw, dist, star_x, star_y = obs

        # --- Sistema de Recompensas Simplificado ---
        reward = STEP_PENALTY  # Penalização pequena por passo
        terminated = False
        truncated = False
        
        # 1. Calcular ângulo para a estrela (normalizado para [-1,1])
        target_angle = np.arctan2(star_y - pos_y, star_x - pos_x) / np.pi
        
        # 2. Calcular diferença de ângulo (valor entre 0 e 1, onde 0 = perfeito)
        angle_diff = abs(normalized_yaw - target_angle)
        if angle_diff > 1.0:
            angle_diff = 2.0 - angle_diff
        angle_diff_normalized = angle_diff / 1.0  # Normalizar para [0,1]
        
        # 3. Calcular progresso na distância
        distance_reduction = self.previous_distance - dist
        self.previous_distance = dist
        
        # 4. Componente principal: Recompensa Shaping (combinação de orientação e progresso)
        # - Orientação perfeita (0°) = 1.0, pior orientação (180°) = 0.0
        orientation_factor = 1.0 - angle_diff_normalized
        
        # Recompensa principal: Progresso ponderado pela qualidade da orientação
        # - Isso naturalmente incentiva primeiro orientar-se e depois mover-se
        main_reward = distance_reduction * 150.0 * (0.2 + 0.8 * orientation_factor)
        reward += main_reward
        
        # 5. Recompensa auxiliar por movimento eficiente (linha reta quando bem orientado)
        motor_diff = abs(left_speed - right_speed) / (2 * MAX_SPEED)
        if orientation_factor > 0.9:  # Bem orientado (< 18°)
            # Incentiva movimento em linha reta quando bem alinhado
            straight_reward = (1.0 - motor_diff) * 5.0 * (distance_reduction > 0)
            reward += straight_reward
            
            if distance_reduction > 0.001:  # Progresso significativo
                # Prints reduzidos apenas para feedback importante
                print(f"[+] Bom progresso: {distance_reduction:.4f}, ângulo: {angle_diff*180:.1f}°")
        
        # 6. Verificar conclusão do episódio
        # Alcançou o objetivo
        if dist < COLLISION_THRESHOLD:
            reward += GOAL_REWARD
            terminated = True
            print("[✓] Objetivo alcançado!")
        
        # Colisão com parede
        if abs(pos_x) >= 0.95 or abs(pos_y) >= 0.95:
            reward += WALL_PENALTY
            terminated = True
            print("[✗] Colisão com parede!")
        
        # Limite de tempo
        self.step_count += 1
        if self.step_count >= self.time_limit:
            truncated = True
            
            # Penalidade se terminou mais longe do que começou
            if dist > self.initial_distance:
                reward -= 50.0
                print(f"[!] Tempo esgotado - Terminou mais longe: {dist:.2f} > {self.initial_distance:.2f}")
            else:
                print(f"[!] Tempo esgotado - Progresso: {(1 - dist/self.initial_distance)*100:.1f}%")
        
        # Print reduzido a cada 10 passos para não sobrecarregar
        if self.step_count % 10 == 0:
            print(f"[i] Dist: {dist:.3f}, Reward: {reward:.2f}")
            
        return obs, reward, terminated, truncated, {}

