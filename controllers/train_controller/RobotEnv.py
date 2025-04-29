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
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.previous_distance = None
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

        # Cálculo otimizado de time_limit
        self.initial_distance = initial_distance
        half_max_linear_speed = 0.5 * MAX_SPEED * 0.02
        self.time_limit = int(4 * (initial_distance / half_max_linear_speed) * 1000 / TIME_STEP)
        self.step_count = 0
        
        self.log(f"[RESET] Robô: ({robot_x:.2f}, {robot_y:.2f}), Estrela: ({star_x:.2f}, {star_y:.2f}), "
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

    def compute_reward(self):
        obs = self._get_obs()
        pos_x, pos_y, normalized_yaw, dist, star_x, star_y = obs

        reward = 0.0
        reward += STEP_PENALTY
        self.log(f"Step penalty applied: {STEP_PENALTY}")

        dx, dy = star_x - pos_x, star_y - pos_y
        direction_to_goal = np.array([dx, dy])
        direction_to_goal_norm = np.sqrt(dx*dx + dy*dy) + 1e-8
        direction_to_goal = direction_to_goal / direction_to_goal_norm

        yaw = normalized_yaw * np.pi
        robot_direction = np.array([np.cos(yaw), np.sin(yaw)])
        direction_dot = np.dot(robot_direction, direction_to_goal)  # ∈ [-1, 1]

        angle_front_deg = np.rad2deg(np.arccos(np.clip(direction_dot, -1.0, 1.0)))
        angle_back_deg = 180.0 - angle_front_deg
        self.log(f"Angle from front: {angle_front_deg:.2f}°, from back: {angle_back_deg:.2f}°")

        distance_reduction = self.previous_distance - dist
        self.log(f"Distance reduced by: {distance_reduction:.4f}")
        self.previous_distance = dist

        movement_direction = np.sign(self.left_motor_speed + self.right_motor_speed)
        self.log(f"Direction dot: {direction_dot:.4f}, Movement direction: {movement_direction}")

        cos_8_deg = np.cos(np.deg2rad(8))

        # --- Recompensa por progresso com orientação precisa ---
        if distance_reduction > 0:
            if direction_dot >= cos_8_deg and movement_direction > 0:
                inc = distance_reduction * 150.0
                reward += inc
                self.log(f"Forward aligned reward: +{inc:.4f}")
            elif direction_dot <= -cos_8_deg and movement_direction < 0:
                inc = distance_reduction * 150.0
                reward += inc
                self.log(f"Reverse aligned reward: +{inc:.4f}")
            else:
                # Penalização com base na direção de movimento (não no menor ângulo)
                if movement_direction > 0:
                    angle_movement = angle_front_deg
                    side = "front"
                elif movement_direction < 0:
                    angle_movement = angle_back_deg
                    side = "back"
                else:
                    angle_movement = None
                    side = "none"

                if angle_movement is not None and angle_movement > 8.0:
                    misalignment_deg = angle_movement - 8.0
                    penalty = -0.02 * misalignment_deg
                    reward += penalty
                    self.log(f"Misaligned movement from {side} ({angle_movement:.2f}°). Penalty: {penalty:.4f}")
                elif angle_movement is not None:
                    self.log(f"Movement in correct direction ({side}) but not precisely aligned (<8°)")

        # --- Bônus por movimento reto ---
        motor_diff = abs(self.left_motor_speed - self.right_motor_speed) / MAX_SPEED
        if abs(direction_dot) > 0.99 and distance_reduction > 0:
            bonus = (1.0 - motor_diff) * 5.0
            reward += bonus
            self.log(f"Straight movement bonus: +{bonus:.4f}")

        # --- Bônus de proximidade ---
        proximity_bonus = (1.0 - dist / self.initial_distance) ** 2 * 30.0
        reward += proximity_bonus
        self.log(f"Proximity bonus: +{proximity_bonus:.4f}")

        # --- Penalização por não progresso e má orientação ---
        if distance_reduction <= 0 and abs(direction_dot) < 0.4:
            reward -= 0.1
            self.log("Penalty: No progress and poor alignment (-0.1)")

        # --- Objetivo atingido ---
        if dist < 0.2:
            time_bonus = (1.0 - self.step_count / self.time_limit) * 100.0
            reward += GOAL_REWARD + time_bonus
            self.log(f"Goal reached! Reward: {GOAL_REWARD + time_bonus:.4f}")
            self.goal_reached = True
            return reward

        # --- Penalização por colisão ---
        if self.touched_wall():
            reward += WALL_PENALTY
            self.log(f"Wall collision! Penalty: {WALL_PENALTY}")

        # --- Penalização por terminar longe demais ---
        if self.step_count >= self.time_limit and dist > self.initial_distance:
            reward -= 50.0
            self.log("Time exceeded and robot ended farther from goal. Penalty: -50.0")

        self.log(f"Total reward: {reward:.4f}")
        return reward


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
        
        # Aplicar limites de aceleração
        if abs(delta_left) > MAX_ACCEL:
            left_speed = current_left_speed + (MAX_ACCEL if delta_left > 0 else -MAX_ACCEL)
        else:
            left_speed = target_left_speed
            
        if abs(delta_right) > MAX_ACCEL:
            right_speed = current_right_speed + (MAX_ACCEL if delta_right > 0 else -MAX_ACCEL)
        else:
            right_speed = target_right_speed
        
        # Guardar velocidades para uso na função de recompensa
        self.left_motor_speed = left_speed
        self.right_motor_speed = right_speed
        
        # Aplicar velocidades e avançar simulação
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        self.supervisor.step(TIME_STEP)

        # Obter observação após ação
        obs = self._get_obs()
        _, _, _, dist, _, _ = obs
        
        # Atualizar contador de passos
        self.step_count += 1
        
        # Variáveis para controle de estado
        self.goal_reached = False
        
        # Calcular recompensa usando a função compute_reward
        reward = self.compute_reward()
        
        # Verificar terminação do episódio
        terminated = self.goal_reached or self.touched_wall()
        truncated = self.step_count >= self.time_limit
        
        # Logging
        if self.goal_reached:
            self.log("[✓] Objetivo alcançado!")
        elif self.touched_wall():
            self.log("[✗] Colisão com parede!")
        elif truncated:
            self.log(f"[!] Tempo esgotado - Progresso: {(1 - dist/self.initial_distance)*100:.1f}%")
        
        # Print reduzido a cada 10 passos para não sobrecarregar
        if self.step_count % 10 == 0:
            self.log(f"[i] Dist: {dist:.3f}, Reward: {reward:.2f}")
            
        return obs, reward, terminated, truncated, {}
        
    def touched_wall(self):
        # Verificar se o robô tocou nas paredes
        pos = self.robot_node.getField("translation").getSFVec3f()
        return abs(pos[0]) >= 0.95 or abs(pos[1]) >= 0.95
