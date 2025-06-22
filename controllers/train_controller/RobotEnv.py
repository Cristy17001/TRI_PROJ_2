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
GOAL_REWARD = 5000.0
WALL_PENALTY = -500.0
ROBOT_COLLISION_PENALTY = -100.0  # Penalização por colidir com o robô rival

MAX_LINEAR_SPEED = 0.05      # Velocidade máxima do robô em m/s
MAX_ANGULAR_SPEED = 6.28     # Velocidade máxima de rotação (rad/s)
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


    def _calculate_progress_reward(self, distance_reduction):
        """Recompensa baseada em progresso em direção ao objetivo."""
        if distance_reduction > 0.001:
            self.log(f"[✓] Progresso: -{distance_reduction:.4f}")
            return distance_reduction * 10000.0
        elif abs(distance_reduction) <= 0.001:
            self.log("[!] Estagnado - sem progresso")
            return -3.0
        else:
            self.log(f"[✗] Afastou-se: +{abs(distance_reduction):.4f}")
            return -5.0

    def _calculate_wheel_penalty(self, left_speed, right_speed):
        """Penalização por diferença entre rodas (evita rotação exagerada sem avanço)."""
        wheel_diff = abs(left_speed - right_speed)
        if wheel_diff > 0.5 * MAX_SPEED:
            penalty = -wheel_diff * 20.0
            self.log(f"[!] Penalização por rotação: {penalty:.2f}")
            return penalty
        return 0.0

    def _calculate_rival_proximity_penalty(self, pos_x, pos_y, rival_x, rival_y):
        """Penalização contínua por proximidade ao robô rival."""
        dx, dy = pos_x - rival_x, pos_y - rival_y
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.001:
            dist = 0.001  # evita divisão por zero

        # Penalização exponencial: forte quando está perto
        penalty = -np.exp(-6 * dist) * 4.0
        self.log(f"[!] Penalização por proximidade ao rival ({dist:.3f} m): {penalty:.2f}")
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

        reward = 0
        terminated = False
        truncated = False

        # Cálculo do progresso
        distance_reduction = self.previous_distance - dist
        self.previous_distance = dist

        # Recompensas / Penalizações organizadas por categoria
        reward += self._calculate_progress_reward(distance_reduction)
        reward += self._calculate_wheel_penalty(left_speed, right_speed)
        reward += self._calculate_rival_proximity_penalty(pos_x, pos_y, rival_x, rival_y)

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

        # Check de colisão com rival
        if self.rival_robot_node:
            rival_dist = np.linalg.norm([pos_x - rival_x, pos_y - rival_y])
            if rival_dist < 0.085:
                reward += ROBOT_COLLISION_PENALTY
                terminated = True
                self.log("[✗] Colisão com robô rival!")

        # Limite de tempo
        self.step_count += 1
        if self.step_count >= self.time_limit:
            truncated = True
            if dist > self.initial_distance:
                reward -= 10.0
                self.log(f"[!] Tempo esgotado - Terminou mais longe: {dist:.2f} > {self.initial_distance:.2f}")
            else:
                self.log(f"[!] Tempo esgotado - Progresso: {(1 - dist / self.initial_distance) * 100:.1f}%")

        if self.step_count % 10 == 0:
            self.log(f"[i] Dist: {dist:.3f}, Reward: {reward:.2f}")

        return obs, reward, terminated, truncated, {}


