import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import Supervisor

TIME_STEP = 64
MAX_SPEED = 6.28
COLLISION_THRESHOLD = 0.08
MIN_INITIAL_DISTANCE = 0.3
SPAWN_RANGE = 0.9

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
        self.prev_dist = None  # To track starting distance from target
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        while True:
            robot_x = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            robot_y = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            star_x = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            star_y = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            if np.linalg.norm([star_x - robot_x, star_y - robot_y]) >= MIN_INITIAL_DISTANCE:
                break

        self.robot_translation.setSFVec3f([robot_x, robot_y, 0])
        self.robot_node.resetPhysics()

        self.star_translation.setSFVec3f([star_x, star_y, 0.03])
        self.star_node.resetPhysics()

        initial_distance = np.linalg.norm([star_x - robot_x, star_y - robot_y])
        half_max_linear_speed = 0.5 * MAX_SPEED * 0.02
        self.time_limit = int(2 * (initial_distance / half_max_linear_speed) * 1000 / TIME_STEP)
        self.step_count = 0
        
        if self.verbose:
            print(f"[RESET] Robô em (x={robot_x:.2f}, y={robot_y:.2f}), Estrela em (x={star_x:.2f}, y={star_y:.2f})")
            print(f"[RESET] Distância inicial: {initial_distance:.2f}, Limite de tempo: {self.time_limit} passos")
        
        for _ in range(10):
            self.supervisor.step(TIME_STEP)
        
        obs = self._get_obs()
        pos_x, pos_y, normalized_yaw, dist, star_x, star_y = obs
        self.prev_dist = dist
        info = {}
        return obs, info

    
    def _get_obs(self):
        pos = self.gps.getValues()
        orientation = self.imu.getRollPitchYaw()[2]
        normalized_orientation = orientation / np.pi
        
        star_pos = self.star_node.getPosition()
        dist = np.linalg.norm([pos[0] - star_pos[0], pos[1] - star_pos[1]])
        
        return np.array([pos[0], pos[1], normalized_orientation, dist, star_pos[0], star_pos[1]], dtype=np.float32)

    def _normalize_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        left, right = action

        if left * right < 0:
            if abs(left) < abs(right):
                left = 0.0
            else:
                right = 0.0
        return np.array([left, right], dtype=np.float32)

    def step(self, action):
        action = self._normalize_action(action)

        # Apply motor velocities
        left_speed = action[0] * MAX_SPEED
        right_speed = action[1] * MAX_SPEED
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        
        self.supervisor.step(TIME_STEP)

        # Get updated observations
        obs = self._get_obs()
        pos_x, pos_y, normalized_yaw, dist, star_x, star_y = obs
        yaw = normalized_yaw * np.pi
        
        # Initialize reward
        reward = 0.0
        terminated = False
        truncated = False
        
        # --- Distance improvement reward ---
        if self.prev_dist is None:
            self.prev_dist = dist
        delta_dist = self.prev_dist - dist
        distance_reward = delta_dist * 100  # Scale factor
        reward += distance_reward
        self.prev_dist = dist  # Update for next step
        
        # --- Proximity bonus ---
        proximity_bonus = (1.0 / (dist + 1e-5)) * 2.0
        reward += proximity_bonus
        
        # --- Alignment bonus ---
        vec_to_star = np.array([star_x - pos_x, star_y - pos_y])
        vec_to_star_norm = vec_to_star / (np.linalg.norm(vec_to_star) + 1e-8)
        robot_dir = np.array([np.cos(yaw), np.sin(yaw)])
        alignment = np.dot(robot_dir, vec_to_star_norm)
        
        alignment_bonus = 0.0
        if alignment > 0.7:
            alignment_bonus = 5.0 * alignment
        elif alignment < -0.7:
            alignment_bonus = 2.5 * abs(alignment)
        reward += alignment_bonus
        
        # --- Rotation penalty ---
        rotation_penalty = 0.0
        if alignment > 0.9:
            # If already well-aligned, penalize unnecessary rotation
            rotation_amount = abs(action[0] - action[1])
            if rotation_amount > 0.2:
                rotation_penalty = -5.0 * rotation_amount
        reward += rotation_penalty
        
        # --- Terminal conditions ---
        if dist < COLLISION_THRESHOLD:
            reward += 200.0
            terminated = True
            if self.verbose:
                print("[STEP] ⭐ Alcançou a estrela!")

        if abs(pos_x) >= 0.95 or abs(pos_y) >= 0.95:
            reward -= 100.0
            terminated = True
            if self.verbose:
                print("[STEP] 🚧 Colidiu com a parede!")

        # --- Debug logging ---
        if self.verbose:
            print(f"[STEP] Posição (x={pos_x:.2f}, y={pos_y:.2f}), Yaw: {yaw:.2f}")
            print(f"[STEP] Distância até à estrela: {dist:.2f}")
            print(f"[STEP] Estrela em (x={star_x:.2f}, y={star_y:.2f})")
            print(f"[STEP] Alinhamento (cosθ): {alignment:.2f}")
            print(f"[STEP] Reward final: {reward:.2f}\n")

        info = {}
        return obs, reward, terminated, truncated, info


