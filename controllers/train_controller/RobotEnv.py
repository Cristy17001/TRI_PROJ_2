import gymnasium as gym
from gymnasium import spaces
import numpy as np
from controller import Supervisor

TIME_STEP = 64
MAX_SPEED = 6.28
COLLISION_THRESHOLD = 0.08
MIN_INITIAL_DISTANCE = 0.3  # distância mínima entre robô e estrela
SPAWN_RANGE = 0.9  # intervalo de spawn aleatório (só afeta X agora)

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
        
        # Espaço de ações (duas velocidades normalizadas entre -1 e 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Espaço de observações: posição x, posição y, yaw, distância para a estrela, pos_x estrela, pos_y estrela
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Spawn aleatório
        while True:
            robot_x = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            robot_y = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            star_x = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            star_y = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
            if np.linalg.norm([star_x - robot_x, star_y - robot_y]) >= MIN_INITIAL_DISTANCE:
                break

        self.robot_translation.setSFVec3f([robot_x, robot_y, 0.03])  # Z = altura = 0.03 para não colidir com o chão
        self.star_translation.setSFVec3f([star_x, star_y, 0.03])    # Estrela ligeiramente acima

        # Calculate straight-line distance and expected time at 50% max speed
        initial_distance = np.linalg.norm([star_x - robot_x, star_y - robot_y])
        # Assuming units are meters and radians/second
        # 50% of MAX_SPEED with approximate conversion to linear velocity
        half_max_linear_speed = 0.5 * MAX_SPEED * 0.02  # Approximate wheel radius as 0.02m
        # Time in simulation steps needed for straight line path
        self.time_limit = int(2 * (initial_distance / half_max_linear_speed) * 1000 / TIME_STEP)
        self.step_count = 0
        
        print(f"[RESET] Robô em (x={robot_x:.2f}, y={robot_y:.2f}), Estrela em (x={star_x:.2f}, y={star_y:.2f})")
        print(f"[RESET] Distância inicial: {initial_distance:.2f}, Limite de tempo: {self.time_limit} passos")
        
        for _ in range(10):
            self.supervisor.step(TIME_STEP)
        
        obs = self._get_obs()
        info = {}
        return obs, info

    
    def _get_obs(self):
        pos = self.gps.getValues()
        orientation = self.imu.getRollPitchYaw()[2]  # Get yaw from IMU
        # Normalize orientation to be between -1 and 1
        normalized_orientation = orientation / np.pi
        
        star_pos = self.star_node.getPosition()
        dist = np.linalg.norm([pos[0] - star_pos[0], pos[1] - star_pos[1]])
        
        # Include star position in the observation
        return np.array([pos[0], pos[1], normalized_orientation, dist, star_pos[0], star_pos[1]], dtype=np.float32)


    def step(self, action):
        # Action: [left_speed, right_speed] entre -1 e 1
        
        # Apply the constraint: motors must have same sign or one can be zero
        if (action[0] < 0 and action[1] > 0) or (action[0] > 0 and action[1] < 0):
            # If signs are different, set the smaller magnitude to zero
            if abs(action[0]) < abs(action[1]):
                action[0] = 0
            else:
                action[1] = 0
            print("[STEP] ⚠️ Action constrained: Motors must have same sign or one must be zero.")
        
        left_speed = action[0] * MAX_SPEED
        right_speed = action[1] * MAX_SPEED
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
        
        self.supervisor.step(TIME_STEP)

        obs = self._get_obs()
        pos_x, pos_y, normalized_yaw, dist, star_x, star_y = obs

        reward = -dist
        terminated = False
        truncated = False

        # Convert normalized yaw back to radians for printing
        yaw = normalized_yaw * np.pi
        print(f"[STEP] Posição (x={pos_x:.2f}, y={pos_y:.2f}), Yaw: {yaw:.2f}")
        print(f"[STEP] Distância até à estrela: {dist:.2f}")
        print(f"[STEP] Estrela em (x={star_x:.2f}, y={star_y:.2f})")

        # Remove the penalty for opposite signs since we're constraining the actions
        # if (action[0] * action[1]) < 0:  # Se as velocidades têm sinais opostos
        #     reward -= 0.05  # Penalidade pequena
        #     print("[STEP] ⚠️ Penalidade: Velocidades com orientações diferentes.")
                
        if dist < COLLISION_THRESHOLD:
            reward += 200.0
            terminated = True
            print("[STEP] ⭐ Alcançou a estrela!")

        if abs(pos_x) >= 0.95 or abs(pos_y) >= 0.95:
            reward -= 10.0
            terminated = True
            print("[STEP] 🚧 Colidiu com a parede!")

        print(f"[STEP] Reward final: {reward:.2f}\n")

        info = {}
        return obs, reward, terminated, truncated, info