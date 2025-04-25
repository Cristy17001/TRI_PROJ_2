from controller import Supervisor, GPS, InertialUnit
import math
import random

TIME_STEP = 64
MAX_SPEED = 6.28
COLLISION_THRESHOLD = 0.08
MIN_INITIAL_DISTANCE = 0.3  # distância mínima entre robô e estrela
SPAWN_RANGE = 1  # intervalo de spawn aleatório (só afeta X agora)

def distance(pos1, pos2):
    return abs(pos1[0] - pos2[0])  # Só considera X

def compute_motor_speeds(current_pos, current_angle, target_pos, max_speed):
    dx = target_pos[0] - current_pos[0]

    # Ângulo objetivo é 0 se dx > 0, pi caso contrário (para andar só no eixo X)
    target_angle = 0.0 if dx >= 0 else math.pi
    angle_diff = target_angle - current_angle
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    kp = 3.0
    turn_speed = kp * angle_diff

    left_speed = max_speed - turn_speed
    right_speed = max_speed + turn_speed

    left_speed = max(min(left_speed, max_speed), -max_speed)
    right_speed = max(min(right_speed, max_speed), -max_speed)

    return left_speed, right_speed

if __name__ == "__main__":
    supervisor = Supervisor()
    
    left_motor = supervisor.getDevice("left wheel motor")
    right_motor = supervisor.getDevice("right wheel motor")
    gps = supervisor.getDevice("gps")
    inertial_unit = supervisor.getDevice("inertial unit")

    left_motor.setPosition(float("inf"))
    right_motor.setPosition(float("inf"))

    gps.enable(TIME_STEP)
    inertial_unit.enable(TIME_STEP)

    robot_node = supervisor.getSelf()
    robot_translation = robot_node.getField("translation")
    
    star_node = supervisor.getFromDef("Star")
    star_translation = star_node.getField("translation") if star_node else None

    # Geração aleatória no eixo X, Z fixo a 0
    while True:
        robot_x = random.uniform(-SPAWN_RANGE, SPAWN_RANGE)
        star_x = random.uniform(-SPAWN_RANGE, SPAWN_RANGE)

        if abs(star_x - robot_x) >= MIN_INITIAL_DISTANCE:
            break

    # Atualizar posições (Z = 0.0)
    robot_translation.setSFVec3f([robot_x, 0.0, 0.0])
    if star_translation:
        star_translation.setSFVec3f([star_x, 0.0, 0.03])

    # Loop principal
    while supervisor.step(TIME_STEP) != -1:
        pos = gps.getValues()
        gps_1d = [pos[0]]  # Só X
        orientation = inertial_unit.getRollPitchYaw()[2]

        if star_node:
            target_pos_raw = star_node.getPosition()
            target_pos = [target_pos_raw[0]]  # Só X

            dist = distance(gps_1d, target_pos)
            print(f"Dist: {dist:.2f}  |  Yaw: {orientation:.2f}")

            if dist < COLLISION_THRESHOLD:
                print("🚨 Reached Star!")
                left_motor.setVelocity(0)
                right_motor.setVelocity(0)
                break

            l_speed, r_speed = compute_motor_speeds(gps_1d, orientation, target_pos, MAX_SPEED * 0.5)
            left_motor.setVelocity(l_speed)
            right_motor.setVelocity(r_speed)

# Observações/Dados para fornecer
# -» Posição atual (X, Z) — por GPS (ignora Y/altura).
# -» Orientação atual — yaw do InertialUnit
# -» Posição da estrela — também X, Z.
# -» Distância à estrela
# -» Velocidade de cada roda do robo

# Ações/Outputs
# -» [left_motor_speed, right_motor_speed]

# Reward (a melhorar)
# -» +1 quando alcançar a estrela
# -» -distance_to_star — encoraja a aproximação
# -» -1 se bater em obstáculo (Quando tiver o outro robo a impedir)
# -» Efficiency (o quão longe se encontra do caminho em linha reta)



