from controller import Supervisor, GPS, InertialUnit
import math
import random

TIME_STEP = 64
MAX_SPEED = 6.28
COLLISION_THRESHOLD = 0.08
MIN_INITIAL_DISTANCE = 0.1  # distância mínima entre robô e estrela
SPAWN_RANGE = 1  # intervalo de spawn aleatório (só afeta X agora)

def distance(pos1, pos2):
    # Consider X and Y (ignore Z)
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def compute_motor_speeds_zigzag(current_pos, current_angle, target_pos, max_speed, step_count, freq=0.05, amplitude=2):
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]

    # Ângulo objetivo em linha reta
    target_angle = math.atan2(dy, dx)

    # Aplica perturbação senoidal para criar zig-zag
    zigzag_angle = amplitude * math.sin(freq * step_count)
    target_angle += zigzag_angle

    # Corrige a diferença de ângulo
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
    
    main_robot_node = supervisor.getFromDef("robot1")
    robot_x = 0.0
    star_x = 0.0
    step_counter = 0
    

    if main_robot_node and star_translation:
        main_robot_pos = main_robot_node.getPosition()
        star_pos = star_translation.getSFVec3f()

    while supervisor.step(TIME_STEP) != -1:
        main_robot_pos = main_robot_node.getPosition() if main_robot_node else None
        pos = gps.getValues()
        gps_2d = [pos[0], pos[1]]
        orientation = inertial_unit.getRollPitchYaw()[2]

        if main_robot_node:
            target_pos = [main_robot_pos[0], main_robot_pos[1]]

            dist = distance(gps_2d, target_pos)
            # print("main_robot_dist:", dist)

            # For compute_motor_speeds, you may want to keep using only X if that's your logic
            l_speed, r_speed = compute_motor_speeds_zigzag(gps_2d, orientation, target_pos, MAX_SPEED * 0.8, step_counter)
            step_counter += 1
            left_motor.setVelocity(l_speed)
            right_motor.setVelocity(r_speed)