from controller import Supervisor, GPS, InertialUnit
import numpy as np
import math
import random
import time

# Constantes
TIME_STEP = 64
MAX_SPEED = 6.28  # Velocidade máxima das rodas
MIN_DISTANCE = 0.02  # Distância mínima para considerar que chegou ao ponto
LINEAR_VELOCITY = 0.2  # Velocidade para interceptação
BLOCKING_DISTANCE = 0.1  # Distância de bloqueio ao e-puck
SAFETY_MARGIN = 0.08  # Margem de segurança para evitar colisões no spawn
COLLISION_THRESHOLD = 0.08  # Distância para considerar que alcançou a estrela

# Parâmetros de comportamento do rival
SPAWN_POSITIONS = [0.3, 0.4, 0.5, 0.6, 0.7]  # Posições relativas entre robô e estrela (0=estrela, 1=robô)
REPOSITION_INTERVAL = 10.0  # Intervalo mínimo entre reposicionamentos (segundos)
INTERCEPT_AGGRESSIVENESS = 0.7  # Quão agressivo é o comportamento de interceptação (0-1)
DYNAMIC_DIFFICULTY = True  # Ajusta dificuldade com base no desempenho do robô principal

class RivalController:
    def __init__(self):
        self.supervisor = Supervisor()
        
        # Inicializar atributos básicos ANTES de qualquer verificação
        self.state = "INIT"  # INIT, INTERCEPT, BLOCK, REPOSITION
        self.target_point = None
        self.last_update_time = 0
        self.last_position_change = 0
        self.last_position = [0, 0]
        self.stuck_counter = 0
        self.success_counter = 0
        self.failure_counter = 0
        self.difficulty_level = 0.5
        
        # Inicializa sensores
        self.gps = self.supervisor.getDevice("gps")
        self.imu = self.supervisor.getDevice("inertial unit")
        self.gps.enable(TIME_STEP)
        self.imu.enable(TIME_STEP)
        
        # Motores
        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Inicializar com velocidade zero para garantir
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Obter referência para os nós
        self.rival_node = self.supervisor.getSelf()
        self.rival_translation = self.rival_node.getField("translation")
        
        # Esperar um pouco para garantir que a simulação esteja totalmente inicializada
        for _ in range(5):
            self.supervisor.step(TIME_STEP)
        
        # Tentar obter referências aos nós
        self.epuck_node = None
        self.star_node = None
        
        # Tentar obter as referências várias vezes
        for _ in range(20):
            self.epuck_node = self.supervisor.getFromDef("robot1")
            self.star_node = self.supervisor.getFromDef("Star")
            
            if self.epuck_node and self.star_node:
                break
                
            print(f"Aguardando nós 'robot1' e 'Star'... ({_+1}/20)")
            self.supervisor.step(TIME_STEP)
        
        # Em vez de retornar, define a flag de inicialização
        self.initialized = self.epuck_node and self.star_node
        
        if not self.initialized:
            print("ERRO: Não foi possível encontrar 'robot1' ou 'Star'.")
        else:
            print("Controlador do robô rival inicializado!")
            # Posicionar o robô rival entre o robô principal e a estrela
            self.position_between_robot_and_star()
        
    def position_between_robot_and_star(self):
        """Posiciona o robô rival entre a estrela e o robô principal"""
        try:
            # Obter posições atuais
            star_pos = self.star_node.getPosition()
            epuck_pos = self.epuck_node.getPosition()
            
            # Verificar se as posições são válidas
            if not star_pos or not epuck_pos:
                raise ValueError("Posições inválidas recebidas")
            
            # Debug - imprimir posições para verificar
            print(f"Estrela: ({star_pos[0]:.3f}, {star_pos[1]:.3f})")
            print(f"E-puck: ({epuck_pos[0]:.3f}, {epuck_pos[1]:.3f})")
            
            # Calcular vetor da estrela para o robô principal
            vec_x = epuck_pos[0] - star_pos[0]
            vec_y = epuck_pos[1] - star_pos[1]
            
            # Calcular distância total
            total_distance = math.sqrt(vec_x**2 + vec_y**2)
            print(f"Distância entre estrela e E-puck: {total_distance:.3f}")
            
            # Verificar se a distância é suficiente para posicionar o rival
            if total_distance <= 2 * SAFETY_MARGIN:
                print("Aviso: Distância muito pequena entre estrela e robô")
                # Usar uma posição alternativa caso estejam muito próximos
                rival_pos = [0.0, 0.0, 0.0]
            else:
                # Normalizar o vetor
                if total_distance > 0:
                    vec_x /= total_distance
                    vec_y /= total_distance
                
                # Escolher uma posição relativa com base no nível de dificuldade
                if DYNAMIC_DIFFICULTY:
                    # Ajustar posição com base no desempenho do robô principal
                    # Mais próximo da estrela = mais difícil
                    position_factor = self._get_dynamic_position_factor()
                else:
                    # Escolher uma posição aleatória da lista
                    position_factor = random.choice(SPAWN_POSITIONS)
                
                # Calcular posição usando a equação da reta paramétrica: P = P1 + t*(P2-P1)
                # onde t é o fator de posição entre 0 (estrela) e 1 (robô)
                rival_x = star_pos[0] + position_factor * vec_x
                rival_y = star_pos[1] + position_factor * vec_y
                rival_z = 0.0  # Altura no chão da arena
                rival_pos = [rival_x, rival_y, rival_z]
            
            print(f"Posicionando rival em: ({rival_pos[0]:.3f}, {rival_pos[1]:.3f}, {rival_pos[2]:.3f})")
            
            # Definir posição do robô rival
            self.rival_translation.setSFVec3f(rival_pos)
            self.rival_node.resetPhysics()
            
            # Atualizar o tempo da última mudança de posição
            self.last_position_change = self.supervisor.getTime()
            
            print("Robô rival posicionado com sucesso!")
            
        except Exception as e:
            print(f"Erro ao posicionar robô rival: {e}")
            print("Usando posição padrão")
            self.rival_translation.setSFVec3f([0.0, -0.5, 0.0])
            self.rival_node.resetPhysics()
    
    def _get_dynamic_position_factor(self):
        """Determina o fator de posição com base no desempenho do robô principal"""
        # Calcular taxa de sucesso do robô rival
        total_interactions = max(1, self.success_counter + self.failure_counter)
        success_rate = self.success_counter / total_interactions
        
        # Ajustar dificuldade com base na taxa de sucesso
        # Se o rival está tendo muito sucesso (bloqueando bem), aumentar a dificuldade
        # Se o rival está falhando muito, diminuir a dificuldade
        if success_rate > 0.7:  # Rival está tendo muito sucesso
            self.difficulty_level = min(0.9, self.difficulty_level + 0.05)
        elif success_rate < 0.3:  # Rival está falhando muito
            self.difficulty_level = max(0.1, self.difficulty_level - 0.05)
        
        # Converter nível de dificuldade para posição relativa
        # Dificuldade alta = mais próximo da estrela
        # Dificuldade baixa = mais próximo do robô
        position_factor = 0.2 + (0.6 * (1 - self.difficulty_level))
        
        print(f"Dificuldade: {self.difficulty_level:.2f}, Posição relativa: {position_factor:.2f}")
        return position_factor
    
    def _find_interception_point(self):
        """Calcula um ponto de interceptação entre o E-puck e a estrela"""
        # Obter posições atuais
        epuck_pos = self.epuck_node.getPosition()
        star_pos = self.star_node.getPosition()
        rival_pos = self.gps.getValues()
        
        # Calcular vetor do e-puck para a estrela
        vec_x = star_pos[0] - epuck_pos[0]
        vec_y = star_pos[1] - epuck_pos[1]
        
        # Calcular distância
        distance = math.sqrt(vec_x**2 + vec_y**2)
        
        if distance < 0.001:  # Evitar divisão por zero
            return star_pos
            
        # Normalizar o vetor
        vec_x /= distance
        vec_y /= distance
        
        # Calcular velocidade do e-puck (para predição de movimento)
        epuck_vel = self.epuck_node.getVelocity()
        epuck_speed = math.sqrt(epuck_vel[0]**2 + epuck_vel[1]**2)
        
        # Ajustar ponto de interceptação com base na distância e velocidade
        if distance < 0.3:
            # Quando o e-puck está próximo da estrela, interceptar mais próximo da estrela
            intercept_distance = distance * 0.3  # 30% do caminho
        else:
            # Normalmente interceptar a 40-50% do caminho entre e-puck e estrela
            intercept_distance = distance * 0.45
        
        # Ponto base de interceptação
        intercept_x = epuck_pos[0] + vec_x * intercept_distance
        intercept_y = epuck_pos[1] + vec_y * intercept_distance
        
        # Ajuste preditivo baseado na velocidade do e-puck
        if epuck_speed > 0.05:  # Se o e-puck estiver se movendo significativamente
            # Normalizar vetor de velocidade
            vel_norm = math.sqrt(epuck_vel[0]**2 + epuck_vel[1]**2)
            if vel_norm > 0.001:
                vel_x = epuck_vel[0] / vel_norm
                vel_y = epuck_vel[1] / vel_norm
                
                # Ajustar ponto de interceptação na direção do movimento
                prediction_factor = min(0.3, epuck_speed * 0.5)  # Limitar o ajuste
                intercept_x += vel_x * prediction_factor
                intercept_y += vel_y * prediction_factor
        
        # Verificar se o rival está muito longe do ponto de interceptação
        rival_to_intercept = math.sqrt((rival_pos[0]-intercept_x)**2 + (rival_pos[1]-intercept_y)**2)
        
        # Se estiver muito longe e já passou tempo suficiente desde o último reposicionamento
        if rival_to_intercept > 0.8 and self.supervisor.getTime() - self.last_position_change > REPOSITION_INTERVAL:
            print("Rival muito longe do ponto de interceptação, considerando reposicionamento")
            self.state = "REPOSITION"
            return [intercept_x, intercept_y]  # Retorna o ponto mesmo assim
        
        return [intercept_x, intercept_y]
        
    def _compute_steering(self, target_x, target_y):
        """Calcula os valores de velocidade para as rodas para ir até o ponto alvo"""
        # Obter posição e orientação atual
        pos = self.gps.getValues()
        yaw = self.imu.getRollPitchYaw()[2]
        
        # Calcular ângulo para o alvo
        dx = target_x - pos[0]
        dy = target_y - pos[1]
        target_angle = math.atan2(dy, dx)
        
        # Calcular diferença de ângulo
        angle_diff = target_angle - yaw
        
        # Normalizar para [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2*math.pi
        while angle_diff < -math.pi:
            angle_diff += 2*math.pi
            
        # Calcular distância ao alvo
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Verificar se o robô está preso (não se move significativamente)
        if hasattr(self, 'last_position'):
            movement = math.sqrt((pos[0]-self.last_position[0])**2 + (pos[1]-self.last_position[1])**2)
            if movement < 0.005 and self.state != "INIT":  # Se moveu menos de 5mm
                self.stuck_counter += 1
                if self.stuck_counter > 20:  # Se ficou preso por muitos ciclos
                    print("Robô rival parece estar preso, tentando reposicionar")
                    self.state = "REPOSITION"
                    self.stuck_counter = 0
            else:
                self.stuck_counter = 0  # Resetar contador se estiver se movendo
        
        # Atualizar última posição
        self.last_position = [pos[0], pos[1]]
        
        # Estratégia de movimento baseada no ângulo
        if abs(angle_diff) > 0.3:  # ~17 graus
            # Girar no local com velocidade proporcional ao erro angular
            turn_factor = min(1.0, abs(angle_diff) / math.pi)
            turn_speed = MAX_SPEED * 0.8 * turn_factor
            
            if angle_diff > 0:
                left_speed = turn_speed
                right_speed = -turn_speed
            else:
                left_speed = -turn_speed
                right_speed = turn_speed
                
            # Adicionar pequena componente para frente apenas em ângulos pequenos
            if abs(angle_diff) < 0.5:
                tiny_forward = MAX_SPEED * 0.1
                left_speed += tiny_forward
                right_speed += tiny_forward
        else:
            # Já está bem orientado, usar controle proporcional
            # Ajustar velocidade com base na distância
            speed = min(LINEAR_VELOCITY * 2.0, distance * 2 + 0.1)
            base_speed = speed * 2/0.0701  # Converter velocidade linear para velocidade angular
            
            # Usar ganho proporcional para curvas
            k_p = 4.5
            left_speed = base_speed - k_p * angle_diff
            right_speed = base_speed + k_p * angle_diff
    
        # Limitar velocidades
        left_speed = min(max(left_speed, -MAX_SPEED), MAX_SPEED)
        right_speed = min(max(right_speed, -MAX_SPEED), MAX_SPEED)
        
        # Aplicar fator de agressividade
        left_speed *= INTERCEPT_AGGRESSIVENESS
        right_speed *= INTERCEPT_AGGRESSIVENESS
        
        # Debug
        print(f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}), Alvo: ({target_x:.2f}, {target_y:.2f})")
        print(f"Ângulo atual: {yaw:.2f}, Alvo: {target_angle:.2f}, Diff: {angle_diff:.2f}")
        print(f"Velocidades: L={left_speed:.2f}, R={right_speed:.2f}, Estado: {self.state}")
        
        return left_speed, right_speed, distance
    
    def _should_update_target(self):
        """Verifica se deve atualizar o ponto alvo com base no tempo"""
        current_time = self.supervisor.getTime()
        # Atualizar frequentemente (10 vezes por segundo)
        if current_time - self.last_update_time > 0.1:
            self.last_update_time = current_time
            return True
        return False
    
    def _check_interception_success(self):
        """Verifica se o robô rival conseguiu interceptar o robô principal"""
        # Obter posições atuais
        epuck_pos = self.epuck_node.getPosition()
        star_pos = self.star_node.getPosition()
        rival_pos = self.gps.getValues()
        
        # Calcular distâncias
        rival_to_epuck = math.sqrt((rival_pos[0]-epuck_pos[0])**2 + (rival_pos[1]-epuck_pos[1])**2)
        epuck_to_star = math.sqrt((epuck_pos[0]-star_pos[0])**2 + (epuck_pos[1]-star_pos[1])**2)
        rival_to_star = math.sqrt((rival_pos[0]-star_pos[0])**2 + (rival_pos[1]-star_pos[1])**2)
        
        # Verificar se o rival está entre o robô principal e a estrela
        # Calculamos se a soma das distâncias (rival->epuck + rival->star) é aproximadamente
        # igual à distância direta epuck->star
        direct_path = epuck_to_star
        path_through_rival = rival_to_epuck + rival_to_star
        
        # Se o rival está próximo do caminho direto (com uma margem de erro)
        is_blocking = abs(path_through_rival - direct_path) < 0.1
        
        # Se o rival está próximo o suficiente do robô principal
        is_close_to_epuck = rival_to_epuck < BLOCKING_DISTANCE * 1.5
        
        if is_blocking and is_close_to_epuck:
            self.success_counter += 1
            print(f"Interceptação bem-sucedida! Sucessos: {self.success_counter}")
            return True
        
        return False
        
    def run(self):
        print("Robô rival iniciando comportamento de interceptação")
        
        if not hasattr(self, 'initialized') or not self.initialized:
            print("Não é possível executar o controlador - inicialização falhou")
            # Movimento simples para mostrar que está funcionando
            for _ in range(50):
                self.left_motor.setVelocity(2.0)
                self.right_motor.setVelocity(-2.0)
                self.supervisor.step(TIME_STEP)
            return
        
        # Posicionamento inicial apenas - não é teleporte durante o jogo
        self.position_between_robot_and_star()
        
        # Iniciar no estado INIT
        self.state = "INIT"
        
        # Loop principal
        while self.supervisor.step(TIME_STEP) != -1:
            if self.state == "INIT":
                # Aguardar sensores estarem prontos
                if self.supervisor.getTime() < 0.5:
                    continue
                
                # Calcular ponto de interceptação inicial
                self.target_point = self._find_interception_point()
                print(f"Ponto de interceptação inicial: {self.target_point}")
                
                self.state = "INTERCEPT"
                self.last_update_time = self.supervisor.getTime()
                
            elif self.state == "INTERCEPT":
                # Verificar se o robô principal alcançou a estrela
                epuck_pos = self.epuck_node.getPosition()
                star_pos = self.star_node.getPosition()
                epuck_to_star = math.sqrt((epuck_pos[0]-star_pos[0])**2 + (epuck_pos[1]-star_pos[1])**2)
                
                if epuck_to_star < COLLISION_THRESHOLD:
                    # O robô principal alcançou a estrela - falha na interceptação
                    self.failure_counter += 1
                    print(f"Robô principal alcançou a estrela! Falhas: {self.failure_counter}")
                    
                    # Não teleporta mais - apenas recalcula o ponto de interceptação
                    self.target_point = self._find_interception_point()
                    continue
                
                # Verificar se o rival conseguiu interceptar o robô principal
                if self._check_interception_success():
                    # Mudar para modo de perseguição quando interceptar com sucesso
                    self.state = "CHASE"
                    print("Interceptação bem-sucedida! Iniciando perseguição!")
                    continue
                
                # Atualizar o alvo periodicamente 
                if self._should_update_target():
                    # Calcular novo ponto de interceptação
                    self.target_point = self._find_interception_point()
                
                # Ir para o ponto alvo
                left_speed, right_speed, _ = self._compute_steering(
                    self.target_point[0], self.target_point[1])
                
                # Aplicar velocidades aos motores
                self.left_motor.setVelocity(left_speed)
                self.right_motor.setVelocity(right_speed)
                
            elif self.state == "CHASE":
                # Modo de perseguição - ir diretamente em direção ao robô 1
                epuck_pos = self.epuck_node.getPosition()
                rival_pos = self.gps.getValues()
                
                # Calcular distância ao E-puck
                rival_to_epuck = math.sqrt((rival_pos[0]-epuck_pos[0])**2 + (rival_pos[1]-epuck_pos[1])**2)
                
                # Definir o alvo como a posição do E-puck
                self.target_point = [epuck_pos[0], epuck_pos[1]]
                
                # Verificar se está muito longe para voltar a interceptar
                if rival_to_epuck > 0.4:
                    print("E-puck fugiu! Voltando ao modo de interceptação")
                    self.state = "INTERCEPT"
                    self.target_point = self._find_interception_point()
                    continue
                
                # Ir diretamente para o robô 1 com velocidade maior (mais agressivo)
                left_speed, right_speed, _ = self._compute_steering(
                    epuck_pos[0], epuck_pos[1])
                    
                # Aumentar velocidade durante perseguição
                left_speed *= 1.2
                right_speed *= 1.2
                
                # Limitar velocidades
                left_speed = min(max(left_speed, -MAX_SPEED), MAX_SPEED)
                right_speed = min(max(right_speed, -MAX_SPEED), MAX_SPEED)
                
                # Aplicar velocidades aos motores
                self.left_motor.setVelocity(left_speed)
                self.right_motor.setVelocity(right_speed)

if __name__ == "__main__":
    print("Iniciando o controlador do robô rival")
    controller = RivalController()
    controller.run()