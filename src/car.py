import numpy as np
import math
import pygame
from typing import List, Tuple, Optional
from neural_network import NeuralNetwork


class Car:
    def __init__(self, x: float, y: float, angle: float = 0):
        # Posição e movimento
        self.x = x
        self.y = y
        self.angle = math.radians(angle)
        self.speed = 0.0
        self.max_speed = 100.0 
        self.acceleration = 5.0 
        self.friction = 1.00
        self.turn_speed = 1.00
        
        # Dimensões do carro
        self.width = 20
        self.height = 10
        
        # Rede neural
        self.neural_network = NeuralNetwork()
        
        # Sensores
        self.sensor_length = 100
        self.sensor_angles = [-60, -30, 0, 30, 60]
        self.sensor_readings = [0.0] * 5
        
        # Fitness e status
        self.fitness = 0.0
        self.distance_traveled = 0.0
        self.is_alive = True
        self.collision_penalty = 0
        
        # Tracking posição anterior
        self.prev_x = x
        self.prev_y = y
        
        # Tracking movimento para evitar carros parados
        self.time_alive = 0.0
        self.time_since_last_movement = 0.0
        self.last_distance = 0.0
        self.stuck_threshold = 3.0 
        
        # Lap tracking
        self.laps_completed = 0
        self.crossed_finish_line = False
        self.last_lap_distance = 0.0  # Distância no início da volta atual
        self.spawn_x = x  # Posição inicial X
        self.spawn_y = y  # Posição inicial Y
        
        # Tracking melhor volta
        self.avg_speed = 0.0
        self.speed_samples = 0
        self.lap_start_time = 0.0
        self.best_lap_time = float('inf')
        self.last_lap_time = 0.0
        
        # Pontos de posição (bônus por posição na corrida)
        self.position_points = 0 
        
        # Neural network outputs
        self.steering_output = 0.0
        self.speed_output = 0.0
        self.network_activations = []
    
    def update(self, track_walls: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
               finish_line: Tuple[Tuple[float, float], Tuple[float, float]] = None, dt: float = 1.0,
               lap_completion_callback=None):
        if not self.is_alive:
            return
        
        self.time_alive += dt / 60.0
        self.update_sensors(track_walls)
        
        # Saídas da rede neural
        normalized_sensors = [reading / self.sensor_length for reading in self.sensor_readings]
        outputs, self.network_activations = self.neural_network.forward(np.array(normalized_sensors))
        
        # Extrai comandos de direção e velocidade
        self.steering_output = outputs[0]
        self.speed_output = outputs[1]
        
        # Aplica direção
        self.angle += self.steering_output * self.turn_speed * dt
        
        normalized_output = (self.speed_output + 1.0) / 2.0
        target_speed = (0.3 + normalized_output * 0.7) * self.max_speed  # Range: 30%-100%
        
        # Acelera suavemente até a velocidade alvo
        if self.speed < target_speed:
            self.speed += self.acceleration * dt
        else:
            self.speed += (target_speed - self.speed) * 0.1  # Desacelera suavemente
        
        # Aplica fricção leve
        self.speed *= self.friction
        
        # Limita velocidade
        self.speed = max(min(self.speed, self.max_speed), self.max_speed * 0.25)
        
        # Atualiza posição do carro
        self.prev_x = self.x
        self.prev_y = self.y
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt
        
        # Calcula distância percorrida neste frame
        distance_this_frame = math.sqrt((self.x - self.prev_x)**2 + (self.y - self.prev_y)**2)
        self.distance_traveled += distance_this_frame
        
        # Check se está parado (movimento < threshold)
        if distance_this_frame < 0.1:  # Praticamente parado
            self.time_since_last_movement += dt / 60.0
        else:
            self.time_since_last_movement = 0.0
        
        # Mata carro se ficar parado por muito tempo
        if self.time_since_last_movement > self.stuck_threshold:
            self.is_alive = False
            self.collision_penalty = 500
        
        # Check se cruzou a linha de chegada
        if finish_line:
            if self.check_finish_line(finish_line):
                if lap_completion_callback:
                    lap_completion_callback(self)
        
        # Check collisions com paredes
        if self.check_collision(track_walls):
            self.is_alive = False
            self.collision_penalty = 200
        
        # Update fitness
        self.update_fitness()

    def update_sensors(self, track_walls: List[Tuple[Tuple[float, float], Tuple[float, float]]]):
        self.sensor_readings = []
        
        for i, sensor_angle_offset in enumerate(self.sensor_angles):
            sensor_angle = self.angle + math.radians(sensor_angle_offset)
            
            # Lança raio a partir do centro do carro
            ray_start = (self.x, self.y)
            ray_end = (
                self.x + math.cos(sensor_angle) * self.sensor_length,
                self.y + math.sin(sensor_angle) * self.sensor_length
            )
            
            # Encontra a interseção mais próxima
            min_distance = self.sensor_length
            for wall in track_walls:
                intersection = self.line_intersection(ray_start, ray_end, wall[0], wall[1])
                if intersection:
                    # Distancia euclidiana até a interseção
                    distance = math.sqrt((intersection[0] - self.x)**2 + (intersection[1] - self.y)**2)
                    min_distance = min(min_distance, distance)
            
            self.sensor_readings.append(min_distance)
    
    def line_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                         p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)
        
        return None
    
    def check_collision(self, track_walls: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
        corners = self.get_car_corners() # Obtém os cantos do carro (extremidades)
        
        for i in range(4):
            edge_start = corners[i]
            edge_end = corners[(i + 1) % 4]
            
            for wall in track_walls:
                if self.line_intersection(edge_start, edge_end, wall[0], wall[1]):
                    return True
        
        return False
    
    def get_car_corners(self) -> List[Tuple[float, float]]:
        half_width = self.width / 2
        half_height = self.height / 2
        
        local_corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]
        
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        
        world_corners = []
        for lx, ly in local_corners:
            wx = self.x + lx * cos_angle - ly * sin_angle
            wy = self.y + lx * sin_angle + ly * cos_angle
            world_corners.append((wx, wy))
        
        return world_corners
    
    def check_finish_line(self, finish_line: Tuple[Tuple[float, float], Tuple[float, float]]):
        car_line = ((self.prev_x, self.prev_y), (self.x, self.y))
        
        # Checa interseção
        if self.line_intersection(car_line[0], car_line[1], finish_line[0], finish_line[1]):
            distance_since_last_lap = self.distance_traveled - self.last_lap_distance
            
            # Sistema de detecção de volta baseado em DISTÂNCIA PERCORRIDA
            if not self.crossed_finish_line and distance_since_last_lap > 400:
                # Completou uma volta!
                self.laps_completed += 1
                self.crossed_finish_line = True
                self.last_lap_distance = self.distance_traveled
                
                # Calcula tempo da volta
                if self.lap_start_time > 0:
                    self.last_lap_time = self.time_alive - self.lap_start_time
                    if self.last_lap_time < self.best_lap_time:
                        self.best_lap_time = self.last_lap_time
                
                self.lap_start_time = self.time_alive  # Inicia timer da próxima volta
                return True
            elif not self.crossed_finish_line:
                # Primeira vez cruzando (no spawn)
                self.crossed_finish_line = True
                self.lap_start_time = self.time_alive  # Inicia timer da primeira volta
        else:
            # Se percorreu bastante, permite detectar próximo cruzamento
            distance_since_last_lap = self.distance_traveled - self.last_lap_distance
            if distance_since_last_lap > 200:
                self.crossed_finish_line = False
        
        return False
    
    def update_fitness(self):
        
        # 1. LAPS COMPLETED - Recompensa MASSIVA por completar voltas
        lap_bonus = self.laps_completed * 10000.0

        # 2. DISTÂNCIA PERCORRIDA - Incentiva progresso
        progress_fitness = self.distance_traveled * 2.0

        # 3. VELOCIDADE MÉDIA - Incentiva carros rápidos!
        if self.time_alive > 0:
            self.avg_speed = self.distance_traveled / self.time_alive
            speed_bonus = self.avg_speed * 100.0  # Recompensa de alta velocidade
        else:
            speed_bonus = 0.0
        
        # 4. MELHOR VOLTA - Recompensa ENORME por voltas rápidas!
        if self.best_lap_time < float('inf'):
            # Volta de 10s = 1000 pontos, 5s = 2000 pontos
            lap_time_bonus = (100.0 / self.best_lap_time) * 1000.0
        else:
            lap_time_bonus = 0.0
        
        # 5. PONTOS DE POSIÇÃO - Recompensa por ordem de chegada!
        position_bonus = self.position_points
        
        # 6. Tempo de sobrevivência (incentiva evitar colisões)
        time_bonus = self.time_alive * 5.0
        
        # 7. Penalidade por ficar parado ou colidir
        death_penalty = self.collision_penalty
        
        # Fitness final: Prioriza voltas completas + velocidade + tempo + POSIÇÃO
        self.fitness = (
            lap_bonus +           # 10000 por volta
            progress_fitness +    # 2x distância
            speed_bonus +         # 100x velocidade média
            lap_time_bonus +      # Até 2000+ por volta rápida
            position_bonus +      # NOVO: 100-500 por posição de chegada
            time_bonus -          # 5x tempo vivo
            death_penalty         # -200 ou -500
        )
        
        # Garante que fitness nunca seja negativo
        self.fitness = max(0, self.fitness)
    
    def get_sensor_endpoints(self) -> List[Tuple[float, float]]:
        endpoints = []
        for i, sensor_angle_offset in enumerate(self.sensor_angles):
            sensor_angle = self.angle + math.radians(sensor_angle_offset)
            distance = self.sensor_readings[i] if i < len(self.sensor_readings) else self.sensor_length
            
            end_x = self.x + math.cos(sensor_angle) * distance
            end_y = self.y + math.sin(sensor_angle) * distance
            endpoints.append((end_x, end_y))
        
        return endpoints
    
    def reset(self, x: float, y: float, angle: float = 0):
        self.x = x
        self.y = y
        self.angle = math.radians(angle)
        self.speed = 0.0
        self.fitness = 0.0
        self.distance_traveled = 0.0
        self.is_alive = True
        self.collision_penalty = 0
        self.sensor_readings = [0.0] * 5
        self.prev_x = x
        self.prev_y = y
        self.time_alive = 0.0
        self.time_since_last_movement = 0.0
        self.last_distance = 0.0