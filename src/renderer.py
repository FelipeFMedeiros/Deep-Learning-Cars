import pygame
import math
import numpy as np
from typing import List, Tuple, Optional
from car import Car
from neural_network import NeuralNetwork


class Renderer:
    """
    Renderizador baseado em Pygame para a simulação de carros.
    Gerencia toda a visualização incluindo carros, pistas, sensores e redes neurais.
    """
    
    def __init__(self, width: int = 1200, height: int = 800):
        pygame.init()
        
        self.width = width
        self.height = height
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("AI Cars - Genetic Algorithm Evolution")
        
        # Colors
        self.colors = {
            'background': (40, 44, 52),
            'track_wall': (200, 200, 200),
            'checkpoint': (255, 255, 0),
            'car_best': (0, 255, 0),
            'car_second': (255, 200, 100),
            'car_normal': (100, 150, 255),
            'car_dead': (100, 100, 100),
            'sensor': (255, 255, 255),
            'sensor_hit': (255, 0, 0),
            'text': (255, 255, 255),
            'ui_bg': (60, 63, 65),
            'ui_border': (100, 100, 100),
            'neural_positive': (0, 255, 0),
            'neural_negative': (255, 0, 0),
            'neural_node': (200, 200, 200),
            'neural_active': (255, 255, 0)
        }
        
        # Fonts
        self.font_large = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        
        # Camera
        self.camera_x = 0
        self.camera_y = 0
        self.camera_target = None
        self.camera_smooth = 0.1
        
        # UI elements
        self.show_sensors = True
        self.show_neural_network = True
        self.ui_panel_width = 300
        
        
    # Atualiza a posição da câmera para seguir o carro alvo
    def update_camera(self, target_car: Optional[Car]):
        if target_car and target_car.is_alive:
            target_x = target_car.x - self.width // 2
            target_y = target_car.y - self.height // 2
            
            self.camera_x += (target_x - self.camera_x) * self.camera_smooth
            self.camera_y += (target_y - self.camera_y) * self.camera_smooth
    
    # Converte coordenadas do mundo para coordenadas da tela
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        screen_x = int(x - self.camera_x)
        screen_y = int(y - self.camera_y)
        return screen_x, screen_y
    
    # Renderiza as paredes da pista e a linha de chegada
    def render_track(self, walls: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                    finish_line: Tuple[Tuple[float, float], Tuple[float, float]] = None):
        # Renderiza paredes
        for wall in walls:
            start_screen = self.world_to_screen(wall[0][0], wall[0][1])
            end_screen = self.world_to_screen(wall[1][0], wall[1][1])
            
            if ((-50 <= start_screen[0] <= self.width + 50 or -50 <= end_screen[0] <= self.width + 50) and
                (-50 <= start_screen[1] <= self.height + 50 or -50 <= end_screen[1] <= self.height + 50)):
                pygame.draw.line(self.screen, self.colors['track_wall'], start_screen, end_screen, 3)
        
        # Renderiza linha de chegada
        if finish_line:
            start_screen = self.world_to_screen(finish_line[0][0], finish_line[0][1])
            end_screen = self.world_to_screen(finish_line[1][0], finish_line[1][1])
            
            num_squares = 10
            dx = (end_screen[0] - start_screen[0]) / num_squares
            dy = (end_screen[1] - start_screen[1]) / num_squares
            
            for i in range(num_squares):
                x1 = start_screen[0] + dx * i
                y1 = start_screen[1] + dy * i
                x2 = start_screen[0] + dx * (i + 1)
                y2 = start_screen[1] + dy * (i + 1)
                
                # Alterna entre branco e preto
                color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
                pygame.draw.line(self.screen, color, (int(x1), int(y1)), (int(x2), int(y2)), 12)
    
    # Renderiza um único carro
    def render_car(self, car: Car, car_type: str = 'normal'):
        if car_type == 'best':
            color = self.colors['car_best']
        elif car_type == 'second':
            color = self.colors['car_second']
        elif car_type == 'dead':
            color = self.colors['car_dead']
        else:
            color = self.colors['car_normal']
        
        # Obtém os cantos do carro em coordenadas do mundo
        corners = car.get_car_corners()
        
        # Converte para coordenadas da tela
        screen_corners = []
        for corner in corners:
            screen_corner = self.world_to_screen(corner[0], corner[1])
            screen_corners.append(screen_corner)
        
        # Renderiza apenas se visível na tela
        car_screen = self.world_to_screen(car.x, car.y)
        if (-100 <= car_screen[0] <= self.width + 100 and 
            -100 <= car_screen[1] <= self.height + 100):
            
            # Desenha o corpo do carro
            pygame.draw.polygon(self.screen, color, screen_corners)
            pygame.draw.polygon(self.screen, self.colors['track_wall'], screen_corners, 2)
            
            # Desenha indicador de direção
            front_x = car.x + math.cos(car.angle) * car.width / 2
            front_y = car.y + math.sin(car.angle) * car.width / 2
            front_screen = self.world_to_screen(front_x, front_y)
            pygame.draw.circle(self.screen, (255, 255, 255), front_screen, 3)
    
    # Renderiza os sensores do carro
    def render_sensors(self, car: Car):
        if not self.show_sensors:
            return
        
        car_screen = self.world_to_screen(car.x, car.y)
        
        # Renderiza apenas se visível na tela
        if (-100 <= car_screen[0] <= self.width + 100 and 
            -100 <= car_screen[1] <= self.height + 100):
            
            sensor_endpoints = car.get_sensor_endpoints()
            
            for i, endpoint in enumerate(sensor_endpoints):
                endpoint_screen = self.world_to_screen(endpoint[0], endpoint[1])
                
                distance = car.sensor_readings[i] if i < len(car.sensor_readings) else car.sensor_length
                if distance < car.sensor_length:
                    color = self.colors['sensor_hit']
                    alpha = int(255 * (1 - distance / car.sensor_length))
                else:
                    color = self.colors['sensor']
                    alpha = 100
                
                temp_surface = pygame.Surface((abs(endpoint_screen[0] - car_screen[0]) + 1, 
                                             abs(endpoint_screen[1] - car_screen[1]) + 1))
                temp_surface.set_alpha(alpha)
                temp_surface.fill(color)
                
                pygame.draw.line(self.screen, color, car_screen, endpoint_screen, 1)
                pygame.draw.circle(self.screen, color, endpoint_screen, 3)
    
    # Renderiza todos os carros
    def render_cars(self, cars: List[Car], best_indices: List[int] = None):
        if best_indices is None:
            best_indices = []
        
        # Renderiza carros mortos
        for car in cars:
            if not car.is_alive:
                self.render_car(car, 'dead')
        
        # Renderiza carros vivos
        for i, car in enumerate(cars):
            if car.is_alive and i not in best_indices:
                self.render_car(car, 'normal')
                if i == 0 or (len(best_indices) > 0 and i == best_indices[0]):
                    self.render_sensors(car)
        
        # Renderiza os melhores carros
        for i, car_idx in enumerate(best_indices):
            if car_idx < len(cars) and cars[car_idx].is_alive:
                car_type = 'best' if i == 0 else 'second'
                self.render_car(cars[car_idx], car_type)
                if i == 0:  # Mostra sensores apenas do melhor carro
                    self.render_sensors(cars[car_idx])
    
    def render_ui(self, generation: int, cars: List[Car], best_car: Optional[Car] = None, 
                  generation_time: float = 0.0):
        # UI background
        ui_rect = pygame.Rect(self.width - self.ui_panel_width, 0, self.ui_panel_width, self.height)
        pygame.draw.rect(self.screen, self.colors['ui_bg'], ui_rect)
        pygame.draw.rect(self.screen, self.colors['ui_border'], ui_rect, 2)
        
        y_offset = 20
        
        # Generation info
        gen_text = self.font_large.render(f"Generation: {generation}", True, self.colors['text'])
        self.screen.blit(gen_text, (self.width - self.ui_panel_width + 10, y_offset))
        y_offset += 40
        
        # Cars alive count
        alive_count = sum(1 for car in cars if car.is_alive)
        alive_text = self.font_medium.render(f"Cars Alive: {alive_count}/{len(cars)}", True, self.colors['text'])
        self.screen.blit(alive_text, (self.width - self.ui_panel_width + 10, y_offset))
        y_offset += 30
        
        # Generation time
        time_text = self.font_medium.render(f"Time: {generation_time:.1f}s", True, self.colors['text'])
        self.screen.blit(time_text, (self.width - self.ui_panel_width + 10, y_offset))
        y_offset += 30
        
        # Best car info
        if best_car:
            fitness_text = self.font_medium.render(f"Best Fitness: {best_car.fitness:.1f}", True, self.colors['text'])
            self.screen.blit(fitness_text, (self.width - self.ui_panel_width + 10, y_offset))
            y_offset += 25
            
            # Mostra DISTÂNCIA PERCORRIDA ao longo da pista
            distance_text = self.font_small.render(f"Distance: {best_car.distance_traveled:.1f}", True, self.colors['text'])
            self.screen.blit(distance_text, (self.width - self.ui_panel_width + 10, y_offset))
            y_offset += 20
            
            laps_text = self.font_small.render(f"Laps: {best_car.laps_completed}", True, self.colors['text'])
            self.screen.blit(laps_text, (self.width - self.ui_panel_width + 10, y_offset))
            y_offset += 20
            
            # Velocidade média
            avg_speed_text = self.font_small.render(f"Avg Speed: {best_car.avg_speed:.1f}", True, self.colors['text'])
            self.screen.blit(avg_speed_text, (self.width - self.ui_panel_width + 10, y_offset))
            y_offset += 20
            
            # Melhor volta
            if best_car.best_lap_time < float('inf'):
                lap_time_str = f"{best_car.best_lap_time:.2f}s"
            else:
                lap_time_str = "N/A"
            best_lap_text = self.font_small.render(f"Best Lap: {lap_time_str}", True, self.colors['text'])
            self.screen.blit(best_lap_text, (self.width - self.ui_panel_width + 10, y_offset))
            y_offset += 20
            
            # Pontos de posição
            position_pts_text = self.font_small.render(f"Position Pts: {best_car.position_points}", True, self.colors['text'])
            self.screen.blit(position_pts_text, (self.width - self.ui_panel_width + 10, y_offset))
            y_offset += 20
            
            time_alive_text = self.font_small.render(f"Time Alive: {best_car.time_alive:.1f}s", True, self.colors['text'])
            self.screen.blit(time_alive_text, (self.width - self.ui_panel_width + 10, y_offset))
            y_offset += 30
            
            # Neural network outputs
            steering_text = self.font_small.render(f"Steering: {best_car.steering_output:.2f}", True, self.colors['text'])
            self.screen.blit(steering_text, (self.width - self.ui_panel_width + 10, y_offset))
            y_offset += 20
            
            speed_text = self.font_small.render(f"Speed: {best_car.speed_output:.2f}", True, self.colors['text'])
            self.screen.blit(speed_text, (self.width - self.ui_panel_width + 10, y_offset))
            y_offset += 40
            
            # Render neural network visualization
            if self.show_neural_network:
                self.render_neural_network(best_car, y_offset)
    
    # Renderiza a rede neural do carro
    def render_neural_network(self, car: Car, start_y: int):
        if not hasattr(car, 'network_activations') or not car.network_activations:
            return
        
        nn_title = self.font_medium.render("Neural Network", True, self.colors['text'])
        self.screen.blit(nn_title, (self.width - self.ui_panel_width + 10, start_y))
        start_y += 30
        
        # Layout da rede
        layer_sizes = [5, 4, 3, 2]  # Entrada, ocultas1, ocultas2, saída
        layer_spacing = 60
        node_spacing = 25
        node_radius = 8
        
        start_x = self.width - self.ui_panel_width + 50
        
        # Calcula posições para todos os nós
        layer_positions = []
        for i, size in enumerate(layer_sizes):
            layer_y_start = start_y + 50 + (max(layer_sizes) - size) * node_spacing / 2
            positions = []
            for j in range(size):
                x = start_x + i * layer_spacing
                y = layer_y_start + j * node_spacing
                positions.append((x, y))
            layer_positions.append(positions)
        
        # Desenha conexões (pesos)
        weights = car.neural_network.weights
        for layer_idx in range(len(weights)):
            weight_matrix = weights[layer_idx]
            for i in range(len(layer_positions[layer_idx])):
                for j in range(len(layer_positions[layer_idx + 1])):
                    weight = weight_matrix[i][j]
                    
                    # Cor e espessura baseadas no peso
                    color = self.colors['neural_positive'] if weight > 0 else self.colors['neural_negative']
                    thickness = max(1, int(abs(weight) * 3))
                    
                    start_pos = layer_positions[layer_idx][i]
                    end_pos = layer_positions[layer_idx + 1][j]
                    
                    pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)
        
        # Desenha nós com valores de ativação
        activations = car.network_activations
        for layer_idx, positions in enumerate(layer_positions):
            for node_idx, (x, y) in enumerate(positions):
                # Obtém valor de ativação se disponível
                if layer_idx < len(activations) and node_idx < len(activations[layer_idx]):
                    activation = float(activations[layer_idx][node_idx])
                    # Normaliza ativação para intensidade de cor
                    intensity = min(255, max(0, int(abs(activation) * 255)))
                    if activation > 0:
                        color = (0, intensity, 0)  # Verde para positivo
                    else:
                        color = (intensity, 0, 0)  # Vermelho para negativo
                else:
                    color = self.colors['neural_node']
                
                # Desenha nó
                pygame.draw.circle(self.screen, color, (int(x), int(y)), node_radius)
                pygame.draw.circle(self.screen, self.colors['text'], (int(x), int(y)), node_radius, 1)
        
        # Desenha rótulos das camadas
        labels = ["Entrada", "Ocultas1", "Ocultas2", "Saída"]
        for i, (label, positions) in enumerate(zip(labels, layer_positions)):
            if positions:
                label_text = self.font_small.render(label, True, self.colors['text'])
                label_x = positions[0][0] - label_text.get_width() // 2
                label_y = positions[-1][1] + 20
                self.screen.blit(label_text, (label_x, label_y))
    
    # Renderiza toda a cena
    def render(self, cars: List[Car], walls: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
              finish_line: Tuple[Tuple[float, float], Tuple[float, float]], generation: int, 
              best_indices: List[int] = None, generation_time: float = 0.0):

        self.screen.fill(self.colors['background'])
        
        # Atualiza câmera para seguir o melhor carro
        if best_indices and len(best_indices) > 0 and best_indices[0] < len(cars):
            self.update_camera(cars[best_indices[0]])
        
        self.render_track(walls, finish_line)
        self.render_cars(cars, best_indices)
        best_car = cars[best_indices[0]] if best_indices and len(best_indices) > 0 and best_indices[0] < len(cars) else None
        self.render_ui(generation, cars, best_car, generation_time)
        
        pygame.display.flip()
    
    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    self.show_sensors = not self.show_sensors
                elif event.key == pygame.K_n:
                    self.show_neural_network = not self.show_neural_network
                elif event.key == pygame.K_c:
                    self.camera_smooth = 0.5 if self.camera_smooth == 0.1 else 0.1
        
        return True
    
    def cleanup(self):
        pygame.quit()