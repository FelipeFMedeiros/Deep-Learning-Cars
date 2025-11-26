import math
import numpy as np
from typing import List, Tuple


class Track:
    """
    Gerador de pista para a simulação de carros.
    Cria diversos layouts de pista com paredes e checkpoints.
    """
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.walls = []
        self.start_position = (100, 400)
        self.start_angle = 0
    
    def create_simple_oval(self) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                                        Tuple[Tuple[float, float], Tuple[float, float]], 
                                        Tuple[float, float], float]:
        self.walls = []
        
        # Parâmetros da pista
        center_x = self.width // 2
        center_y = self.height // 2
        outer_radius_x = 400
        outer_radius_y = 200
        inner_radius_x = 250
        inner_radius_y = 100
        
        # Gera paredes do oval
        num_points = 50
        outer_points = []
        inner_points = []
        
        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            
            # Pontos da parede externa
            outer_x = center_x + outer_radius_x * math.cos(angle)
            outer_y = center_y + outer_radius_y * math.sin(angle)
            outer_points.append((outer_x, outer_y))
            
            # Pontos da parede interna
            inner_x = center_x + inner_radius_x * math.cos(angle)
            inner_y = center_y + inner_radius_y * math.sin(angle)
            inner_points.append((inner_x, inner_y))
        
        # Cria segmentos de parede para o limite externo
        for i in range(len(outer_points) - 1):
            self.walls.append((outer_points[i], outer_points[i + 1]))
        
        # Cria segmentos de parede para o limite interno
        for i in range(len(inner_points) - 1):
            self.walls.append((inner_points[i], inner_points[i + 1]))
        
        # LINHA DE CHEGADA
        finish_x = center_x
        finish_y_top = center_y + inner_radius_y
        finish_y_bottom = center_y + outer_radius_y
        finish_line = ((finish_x, finish_y_top), (finish_x, finish_y_bottom))
        
        # SPAWN POINT
        self.start_position = (finish_x - 30, center_y + (outer_radius_y + inner_radius_y) / 2)
        self.start_angle = 0  # Apontando PARA A DIREITA (vai cruzar a linha e seguir na pista)
        
        return self.walls, finish_line, self.start_position, self.start_angle