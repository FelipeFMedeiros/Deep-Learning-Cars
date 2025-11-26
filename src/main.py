import pygame
import sys
import time
import numpy as np
from typing import List, Optional

from car import Car
from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm
from track import Track
from renderer import Renderer


class CarSimulation:
    """
    Classe principal da simulação que orquestra todo o processo de evolução dos carros.
    """
    
    def __init__(self, config: dict = None):
        # Default configuration
        default_config = {
            'population_size': 20,
            'mutation_rate': 0.1,
            'mutation_strength': 0.5,
            'max_generations': 100,
            'fps': 60,
            'screen_width': 1400,
            'screen_height': 800,
            'max_laps_per_generation': 3,  # Avança geração após 3 voltas
            'max_generation_time': 120.0   # Tempo máximo: 120 segundos
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.genetic_algorithm = GeneticAlgorithm(
            population_size=self.config['population_size'],
            mutation_rate=self.config['mutation_rate'],
            mutation_strength=self.config['mutation_strength']
        )
        
        self.track = Track(
            width=self.config['screen_width'] - 300,
            height=self.config['screen_height']
        )
        
        self.renderer = Renderer(
            width=self.config['screen_width'],
            height=self.config['screen_height']
        )
        
        self.cars = []
        self.track_walls = []
        self.finish_line = None
        self.start_position = (100, 400)
        self.start_angle = 0
        
        self.generation = 0
        self.generation_start_time = 0
        self.paused_time_accumulator = 0
        self.pause_start_time = 0
        self.running = False
        self.paused = False
        self.speed_boost = False
        
        self.lap_rankings = {}
        self.current_lap_finishers = []
        
        # Performance tracking
        self.best_fitness_ever = 0
        self.best_car_ever = None
        
        self.clock = pygame.time.Clock()
        
        print(f"Inicializada simulação de carros com {self.config['population_size']} carros")
        print(f"Máximo de gerações: {self.config['max_generations']}")
        print(f"Geração termina após: {self.config['max_laps_per_generation']} voltas OU {self.config['max_generation_time']:.0f}s OU todos os carros mortos")
    
    def initialize_track(self):
        track_data = self.track.create_simple_oval()
        self.track_walls, self.finish_line, self.start_position, self.start_angle = track_data
    
    def initialize_population(self):
        self.cars = self.genetic_algorithm.create_initial_population(
            self.start_position[0], self.start_position[1], self.start_angle
        )
        self.generation = 0
        self.generation_start_time = time.time()
    
    def on_lap_completed(self, car: Car):
        lap_number = car.laps_completed
        
        # Inicializa lista de rankings para a volta (se não existir)
        if lap_number not in self.lap_rankings:
            self.lap_rankings[lap_number] = []
        
        # Registra os carros na lista de rankings da volta
        if car not in self.lap_rankings[lap_number]:
            self.lap_rankings[lap_number].append(car)
            position = len(self.lap_rankings[lap_number])
            
            # Atribui pontos baseado na posição (top 3)
            if position == 1:
                car.position_points += 600  # 1º lugar
                print(f"    🥇 1st place - Lap {lap_number} completed!")
            elif position == 2:
                car.position_points += 300  # 2º lugar
                print(f"    🥈 2nd place - Lap {lap_number} completed")
            elif position == 3:
                car.position_points += 100  # 3º lugar
                print(f"    🥉 3rd place - Lap {lap_number} completed")
    
    def update_simulation(self, dt: float):
        if self.paused:
            return

        # 
        updates_per_frame = 1
        
        for _ in range(updates_per_frame):
            for car in self.cars:
                if car.is_alive:
                    car.update(self.track_walls, self.finish_line, dt, self.on_lap_completed)
        
        # Verifica se a geração deve terminar
        cars_alive = sum(1 for car in self.cars if car.is_alive)
        # Desconta o tempo pausado do tempo total
        generation_time = time.time() - self.generation_start_time - self.paused_time_accumulator
        max_laps = max(car.laps_completed for car in self.cars)
        
        should_end = False
        end_reason = ""
        
        if max_laps >= self.config['max_laps_per_generation']:
            should_end = True
            end_reason = f"Limite de voltas atingido ({max_laps} voltas)"
        elif generation_time >= self.config['max_generation_time']:
            should_end = True
            end_reason = f"Tempo limite atingido ({generation_time:.1f}s)"
        elif cars_alive == 0:
            should_end = True
            end_reason = "Todos os carros morreram"
        
        if should_end and self.generation < self.config['max_generations']:
            print(f"  -> Geração encerrada: {end_reason}")
            self.evolve_generation()
    
    def evolve_generation(self):
        # Calcula o tempo de geração (descontando tempo pausado)
        generation_time = time.time() - self.generation_start_time - self.paused_time_accumulator
        
        # Registra melhor desempenho
        current_best = self.genetic_algorithm.get_best_car(self.cars)
        if current_best.fitness > self.best_fitness_ever:
            self.best_fitness_ever = current_best.fitness
            self.best_car_ever = current_best.neural_network.copy()
        
        # Cria próxima geração
        self.cars = self.genetic_algorithm.evolve_population(
            self.cars, self.start_position[0], self.start_position[1], self.start_angle,
            generation_time
        )
        
        self.generation = self.genetic_algorithm.generation
        self.generation_start_time = time.time()
        self.paused_time_accumulator = 0  # Reseta tempo pausado
        
        # Reseta sistema de ranking para nova geração
        self.lap_rankings = {}
        self.current_lap_finishers = []
        
        # Salva a melhor rede periodicamente
        if self.generation % 10 == 0:
            filename = f"best_network_gen_{self.generation}.npy"
            self.genetic_algorithm.save_best_network(self.cars, filename)
    
    def get_best_car_indices(self) -> List[int]:
        # Lista de tuplas (índice, distância)
        car_progress_pairs = [(i, car.distance_traveled) for i, car in enumerate(self.cars)]
        # Ordena pela distância percorrida
        car_progress_pairs.sort(key=lambda x: x[1], reverse=True)
        # Retorna os índices dos dois melhores carros
        return [pair[0] for pair in car_progress_pairs[:2]]
    
    def render_frame(self):
        best_indices = self.get_best_car_indices()
        generation_time = time.time() - self.generation_start_time - self.paused_time_accumulator
        
        self.renderer.render(
            cars=self.cars,
            walls=self.track_walls,
            finish_line=self.finish_line,
            generation=self.generation,
            best_indices=best_indices,
            generation_time=generation_time
        )
    
    def handle_input(self):
        if not self.renderer.handle_events():
            self.running = False
            return
        
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_p]:
            was_paused = self.paused
            self.paused = not self.paused
            
            if self.paused:
                # Começou a pausar - marca o tempo
                self.pause_start_time = time.time()
            else:
                # Despausou - acumula o tempo que ficou pausado
                if was_paused:
                    self.paused_time_accumulator += time.time() - self.pause_start_time
            
            time.sleep(0.2)
               
        if keys[pygame.K_r]:
            self.restart_simulation()
        
        if keys[pygame.K_ESCAPE]:
            self.running = False
    
    def restart_simulation(self):
        self.initialize_population()
        self.paused_time_accumulator = 0
        self.best_fitness_ever = 0
        self.best_car_ever = None
        print("Simulação reiniciada.")
    
    def run(self):
        print("Iniciando Simulação de Evolução de Carros...")
        print("Controls:")
        print("  P - Pausar/Retomar")
        print("  R - Reiniciar")
        print("  S - Alternar visualização dos sensores")
        print("  N - Alternar visualização da rede neural")
        print("  ESC - Sair")
        print()
        
        self.initialize_track()
        self.initialize_population()
        
        self.running = True
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            self.handle_input()
            self.update_simulation(dt)
            self.render_frame()
            
            # Mantém FPS consistente (não precisa dobrar FPS)
            self.clock.tick(self.config['fps'])
            
            # Verifica se o máximo de gerações foi alcançado
            if self.generation >= self.config['max_generations']:
                print("\nMáximo de gerações alcançado!")
                self.print_final_statistics()
                break
        
        self.cleanup() # Finaliza a simulação
    
    def print_final_statistics(self):
        stats = self.genetic_algorithm.get_statistics_summary()
        
        print("\n" + "="*50)
        print("SIMULAÇÃO COMPLETA")
        print("="*50)
        print(f"Total de Gerações: {stats.get('total_generations', 0)}")
        print(f"Melhor Aptidão Já Alcançada: {stats.get('best_fitness_ever', 0):.1f}")
        print(f"Melhor Aptidão Final: {stats.get('final_best_fitness', 0):.1f}")
        print(f"Melhoria Média por Geração: {stats.get('average_improvement', 0):.2f}")
        
        convergence_gen = stats.get('convergence_generation', -1)
        if convergence_gen > 0:
            print(f"A convergência começou na geração: {convergence_gen}")
        
        # Salva estatísticas em CSV
        self.genetic_algorithm.save_statistics_to_csv("evolution_statistics.csv")
        
        if self.cars:
            final_filename = "best_network_final.npy"
            self.genetic_algorithm.save_best_network(self.cars, final_filename)
        
        print("="*50)
    
    def cleanup(self):
        self.renderer.cleanup()
        print("Simulação finalizada.")


def main():
    # Configuração para a simulação
    config = {
        'population_size': 25,
        'mutation_rate': 0.15,
        'mutation_strength': 0.4,
        'max_generations': 100,
        'fps': 60,
        'screen_width': 1400,
        'screen_height': 800,
        'track_type': 'oval'
    }
    
    try:
        # Cria e executa a simulação
        simulation = CarSimulation(config)
        simulation.run()
    
    except KeyboardInterrupt:
        print("\nSimulação interrompida pelo usuário")
    except Exception as e:
        print(f"\nOcorreu um erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    main()