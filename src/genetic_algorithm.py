import numpy as np
import random
from typing import List, Tuple
from neural_network import NeuralNetwork
from car import Car


class GeneticAlgorithm:
    """
    Implementação do algoritmo genético para evolução de redes neurais dos carros.
    """
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.15, 
                 mutation_strength: float = 0.4):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.generation = 0
        
        # Statistics tracking
        self.best_fitness_history = []
        self.average_fitness_history = []
        self.generation_stats = []
    
    def create_initial_population(self, start_x: float, start_y: float, 
                                start_angle: float = 0) -> List[Car]:
        population = []
        for _ in range(self.population_size):
            car = Car(start_x, start_y, start_angle)
            population.append(car) # Carro com rede neural inicializada aleatoriamente
        return population
    
    def evolve_population(self, cars: List[Car], start_x: float, start_y: float, 
                         start_angle: float = 0, generation_time: float = 0.0) -> List[Car]:
        """
        Evolui a população para criar uma nova geração.
        
        Args:
            cars: Geração atual de carros
            start_x: Posição x inicial para a nova geração
            start_y: Posição y inicial para a nova geração
            start_angle: Ângulo inicial para a nova geração
            generation_time: Tempo que a geração levou em segundos
            
        Returns:
            Nova geração de carros
        """
        self._calculate_statistics(cars, generation_time)
        
        cars.sort(key=lambda car: car.fitness, reverse=True)
        
        new_population = []
        
        # Mantem os top 3 carros (elitismo forte)
        for i in range(min(3, len(cars))):
            elite_car = Car(start_x, start_y, start_angle)
            elite_car.neural_network = cars[i].neural_network.copy()
            new_population.append(elite_car)
        
        # Gera o restante da população através de crossover e mutação
        while len(new_population) < self.population_size:
            # Seleciona os pais usando seleção por torneio
            parent1 = self._tournament_selection(cars)
            parent2 = self._tournament_selection(cars)
            
            # Cria filho através de crossover
            child_weights = self._crossover(
                parent1.neural_network.get_weights_flat(),
                parent2.neural_network.get_weights_flat()
            )
            
            # Aplica mutação
            child_weights = self._mutate(child_weights)
            
            # Cria novo carro com pesos evoluídos
            child_car = Car(start_x, start_y, start_angle)
            child_car.neural_network.set_weights_flat(child_weights)
            new_population.append(child_car)
        
        self.generation += 1
        return new_population
    
    def _tournament_selection(self, cars: List[Car], tournament_size: int = 3) -> Car:
        """
        Seleciona um pai usando seleção por torneio.
        
        Args:
            cars: População de carros
            tournament_size: Número de carros no torneio
            
        Returns:
            Carro pai selecionado
        """
        tournament = random.sample(cars, min(tournament_size, len(cars)))
        return max(tournament, key=lambda car: car.fitness)
    
    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray) -> np.ndarray:
        """
        Cria pesos do filho através de crossover.
        
        Args:
            parent1_weights: Pesos do primeiro pai
            parent2_weights: Pesos do segundo pai
            
        Returns:
            Pesos do filho
        """
        crossover_point = random.randint(1, len(parent1_weights) - 1)
        
        child_weights = np.concatenate([
            parent1_weights[:crossover_point],
            parent2_weights[crossover_point:]
        ])
        
        return child_weights
    
    def _mutate(self, weights: np.ndarray) -> np.ndarray:
        mutated_weights = weights.copy()
        
        # Aplica mutação a cada peso com a probabilidade dada
        for i in range(len(mutated_weights)):
            if random.random() < self.mutation_rate:
                # Adiciona ruído gaussiano
                mutated_weights[i] += np.random.normal(0, self.mutation_strength)
        
        return mutated_weights
    
    def _calculate_statistics(self, cars: List[Car], generation_time: float = 0.0):
        fitnesses = [car.fitness for car in cars]
        
        best_fitness = max(fitnesses)
        average_fitness = np.mean(fitnesses)
        worst_fitness = min(fitnesses)
        std_fitness = np.std(fitnesses)
        
        self.best_fitness_history.append(best_fitness)
        self.average_fitness_history.append(average_fitness)
        
        generation_stat = {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'average_fitness': average_fitness,
            'worst_fitness': worst_fitness,
            'std_fitness': std_fitness,
            'cars_alive': sum(1 for car in cars if car.is_alive),
            'generation_time': generation_time
        }
        
        self.generation_stats.append(generation_stat)
        
        best_car = max(cars, key=lambda c: c.fitness)
        
        # Formata tempo da melhor volta
        if best_car.best_lap_time < float('inf'):
            lap_time_str = f"{best_car.best_lap_time:.2f}s"
        else:
            lap_time_str = "N/A"
        
        print(f"Generation {self.generation:3d} | "
              f"Best: {best_fitness:8.1f} | "
              f"Avg: {average_fitness:8.1f} | "
              f"Laps: {best_car.laps_completed} | "
              f"Distance: {best_car.distance_traveled:6.1f} | "
              f"AvgSpeed: {best_car.avg_speed:5.1f} | "
              f"BestLap: {lap_time_str} | "
              f"PosPts: {best_car.position_points} | "
              f"Time: {generation_time:5.1f}s")
    
    def get_best_car(self, cars: List[Car]) -> Car:
        return max(cars, key=lambda car: car.fitness)
    
    def get_top_cars(self, cars: List[Car], n: int = 2) -> List[Car]:
        sorted_cars = sorted(cars, key=lambda car: car.fitness, reverse=True)
        return sorted_cars[:n]
    
    def save_best_network(self, cars: List[Car], filename: str):
        best_car = self.get_best_car(cars)
        weights = best_car.neural_network.get_weights_flat()
        np.save(filename, weights)
        print(f"Melhor rede salva em {filename}")
    
    def load_network(self, filename: str, start_x: float, start_y: float, 
                    start_angle: float = 0) -> Car:
        """
        Carrega uma rede neural de um arquivo.
        
        Args:
            filename: Arquivo para carregar
            start_x: Posição x inicial
            start_y: Posição y inicial
            start_angle: Ângulo inicial
            
        Returns:
            Carro com rede neural carregada
        """
        weights = np.load(filename)
        car = Car(start_x, start_y, start_angle)
        car.neural_network.set_weights_flat(weights)
        return car