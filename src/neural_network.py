import numpy as np
from typing import List, Tuple


class NeuralNetwork:
    """
    Rede neural feedforward para controle autônomo do carro.
    
    Arquitetura:
        - Camada de entrada: 5 neurônios (sensores de distância)
        - Camada oculta 1: 4 neurônios
        - Camada oculta 2: 3 neurônios
        - Camada de saída: 2 neurônios (ângulo de direção e velocidade)
    
    Funções de ativação:
        - ReLU nas camadas ocultas
        - Tanh na camada de saída (valores entre -1 e 1)
    """
    
    def __init__(self, weights: List[np.ndarray] = None):
        # Definição da arquitetura
        self.input_size = 5
        self.hidden1_size = 4
        self.hidden2_size = 3
        self.output_size = 2
        
        # Inicializa pesos aleatoriamente ou usa pesos fornecidos
        if weights is None:
            self.weights = self._initialize_random_weights()
        else:
            self.weights = weights.copy()
    
    def _initialize_random_weights(self) -> List[np.ndarray]:
        """
        Inicializa pesos.
        Normaliza pela raiz quadrada do número de entradas de cada camada.
        """
        weights = []
        
        w1 = np.random.randn(self.input_size, self.hidden1_size) * np.sqrt(1.0 / self.input_size)
        weights.append(w1)
        
        w2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(1.0 / self.hidden1_size)
        weights.append(w2)
        
        w3 = np.random.randn(self.hidden2_size, self.output_size) * np.sqrt(1.0 / self.hidden2_size)
        weights.append(w3)
        
        return weights
    
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Propaga os inputs pela rede (forward propagation).
        
        Args:
            inputs: Array com 5 valores dos sensores de distância 
        Returns:
            outputs: Array com 2 valores (direção e velocidade)
            activations: Lista com ativações de cada camada
        """
        activations = []
        current_input = inputs.reshape(-1, 1) if inputs.ndim == 1 else inputs
        activations.append(current_input)
        
        # Camada oculta 1
        z1 = np.dot(current_input.T, self.weights[0])
        a1 = self._relu(z1)
        activations.append(a1.T)
        
        # Camada oculta 2
        z2 = np.dot(a1, self.weights[1])
        a2 = self._relu(z2)
        activations.append(a2.T)
        
        # Camada de saída
        z3 = np.dot(a2, self.weights[2])
        outputs = self._tanh(z3)
        activations.append(outputs.T)
        
        return outputs.flatten(), activations
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU: Retorna max(0, x) - ativa apenas valores positivos"""
        return np.maximum(0, x)
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh: Retorna valores entre -1 e 1"""
        return np.tanh(x)
    
    def get_weights_flat(self) -> np.ndarray:
        """
        Converte todas as matrizes de pesos em um único array 1D.
        Útil para algoritmos genéticos que precisam manipular os pesos como um vetor.
        """
        flat_weights = []
        for weight_matrix in self.weights:
            flat_weights.extend(weight_matrix.flatten())
        return np.array(flat_weights)
    
    def set_weights_flat(self, flat_weights: np.ndarray) -> None:
        """
        Reconstrói as matrizes de pesos a partir de um array 1D.
        Operação inversa de get_weights_flat().
        """
        start_idx = 0
        
        # Pesos da camada input -> hidden1
        end_idx = start_idx + self.input_size * self.hidden1_size
        self.weights[0] = flat_weights[start_idx:end_idx].reshape(
            self.input_size, self.hidden1_size
        )
        start_idx = end_idx
        
        # Pesos da camada hidden1 -> hidden2
        end_idx = start_idx + self.hidden1_size * self.hidden2_size
        self.weights[1] = flat_weights[start_idx:end_idx].reshape(
            self.hidden1_size, self.hidden2_size
        )
        start_idx = end_idx
        
        # Pesos da camada hidden2 -> output
        end_idx = start_idx + self.hidden2_size * self.output_size
        self.weights[2] = flat_weights[start_idx:end_idx].reshape(
            self.hidden2_size, self.output_size
        )
    
    def copy(self) -> 'NeuralNetwork':
        """Cria uma cópia independente da rede neural"""
        return NeuralNetwork([w.copy() for w in self.weights])
    
    def get_total_weights(self) -> int:
        """Retorna o número total de pesos na rede"""
        total = 0
        for weight_matrix in self.weights:
            total += weight_matrix.size
        return total