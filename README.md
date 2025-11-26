# AI Cars - Genetic Algorithm Evolution Simulation

Este projeto implementa uma simulação de carros autônomos que aprendem a navegar através de diferentes percursos usando algoritmos genéticos e redes neurais.
> **Inspiração**: Este projeto foi inspirado em [Applying_EANNs](https://github.com/ArztSamuel/Applying_EANNs) por ArztSamuel.

![Simulação de Carros Autônomos](/assets/screenshot-1.png)

## Visão Geral

A simulação funciona da seguinte forma:

1. **População Inicial**: 20 carros são gerados com redes neurais aleatórias
2. **Sensores**: Cada carro possui 5 sensores frontais que detectam obstáculos (ângulos: -60°, -30°, 0°, 30°, 60°)
3. **Rede Neural**: Feedforward com arquitetura 5→4→3→2 (sensores→saída)
4. **Evolução**: Algoritmo genético evolui os carros baseado em fitness
5. **Fitness**: Calculado por voltas completadas, distância percorrida, velocidade média, melhor tempo de volta, posição de chegada e tempo de sobrevivência
6. **Sistema de Voltas**: Linha de chegada detecta quando carros completam voltas, registra tempos e posições

## Características

- **Pista oval** com linha de chegada e sistema de voltas
- **Visualização em tempo real** das redes neurais e sensores
- **Câmera que segue** o melhor carro automaticamente
- **Interface gráfica** com estatísticas detalhadas (geração, carros vivos, tempo, voltas, fitness)
- **Salvamento automático** das melhores redes neurais a cada 10 gerações
- **Sistema de ranking** por posição de chegada em cada volta
- **Detecção de colisão** e carros parados

## Instalação

1. Certifique-se de ter Python 3.7+ instalado
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Executar

```bash
python run.py
```

## Controles

- **P**: Pausar/Retomar simulação
- **R**: Reiniciar simulação
- **ESC**: Sair

## Arquitetura do Código

```
src/
├── main.py              # Arquivo principal da simulação
├── car.py               # Classe do carro com física e sensores
├── neural_network.py    # Implementação da rede neural
├── genetic_algorithm.py # Algoritmo genético para evolução
├── track.py             # Gerador de pistas
└── renderer.py          # Sistema de renderização com Pygame
```

## Configuração

Você pode modificar os parâmetros da simulação editando o dicionário `config` em `main.py` ou em `run.py` na função `run_custom_config()`:

```python
config = {
    'population_size': 20,              # Número de carros por geração
    'mutation_rate': 0.15,              # Taxa de mutação (0.0-1.0)
    'mutation_strength': 0.3,           # Intensidade da mutação
    'max_generations': 100,             # Máximo de gerações
    'max_laps_per_generation': 3,       # Geração termina após 3 voltas
    'max_generation_time': 120.0,       # Tempo máximo por geração (120s)
    'fps': 60,                          # Frames por segundo
    'screen_width': 1400,               # Largura da tela
    'screen_height': 800                # Altura da tela
}
```

## Pista

Atualmente a simulação utiliza uma **pista oval** com:
- Paredes externas e internas
- Linha de chegada vertical (checkered flag pattern)
- Spawn point posicionado antes da linha de chegada
- Sistema de detecção de cruzamento da linha para contagem de voltas

## Algoritmo Genético

O algoritmo implementa:

- **Seleção por torneio** para escolher pais
- **Crossover de ponto único** para gerar filhos
- **Mutação gaussiana** nos pesos
- **Elitismo** (melhor carro sempre sobrevive)

## Rede Neural

Arquitetura:
- **5 neurônios de entrada**: Leituras dos sensores (normalizadas 0-1)
- **4 neurônios na camada oculta 1**: Ativação ReLU
- **3 neurônios na camada oculta 2**: Ativação ReLU  
- **2 neurônios de saída**: Direção (-1 a 1) e Velocidade (-1 a 1)

### Interpretação da Visualização da Rede Neural

![Visualização da Rede Neural](/assets/screenshot-2.png)

A visualização em tempo real mostra como a rede neural "pensa" durante a simulação:

#### **Camada de Entrada (5 neurônios - esquerda)**
Representam os **5 sensores de distância** do carro:
- **Verde intenso** = Sensor detectando espaço livre (valor alto, longe da parede)
- **Preto/apagado** = Sensor detectando parede próxima (valor baixo)

Cada neurônio corresponde a um sensor nos ângulos: -60°, -30°, 0° (frente), 30°, 60°.

#### **Camadas Ocultas (4 e 3 neurônios - meio)**
Processam as informações dos sensores para extrair padrões:
- **Verde** = Ativação positiva (neurônio "ativo")
- **Vermelho** = Ativação negativa
- **Preto** = Neurônio inativo (valor próximo de zero)

Essas camadas identificam situações complexas como "parede à esquerda", "caminho livre à frente", ou "curva acentuada".

#### **Camada de Saída (2 neurônios - direita)**
Controlam as ações do carro:
- **Neurônio superior (Steering/Direção)**:
  - Verde → Virar para direita
  - Vermelho → Virar para esquerda
  - Preto → Seguir reto
  
- **Neurônio inferior (Speed/Velocidade)**:
  - Verde → Acelerar
  - Vermelho → Desacelerar
  - Preto → Velocidade moderada

#### **Conexões (Linhas entre neurônios)**
- **Cor verde** = Peso positivo (conexão excitatória - reforça a ativação)
- **Cor vermelha** = Peso negativo (conexão inibitória - inibe a ativação)
- **Espessura da linha** = Magnitude do peso (quanto mais grossa, mais forte é a influência)

As linhas mostram como cada sensor e camada intermediária influencia a decisão final do carro. Durante a evolução genética, esses pesos são otimizados para que o carro aprenda a navegar pela pista de forma eficiente.

## Cálculo de Fitness

O fitness é calculado com base em múltiplos fatores que incentivam diferentes comportamentos:

```python
Fitness = (Voltas_Completadas × 10000) +     # Recompensa massiva por voltas
          (Distância_Percorrida × 2) +        # Incentiva progresso
          (Velocidade_Média × 100) +          # Incentiva velocidade
          (Bônus_Melhor_Volta) +              # 100/tempo × 1000 (voltas rápidas)
          (Pontos_Posição) +                  # 100-500 por posição de chegada
          (Tempo_Vivo × 5) -                  # Incentiva sobrevivência
          (Penalidade_Colisão)                # -200 ou -500
```

**Prioridades:**
1. Completar voltas (10.000 pontos por volta)
2. Fazer voltas rápidas (até 2.000+ pontos)
3. Chegar em boas posições (100-500 pontos)
4. Manter velocidade alta (100x velocidade média)
5. Percorrer distância (2x distância)
6. Sobreviver mais tempo (5x tempo)

## Visualização

### Cores dos Carros:
- **Verde**: Melhor carro da geração (maior distância percorrida)
- **Laranja/Bege**: Segundo melhor carro
- **Azul**: Carros normais vivos
- **Cinza**: Carros mortos (colisão ou parados)

### Sensores:
- **Linhas brancas**: Sensores não detectando obstáculos
- **Linhas vermelhas**: Sensores detectando paredes
- **5 sensores**: Posicionados em -60°, -30°, 0°, 30°, 60°

### Pista:
- **Linhas brancas**: Paredes da pista (externa e interna)
- **Faixa quadriculada**: Linha de chegada

### UI:
- **Painel direito**: Estatísticas em tempo real
- **Visualização da rede neural**: Mostra ativações e pesos

## Salvamento

As melhores redes neurais são automaticamente salvas:
- A cada 10 gerações: `best_network_gen_X.npy`
- No final da simulação: `best_network_final.npy`

## Estrutura Técnica

### Classes Principais

- **Car**: Representa um carro individual com física, sensores e rede neural
- **NeuralNetwork**: Implementação de rede neural feedforward
- **GeneticAlgorithm**: Lógica de evolução e seleção
- **Track**: Gerador procedural de pistas
- **Renderer**: Sistema de visualização com Pygame
- **CarSimulation**: Orquestrador principal

### Padrões Utilizados

- **Callback Pattern**: Para notificações de completar voltas
- **Component Pattern**: Para separar lógica de física/IA/rendering
- **Observer Pattern**: Para sistema de ranking e estatísticas

## Licença

Este é um projeto acadêmico open source e está disponível sob a licença MIT.

---

**Desenvolvido como um projeto acadêmico para demonstrar:**
- Algoritmos genéticos
- Redes neurais
- Simulações físicas
- Programação orientada a objetos
- Visualização de dados científicos