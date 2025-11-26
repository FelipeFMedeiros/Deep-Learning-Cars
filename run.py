"""
AI Cars - Launcher Script
Simplifica a execução da simulação
"""

import sys
import os

sys.path.append('src')
from main import CarSimulation

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("🚗 AI Cars - Genetic Algorithm Evolution")
    print("=" * 40)
    
    while True:
        print("\\nEscolha uma opção:")
        print("1. Demo Rápida")
        print("2. Configuração Personalizada")
        print("0. Sair")
        
        choice = input("\\nDigite sua escolha (0-2): ").strip()
        
        if choice == "0":
            print("Até logo! 👋")
            break
        elif choice == "1":
            quick_demo()
        elif choice == "2":
            run_custom_config()
        else:
            print("❌ Opção inválida! Tente novamente.")

def quick_demo():
    """Configuração otimizada para demonstração rápida."""
    
    print("=== DEMO RÁPIDA - CARROS EVOLUTIVOS ===")
    print()
    
    # Configuração otimizada para demo
    demo_config = {
        'population_size': 12,
        'mutation_rate': 0.25,      # Mutação alta para evolução rápida
        'mutation_strength': 0.4,   # Mudanças significativas
        'max_generations': 15,
        'fps': 60,
        'screen_width': 1360,
        'screen_height': 800,
        'max_laps_per_generation': 3,
        'max_generation_time': 80.0
    }
    
    try:
        simulation = CarSimulation(demo_config)
        simulation.run()
    except KeyboardInterrupt:
        print("\\nDemo interrompida pelo usuário")
    except Exception as e:
        print(f"\\nErro na demo: {e}")
        import traceback
        traceback.print_exc()

def run_custom_config():
    """Interface para configuração personalizada."""
    print("\\n🔧 Configuração Personalizada")
    print("-" * 30)
    
    try:
        pop_size = int(input("Tamanho da população (10-50, padrão 20): ") or "20")
        max_gen = int(input("Máximo de gerações (10-100, padrão 30): ") or "30")
        
        config = {
            'population_size': pop_size,
            'max_generations': max_gen,
            'mutation_rate': 0.15,
            'mutation_strength': 0.3,
            'fps': 60,
            'screen_width': 1400,
            'screen_height': 800,
            'max_laps_per_generation': 3,
            'max_time_per_generation': 120.0
        }
        
        print(f"\n🚀 Iniciando simulação personalizada...")
        print(f"População: {pop_size} carros")
        print(f"Gerações: {max_gen}")
        print(f"Modo: Geração termina após 3 voltas OU 120s OU todos mortos")
        print(f"Pista: Oval")
        
        from main import CarSimulation
        simulation = CarSimulation(config)
        simulation.run()
        
    except ValueError:
        print("❌ Valores inválidos! Usando configuração padrão.")
        from quick_demo import quick_demo
        quick_demo()
    except KeyboardInterrupt:
        print("\\n⏸️ Configuração cancelada")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\n👋 Simulação encerrada pelo usuário")
    except Exception as e:
        print(f"\\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()