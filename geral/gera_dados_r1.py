import numpy as np

def gerar_dados_totalmente_aleatorios(tamanho_dataset="medio"):
    """
    Gera arquivos de dados (dados.csv) e centroides iniciais (centroides_iniciais.csv)
    com distribuição TOTALMENTE ALEATÓRIA (Uniforme).
    
    Args:
        tamanho_dataset (str): 'pequeno', 'medio' ou 'grande'.
    """

    # --- Configurações ---
    # Intervalo dos dados: valores entre 0.0 e 100.0 (ajuste se quiser outra escala)
    MIN_VAL = 0.0
    MAX_VAL = 100.0

    configs = {
        "pequeno": { "N": 10**4, "K": 4 },
        "medio":   { "N": 10**5, "K": 8 },
        "grande":  { "N": 10**6, "K": 16 }
    }

    if tamanho_dataset not in configs:
        print(f"Erro: Tamanho '{tamanho_dataset}' inválido.")
        return

    config = configs[tamanho_dataset]
    N = config["N"]
    K = config["K"]

    print(f"Gerando dados ALEATÓRIOS (Uniforme) - {tamanho_dataset}")
    print(f"N={N}, K={K}, Intervalo=[{MIN_VAL}, {MAX_VAL}]")

    # --- 1. Geração dos Pontos de Dados (dados.csv) ---
    # Gera N números aleatórios distribuídos uniformemente entre MIN e MAX
    dados = np.random.uniform(MIN_VAL, MAX_VAL, N)
    
    np.savetxt("dados.csv", dados, fmt='%.6f')
    print("-> 'dados.csv' gerado.")

    # --- 2. Geração dos Centróides Iniciais (centroides_iniciais.csv) ---
    # Gera K pontos aleatórios para servirem de ponto de partida
    centroides = np.random.uniform(MIN_VAL, MAX_VAL, K)
    
    # Opcional: Ordenar os centróides facilita a visualização, mas não é obrigatório pro K-Means
    centroides.sort() 

    np.savetxt("centroides_iniciais.csv", centroides, fmt='%.6f')
    print("-> 'centroides_iniciais.csv' gerado.")
    print("\nConcluído!")

# --- PONTO DE EXECUÇÃO ---
# Altere aqui para "pequeno", "medio" ou "grande"
TAMANHO_DESEJADO = "pequeno"
gerar_dados_totalmente_aleatorios(TAMANHO_DESEJADO)