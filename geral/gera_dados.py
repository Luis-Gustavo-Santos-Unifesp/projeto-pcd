import numpy as np
import os

def gerar_dados_kmeans(tamanho_dataset="medio"):
    """
    Gera arquivos de dados (dados.csv) e centroides iniciais (centroides_iniciais.csv)
    para o algoritmo K-means 1D.

    Args:
        tamanho_dataset (str): O nome da configuração a ser gerada ('pequeno', 'medio', 'grande').
    """

    # --- Configurações dos conjuntos de teste ---
    # O valor de 'K' está definido dentro de cada configuração.
    configs = {
        "pequeno": {
            "N": 10**4,
            "K": 4,
            "faixas_centroides": [10.0, 30.0, 60.0, 90.0],
            "desvio_padrao": 2.5
        },
        "medio": {
            "N": 10**5,
            "K": 8,
            "faixas_centroides": [5.0, 15.0, 25.0, 35.0, 55.0, 65.0, 75.0, 85.0],
            "desvio_padrao": 2.0
        },
        "grande": {
            "N": 10**6,
            "K": 16,
            "faixas_centroides": [5, 10, 15, 20, 25, 30, 35, 40, 60, 65, 70, 75, 80, 85, 90, 95],
            "desvio_padrao": 1.5
        }
    }

    if tamanho_dataset not in configs:
        print(f"Erro: Tamanho do dataset '{tamanho_dataset}' não reconhecido. Use 'pequeno', 'medio' ou 'grande'.")
        return

    config = configs[tamanho_dataset]
    N = config["N"]
    K = config["K"]
    faixas = config["faixas_centroides"]
    std_dev = config["desvio_padrao"]

    print(f"Gerando conjunto de dados '{tamanho_dataset}' com N={N}, K={K}...")

    # --- Geração dos Pontos de Dados (dados.csv) ---
    pontos_por_cluster = N // K
    todos_os_pontos = []

    for i in range(K):
        media_cluster = faixas[i]
        pontos = np.random.normal(loc=media_cluster, scale=std_dev, size=pontos_por_cluster)
        todos_os_pontos.extend(pontos)

    dados_finais = np.array(todos_os_pontos)
    np.random.shuffle(dados_finais)

    np.savetxt("dados.csv", dados_finais, fmt='%.6f')
    print("-> Arquivo 'dados.csv' gerado com sucesso.")

    # --- Geração dos Centróides Iniciais (centroides_iniciais.csv) ---
    centroides_iniciais = np.array(faixas)
    np.savetxt("centroides_iniciais.csv", centroides_iniciais, fmt='%.6f')
    print("-> Arquivo 'centroides_iniciais.csv' gerado com sucesso.")
    print("\nGeração concluída!")


# Mude a string abaixo para "pequeno", "medio", ou "grande" para gerar o conjunto de dados desejado.
TAMANHO_DESEJADO = "grande"
gerar_dados_kmeans(TAMANHO_DESEJADO)