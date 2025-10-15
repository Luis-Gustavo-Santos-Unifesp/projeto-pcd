import pandas as pd
import numpy as np

# Nomes dos arquivos a serem comparados
path_centroids_naive = 'centroids_naive.csv'
path_centroids_omp = 'centroids_omp_4threads.csv'
path_assign_naive = 'assign_naive.csv'
path_assign_omp = 'assign_omp_4threads.csv'

try:
    # Lê os arquivos
    centroids_naive_df = pd.read_csv(path_centroids_naive, header=None, names=['Centroid'])
    centroids_omp_df = pd.read_csv(path_centroids_omp, header=None, names=['Centroid'])
    assign_naive_df = pd.read_csv(path_assign_naive, header=None, names=['Cluster_ID'])
    assign_omp_df = pd.read_csv(path_assign_omp, header=None, names=['Cluster_ID'])

    print("--- Análise dos Arquivos de Saída do K-Means ---\n")

    # 1. Comparação dos Centróides
    print("--- 1. Comparação dos Centróides Finais ---")
    are_centroids_equal = False
    try:
        # np.allclose lida com pequenas diferenças de ponto flutuante
        if np.allclose(centroids_naive_df.sort_values(by='Centroid'), centroids_omp_df.sort_values(by='Centroid')):
            are_centroids_equal = True
    except Exception:
        pass # Mantém are_centroids_equal como False se houver erro

    if are_centroids_equal:
        print("RESULTADO: SUCESSO! Os conjuntos de centróides finais são funcionalmente idênticos.\n")
    else:
        print("RESULTADO: FALHA! Os centróides finais são DIFERENTES.\n")
        print("Sequencial:\n", centroids_naive_df)
        print("\nParalelo:\n", centroids_omp_df)


    # 2. Comparação das Atribuições
    print("--- 2. Comparação das Atribuições de Cluster ---")
    if assign_naive_df.equals(assign_omp_df):
        print("RESULTADO: SUCESSO! Os arquivos de atribuição de pontos são idênticos.")
    else:
        print("RESULTADO: FALHA! Os arquivos de atribuição são diferentes.")

except FileNotFoundError as e:
    print(f"ERRO: Arquivo não encontrado: {e.filename}")
    print("Verifique se o script está na mesma pasta que os 4 arquivos CSV.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")