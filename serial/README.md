Etapa 0 — Versão sequencial (baseline)

Compilar e executar a versão sequencial kmeans_1d_naive.c digitando o código abaixo no console de sua IDE:
gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm
./kmeans_1d_naive dados.csv centroides_iniciais.csv 50 0.000001 assign_naive.csv centroids_naive.csv
cat centroids_naive.csv

Nota 1: os arquivos de entrada são: dados.csv, centroides_iniciais.csv
Nota 2: os arquivos de saída são: assign_naive.csv, centroids_naive.csv
Nota 3: você pode alterar os valores de max_iter (50) e eps (0.000001).

O resultado da execução de kmeans_id_naive.exe será algo como:
K-means 1D (naive)
N=10000 K=4 max_iter=50 eps=1e-06
Iterações: 3 | SSE final: 62193.762237 | Tempo: 0.3 ms
9.958585
29.970522
59.999429
89.968199

Nota 1: Cada um dos 4 valores acima é um centróide.
Nota 2: O número de centróides é igual ao valor do K do arquivo centroides_iniciais.csv utilizado.

Foram utilizados 3 conjuntos de teste (1D), que estão disponíveis nas pastas correspondentes:
Pequeno: N=10^4, K=4
Médio: N=10^5, K=8
Grande: N=10^6, K=16

