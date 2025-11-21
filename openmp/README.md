Como compilar a implementação do algoritmo kmeans 1D em OpenMP:
gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
./kmeans_1d_omp dados.csv centroides_iniciais.csv [max_iter] [eps] [assign.csv] [centroids.csv] [num_threads]

Neste projeto, adotamos max_iter = 50 e eps = 0.000001 para compilar a implementação do algoritmo kmeans 1D em OpenMP no Google Colab, como indicado no exemplo abaixo.

%%shell

gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
./kmeans_1d_omp dados.csv centroides_iniciais.csv 50 0.000001 assign_omp_16threads.csv centroids_omp_16threads.csv 16 cat centroids_omp_16threads.csv
