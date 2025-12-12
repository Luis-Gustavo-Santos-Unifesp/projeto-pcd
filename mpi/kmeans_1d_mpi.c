/* kmeans_1d_mpi.c
   K-Means 1D usando MPI (memória distribuída).
   Compilação: mpicc -O2 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
   Execução: mpirun --allow-run-as-root -np P ./kmeans_1d_mpi dados.csv centroides_iniciais.csv [max_iter] [eps] [assign_out.csv] [cent_out.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define MAX_LINE_LEN 4096

static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int rows = 0; char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') { only_ws = 0; break; }
        }
        if (!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double* read_csv(const char *path, int *n_out) {
    int R = count_rows(path);
    if (R == 0) { *n_out = 0; return NULL; }
    double *A = (double*)malloc(R * sizeof(double));
    FILE *f = fopen(path, "r");
    if (!f) { free(A); *n_out = 0; return NULL; }
    char line[MAX_LINE_LEN];
    int r = 0;
    while (fgets(line, sizeof(line), f) && r < R) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') { only_ws = 0; break; }
        }
        if (only_ws) continue;
        A[r++] = atof(line);
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_csv_int(const char *path, int *arr, int n) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int i = 0; i < n; i++) fprintf(f, "%d\n", arr[i]);
    fclose(f);
}

static void write_csv_double(const char *path, double *arr, int n) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int i = 0; i < n; i++) fprintf(f, "%.6f\n", arr[i]);
    fclose(f);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3 && rank == 0) {
        printf("Uso: mpirun -np P %s dados.csv centroides.csv [max_iter] [eps] [assign_out.csv] [cent_out.csv]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int max_iter = 50;
    double eps = 1e-4;
    char *pathX = NULL, *pathC = NULL;

    if (rank == 0) {
        pathX = argv[1];
        pathC = argv[2];
        if (argc > 3) max_iter = atoi(argv[3]);
        if (argc > 4) eps = atof(argv[4]);
    }

    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int N_global = 0, K = 0;
    double *X_global = NULL;
    double *C = NULL;

    if (rank == 0) {
        X_global = read_csv(pathX, &N_global);
        C = read_csv(pathC, &K);
        if (!X_global || !C) {
            fprintf(stderr, "Erro na leitura dos arquivos (rank 0). N=%d K=%d\n", N_global, K);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Rank 0: Lido N=%d pontos, K=%d centroides\n", N_global, K);
    }

    MPI_Bcast(&N_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N_global <= 0 || K <= 0) {
        if (rank == 0) fprintf(stderr, "N ou K inválidos.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank != 0) {
        C = (double*)malloc(K * sizeof(double));
        if (!C) { fprintf(stderr, "malloc C falhou (rank %d)\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // sendcounts/displs: root calcula e passa para todos via Bcast
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    if (!sendcounts || !displs) { fprintf(stderr, "malloc sendcounts/displs falhou\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    if (rank == 0) {
        int base = N_global / size;
        int rem = N_global % size;
        int sum = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = base + (i < rem ? 1 : 0);
            displs[i] = sum;
            sum += sendcounts[i];
        }
    }

    // Broadcast sendcounts/displs para todos
    MPI_Bcast(sendcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);

    // Agora cada rank sabe quantos elementos receberá
    int local_n = sendcounts[rank];
    double *local_X = NULL;
    if (local_n > 0) {
        local_X = (double*)malloc(local_n * sizeof(double));
        if (!local_X) { fprintf(stderr, "malloc local_X falhou (rank %d)\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    } else {
        // permite local_n == 0 (nenhum dado para este rank)
        local_X = NULL;
    }

    int *local_assign = (int*)malloc((local_n>0?local_n:1) * sizeof(int));
    if (!local_assign) { fprintf(stderr, "malloc local_assign falhou (rank %d)\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }

    // Scatterv: root provê X_global, outros recebem em local_X
    MPI_Scatterv(X_global, sendcounts, displs, MPI_DOUBLE,
                 local_X, local_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Preparar buffers para k-means
    double *local_sum = (double*)malloc(K * sizeof(double));
    int *local_cnt = (int*)malloc(K * sizeof(int));
    double *global_sum = (double*)malloc(K * sizeof(double));
    int *global_cnt = (int*)malloc(K * sizeof(int));
    if (!local_sum || !local_cnt || !global_sum || !global_cnt) {
        fprintf(stderr, "malloc acumuladores falhou (rank %d)\n", rank); MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double prev_sse = 1e300;
    double local_sse = 0.0, global_sse = 0.0;
    int it = 0;

    double start_time = MPI_Wtime();
    double total_comm_time = 0.0;

    for (it = 0; it < max_iter; it++) {
        // zera
        local_sse = 0.0;
        for (int k = 0; k < K; k++) { local_sum[k] = 0.0; local_cnt[k] = 0; }

        // assignment local
        for (int i = 0; i < local_n; i++) {
            double xi = local_X[i];
            int best_k = 0;
            double min_dist = (xi - C[0])*(xi - C[0]);
            for (int k = 1; k < K; k++) {
                double dist = (xi - C[k])*(xi - C[k]);
                if (dist < min_dist) { min_dist = dist; best_k = k; }
            }
            local_assign[i] = best_k;
            local_sse += min_dist;
            local_sum[best_k] += xi;
            local_cnt[best_k] ++;
        }

        double t_comm_start = MPI_Wtime();

        MPI_Allreduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_sum, global_sum, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cnt, global_cnt, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        total_comm_time += (MPI_Wtime() - t_comm_start);

        // update centroides localmente
        for (int k = 0; k < K; k++) {
            if (global_cnt[k] > 0) C[k] = global_sum[k] / (double)global_cnt[k];
            // caso global_cnt[k] == 0: mantem C[k] (ou poderia re-inicializar)
        }

        double rel_err = fabs(global_sse - prev_sse) / (prev_sse > 0 ? prev_sse : 1.0);
        if (it > 0 && rel_err < eps) { it++; break; }
        prev_sse = global_sse;
    }

    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;
    double max_total_time = 0.0, max_comm_time = 0.0;

    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gatherv: reunir assignments no root
    int *global_assign = NULL;
    if (rank == 0) {
        global_assign = (int*)malloc(N_global * sizeof(int));
        if (!global_assign) { fprintf(stderr, "malloc global_assign falhou\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    // sendcounts/displs são válidos em todos os ranks
    MPI_Gatherv(local_assign, local_n, MPI_INT,
                global_assign, sendcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("--------------------------------------------------\n");
        printf("MPI Resultados (P=%d):\n", size);
        printf("Iterações: %d\n", it);
        printf("SSE Final: %.6f\n", global_sse);
        printf("Tempo Total (max): %.4f s\n", max_total_time);
        printf("Tempo Comunicação (Allreduce): %.4f s\n", max_comm_time);
        printf("Fraçao Comunicação: %.2f%%\n", (max_comm_time/max_total_time)*100.0);
        printf("--------------------------------------------------\n");

        if (argc > 5) write_csv_int(argv[5], global_assign, N_global);
        if (argc > 6) write_csv_double(argv[6], C, K);
    }

    // cleanup
    if (X_global) free(X_global);
    if (global_assign) free(global_assign);
    if (sendcounts) free(sendcounts);
    if (displs) free(displs);
    if (local_X) free(local_X);
    if (local_assign) free(local_assign);
    if (C) free(C);
    if (local_sum) free(local_sum);
    if (local_cnt) free(local_cnt);
    if (global_sum) free(global_sum);
    if (global_cnt) free(global_cnt);

    MPI_Finalize();
    return 0;
}
