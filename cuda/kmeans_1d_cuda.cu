
/* kmeans_1d_cuda.cu
   Implementação K-Means 1D em CUDA (Etapa 2)
   
   Características:
   - Assignment na GPU (1 thread/ponto).
   - Update na GPU (usando atomicAdd para soma e contagem).
   - Medição de tempos distintos: Transferência (H2D/D2H) e Processamento (Kernels).
   - Permite configurar o tamanho do bloco (BlockSize) via argumento.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_LINE_LEN 4096

// Macro para verificar erros da API CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

/* --- Funções Auxiliares (Leitura/Escrita CSV) --- */
static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Erro ao abrir %s\n", path); exit(1); }
    int rows = 0; char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), f)) {
        // Verifica se a linha não é vazia/espaço em branco
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
    double *A = (double*)malloc(R * sizeof(double));
    FILE *f = fopen(path, "r");
    char line[MAX_LINE_LEN];
    int r = 0;
    while (fgets(line, sizeof(line), f) && r < R) {
        // Pula linhas vazias
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
    for (int i = 0; i < n; i++) fprintf(f, "%d\n", arr[i]);
    fclose(f);
}

static void write_csv_double(const char *path, double *arr, int n) {
    if (!path) return;
    FILE *f = fopen(path, "w");
    for (int i = 0; i < n; i++) fprintf(f, "%.6f\n", arr[i]);
    fclose(f);
}

/* --- KERNELS CUDA --- */

/**
 * 1) Kernel de Assignment
 * - Cada thread processa um ponto i.
 * - Varre K centróides, encontra o mais próximo.
 * - Escreve em assign[i].
 * - Acumula o erro (SSE) atomicamente em d_sse.
 */
__global__ void assignment_kernel(const double *X, const double *C, int *assign, double *d_sse, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        double my_point = X[i];
        int best_k = -1;
        double min_dist_sq = 1.0e30; // Infinito
        
        // 1.1) Varre K centróides
        for (int c = 0; c < K; c++) {
            double dist = my_point - C[c];
            double dist_sq = dist * dist;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                best_k = c;
            }
        }
        
        assign[i] = best_k;
        
        // 2) SSE: acumula erro globalmente (redução atômica simples)
        atomicAdd(d_sse, min_dist_sq);
    }
}

/**
 * 3.1b) Opção B: Resetar acumuladores
 */
__global__ void clear_accumulators_kernel(double *sum, int *cnt, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) {
        sum[i] = 0.0;
        cnt[i] = 0;
    }
}

/**
 * 3.1b) Opção B: Update (Soma)
 * - Usa atomics para somar pontos e contar ocorrências nos clusters.
 */
__global__ void update_sum_kernel(const double *X, const int *assign, double *sum, int *cnt, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        int cluster_id = assign[i];
        // Acumulação atômica na memória global
        atomicAdd(&sum[cluster_id], X[i]);
        atomicAdd(&cnt[cluster_id], 1);
    }
}

/**
 * 3.1b) Opção B: Update (Média)
 * - 1 thread por centróide calcula a média final.
 */
__global__ void update_mean_kernel(double *C, const double *sum, const int *cnt, const double *X_orig, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c < K) {
        int count = cnt[c];
        if (count > 0) {
            C[c] = sum[c] / (double)count;
        } else {
            // Cluster vazio: estratégia naive (reutiliza X[0])
            C[c] = X_orig[0];
        }
    }
}

/* --- MAIN --- */
int main(int argc, char **argv) {
    if (argc < 6) {
        printf("Uso: %s <dados.csv> <centroides.csv> <max_iter> <eps> <assign_out.csv> <centroids_out.csv> [block_size]\n", argv[0]);
        return 1;
    }

    const char *pX = argv[1];
    const char *pC = argv[2];
    int max_iter = atoi(argv[3]);
    double eps = atof(argv[4]);
    const char *outAssign = argv[5];
    const char *outCentroids = argv[6];
    int blockSize = (argc > 7) ? atoi(argv[7]) : 256; // Padrão 256 se não informado

    // 1. Leitura no Host
    int N, K;
    double *h_X = read_csv(pX, &N);
    double *h_C = read_csv(pC, &K);
    int *h_assign = (int*)malloc(N * sizeof(int));

    printf("K-Means CUDA | N=%d, K=%d, BlockSize=%d\n", N, K, blockSize);

    // 2. Alocação no Device
    double *d_X, *d_C, *d_sse, *d_sum;
    int *d_assign, *d_cnt;
    
    CUDA_CHECK(cudaMalloc(&d_X, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_assign, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sse, sizeof(double)));
    // Arrays auxiliares para Update (Opção B)
    CUDA_CHECK(cudaMalloc(&d_sum, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));

    // Eventos para medição de tempo
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel, start_transfer, stop_transfer;
    cudaEventCreate(&start_total); cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel); cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_transfer); cudaEventCreate(&stop_transfer);

    // --- INÍCIO TEMPO TOTAL ---
    cudaEventRecord(start_total);

    // --- TRANSFERÊNCIA H2D (Host to Device) ---
    cudaEventRecord(start_transfer);
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, K * sizeof(double), cudaMemcpyHostToDevice));
    cudaEventRecord(stop_transfer);

    // Configuração do Grid
    int gridN = (N + blockSize - 1) / blockSize;
    int gridK = (K + blockSize - 1) / blockSize;

    double prev_sse = 1.0e30;
    double cur_sse = 0.0;
    int it = 0;
    float total_kernel_ms = 0.0f;

    // --- LOOP PRINCIPAL ---
    for (it = 0; it < max_iter; it++) {
        
        // Inicia medição da iteração de kernel
        cudaEventRecord(start_kernel);

        // 1. Zera SSE
        CUDA_CHECK(cudaMemset(d_sse, 0, sizeof(double)));

        // 2. ASSIGNMENT KERNEL
        assignment_kernel<<<gridN, blockSize>>>(d_X, d_C, d_assign, d_sse, N, K);

        // 3. UPDATE (Opção B - Atomics na GPU)
        // 3.a) Zera acumuladores
        clear_accumulators_kernel<<<gridK, blockSize>>>(d_sum, d_cnt, K);
        
        // 3.b) Acumula somas e contagens (O peso está aqui)
        update_sum_kernel<<<gridN, blockSize>>>(d_X, d_assign, d_sum, d_cnt, N);

        // 3.c) Calcula médias (novos centróides)
        // Passamos d_X para o caso de fallback (cluster vazio)
        update_mean_kernel<<<gridK, blockSize>>>(d_C, d_sum, d_cnt, d_X, K);

        cudaEventRecord(stop_kernel);
        cudaEventSynchronize(stop_kernel);
        float iter_ms;
        cudaEventElapsedTime(&iter_ms, start_kernel, stop_kernel);
        total_kernel_ms += iter_ms;

        // Recupera SSE para verificação de convergência (Pequena D2H)
        CUDA_CHECK(cudaMemcpy(&cur_sse, d_sse, sizeof(double), cudaMemcpyDeviceToHost));

        // Verificação de Convergência
        if (it > 0 && fabs(prev_sse - cur_sse) < eps) {
            it++; // Conta a iteração atual
            break;
        }
        prev_sse = cur_sse;
    }

    // --- TRANSFERÊNCIA D2H (Device to Host - Resultados Finais) ---
    // Aproveitamos o evento de transferência criado antes para somar o tempo
    float h2d_ms = 0.0f, d2h_ms = 0.0f;
    cudaEventSynchronize(stop_transfer);
    cudaEventElapsedTime(&h2d_ms, start_transfer, stop_transfer);

    cudaEventRecord(start_transfer); // Reutiliza handle para D2H
    CUDA_CHECK(cudaMemcpy(h_assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, K * sizeof(double), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop_transfer);
    cudaEventSynchronize(stop_transfer);
    cudaEventElapsedTime(&d2h_ms, start_transfer, stop_transfer);

    // --- FIM TEMPO TOTAL ---
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start_total, stop_total);

    // Cálculo de Throughput (Pontos por segundo no Kernel)
    // Total de pontos processados = N * iterações
    double points_per_sec = (double)N * it / (total_kernel_ms / 1000.0);

    printf("--------------------------------------------------\n");
    printf("Resultados CUDA:\n");
    printf("Iterações: %d\n", it);
    printf("SSE Final: %.6f\n", cur_sse);
    printf("Tempo H2D (Input): %.3f ms\n", h2d_ms);
    printf("Tempo Kernels (Loop): %.3f ms\n", total_kernel_ms);
    printf("Tempo D2H (Output): %.3f ms\n", d2h_ms);
    printf("Tempo TOTAL: %.3f ms\n", total_ms);
    printf("Throughput (Kernels): %.2e pontos/s\n", points_per_sec);
    printf("--------------------------------------------------\n");

    // Escrita dos arquivos
    write_csv_int(outAssign, h_assign, N);
    write_csv_double(outCentroids, h_C, K);

    // Limpeza
    free(h_X); free(h_C); free(h_assign);
    cudaFree(d_X); cudaFree(d_C); cudaFree(d_assign); 
    cudaFree(d_sse); cudaFree(d_sum); cudaFree(d_cnt);

    return 0;
}
