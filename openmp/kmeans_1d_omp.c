
/* kmeans_1d_omp.c
   K-means 1D (C99) com paralelização OpenMP.
   - Paraleliza os laços principais das funções de assignment e update.
   - A função de update usa acumuladores locais por thread para evitar condições de corrida.

   Compilar: gcc -O2 -fopenmp -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm
   Uso:      ./kmeans_1d_omp dados.csv centroides_iniciais.csv [max_iter] [eps] [assign.csv] [centroids.csv] [num_threads]
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h> // Incluído para OpenMP e medição de tempo de parede

/* ---------- Funções de Leitura/Escrita de CSV (sem alterações) ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;

        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}


/* ---------- k-means 1D - VERSÃO PARALELA COM OPENMP ---------- */

/* Etapa de atribuição (assignment) paralelizada */
static double assignment_step_1d_omp(const double *X, const double *C, int *assign, int N, int K){
    double sse = 0.0;

    // O laço principal sobre os pontos (N) é paralelizado.
    // A variável sse é somada usando a cláusula 'reduction' para evitar condição de corrida.
    // Versão implícita (não indica explicitamente que distribuição de blocos é estática)
    #pragma omp parallel for reduction(+:sse)
    // Versão estática (indicada explicitamente)
    //#pragma omp parallel for reduction(+:sse) schedule(static)
    // Versão estática com chunk size de 1000. O sistema distribui blocos de 1000 iterações.
    //#pragma omp parallel for reduction(+:sse) schedule(static, 1000)
    // Versão dinâmica com chunk de 1000. Threads pegam um bloco de 1000, e quando terminam, pedem outro.
    //#pragma omp parallel for reduction(+:sse) schedule(dynamic, 1000)

    for(int i=0; i<N; i++){
        int best = -1;
        double bestd = 1e300;
        // O laço interno sobre os centróides (K) é pequeno e executado por cada thread.
        for(int c=0; c<K; c++){
            double diff = X[i] - C[c];
            double d = diff * diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

/* Etapa de atualização (update) paralelizada - Opção A (acumuladores locais) */
static void update_step_1d_omp(const double *X, double *C, const int *assign, int N, int K){
    int num_threads = omp_get_max_threads();

    // Aloca acumuladores por thread: uma matriz para somas e uma para contagens.
    double *sum_threads = (double*)calloc((size_t)K * num_threads, sizeof(double));
    int *cnt_threads = (int*)calloc((size_t)K * num_threads, sizeof(int));
    if(!sum_threads || !cnt_threads){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    // O laço principal sobre os pontos (N) é paralelizado.
    // Versão implícita (não indica explicitamente que distribuição de blocos é estática)
    #pragma omp parallel for
    // Versão estática (indicada explicitamente)
    //#pragma omp parallel for schedule(static)
    // Versão estática com chunk size de 1000. O sistema distribui blocos de 1000 iterações.
    //#pragma omp parallel for schedule(static, 1000)
    // Versão dinâmica com chunk de 1000. Threads pegam um bloco de 1000, e quando terminam, pedem outro.
    //#pragma omp parallel for schedule(dynamic, 1000)

    for(int i=0; i<N; i++){
        int th_id = omp_get_thread_num();
        int a = assign[i];

        // Cada thread atualiza sua própria linha na matriz de acumuladores.
        // O índice é calculado para evitar que threads diferentes escrevam no mesmo lugar.
        cnt_threads[th_id * K + a] += 1;
        sum_threads[th_id * K + a] += X[i];
    }

    // --- Redução Sequencial ---
    // A thread principal (mestre) soma os resultados de todos os acumuladores parciais.
    for(int c=0; c<K; c++){
        double total_sum = 0.0;
        int total_cnt = 0;
        for (int t=0; t<num_threads; t++){
            total_sum += sum_threads[t * K + c];
            total_cnt += cnt_threads[t * K + c];
        }

        if(total_cnt > 0) C[c] = total_sum / (double)total_cnt;
        else              C[c] = X[0]; // Estratégia naive para clusters vazios
    }

    free(sum_threads);
    free(cnt_threads);
}


static void kmeans_1d_omp(const double *X, double *C, int *assign,
                       int N, int K, int max_iter, double eps,
                       int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d_omp(X, C, assign, N, K);

        // =================== VALIDAÇÃO DO SSE ===================
        if (it > 0 && sse > prev_sse) {
            // Imprime um aviso se o SSE aumentar. Isso indica um erro na lógica paralela!
            printf("AVISO na iteração %d: SSE aumentou! (Atual: %.6f > Anterior: %.6f)\n", it, sse, prev_sse);
        }
        // ==========================================================

        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }
        update_step_1d_omp(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv] [num_threads=4]\n", argv[0]);
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;
    int num_threads = (argc>7)? atoi(argv[7]) : 4; // Novo argumento para número de threads

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    // Define o número de threads para todas as regiões paralelas subsequentes
    if(num_threads > 0) omp_set_num_threads(num_threads);

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    // Usa omp_get_wtime() para medir o tempo de parede, mais preciso para benchmarks
    double t0 = omp_get_wtime();
    int iters = 0; double sse = 0.0;
    kmeans_1d_omp(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    double t1 = omp_get_wtime();
    double ms = 1000.0 * (t1 - t0);

    printf("K-means 1D (OpenMP)\n");
    printf("N=%d K=%d max_iter=%d eps=%g num_threads=%d\n", N, K, max_iter, eps, num_threads);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}
