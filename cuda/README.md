A compilação da implementação do algoritmo kmeans 1d em CUDA foi feita como indicado abaixo, item 1, no ambiente do Google Colab.
Os testes foram feitas como indicado no item 2.

%%shell

# 1. Compilação
# A flag -arch=sm_75 é essencial para ativar suporte a double na Tesla T4
echo "Compilando kmeans_1d_cuda.cu..."
nvcc -O2 -arch=sm_75 kmeans_1d_cuda.cu -o kmeans_1d_cuda -lm

if [ ! -f "kmeans_1d_cuda" ]; then
    echo "ERRO CRÍTICO: O executável não foi criado."
    exit 1
fi

echo "Sucesso! O programa está pronto."
echo ""

# 2. Execução dos Testes
echo ">>> Teste A: Block Size = 128 <<<"
./kmeans_1d_cuda dados.csv centroides_iniciais.csv 50 1e-6 assign_cuda.csv centroids_cuda.csv 128

echo ""
echo ">>> Teste B: Block Size = 256 <<<"
./kmeans_1d_cuda dados.csv centroides_iniciais.csv 50 1e-6 assign_cuda.csv centroids_cuda.csv 256

echo ""
echo ">>> Teste C: Block Size = 512 <<<"
./kmeans_1d_cuda dados.csv centroides_iniciais.csv 50 1e-6 assign_cuda.csv centroids_cuda.csv 512
