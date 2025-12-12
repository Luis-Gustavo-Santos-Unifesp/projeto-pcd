Para rodar no Google Colab a implementação do kmeans 1D com MPI desenvolvida neste trabalho, execute o código abaixo. Importante lembrar de salvar os arquivos dados.csv e centroides.csv no mesmo diretório do arquivo kmeans_1d_mpi.c

%%shell

# Instala MPICH se necessário
if ! command -v mpicc &> /dev/null
then
    echo "Instalando MPICH..."
    apt-get update -qq
    apt-get install -y -qq mpich
fi

echo "Compilando..."
mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm

if [ ! -f "./kmeans_1d_mpi" ]; then
    echo "Erro na compilação"
    exit 1
fi

echo "Compilação OK!"

# Executa K-Means em 1, 2, 4, 8 e 16 processos
for P in 1 2 4 8 16; do
    echo ""
    echo ">>> Executando com NP = $P <<<"
    mpirun --allow-run-as-root --oversubscribe -np $P ./kmeans_1d_mpi dados.csv centroides_iniciais.csv 50 1e-6 assign_mpi.csv centroids_mpi.csv
done
