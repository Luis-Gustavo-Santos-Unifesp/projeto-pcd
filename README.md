# projeto-pcd
Projeto para implementar o k-means em 1 dimensão (pontos X[i] e centróides C[c]), medir SSE e desempenho, e paralelizar o núcleo do algoritmo em três etapas independentes:  OpenMP (CPU memória compartilhada) CUDA (GPU) MPI (memória distribuída)

Na pasta projeto-pcd/serial/ encontram-se todas as informações e arquivos usados para gerar os resultados relativos à versão serial (naive) do programa de cálculo do K-means 1D, incluindo um arquivo README.md com as instruções de como compilar/rodar os programas desenvolvidos.

Na pasta projeto-pcd/openmp/ encontram-se todas as informações e arquivos usados para gerar os resultados relativos à versão openmp (paralelizada) do programa de cálculo do K-means 1D, com um arquivo README.md com as instruções de como compilar/rodar os programas desenvolvidos.
