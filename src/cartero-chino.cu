#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <curand_kernel.h>
#include <chrono>

using namespace std;

#define NUM_HORMIGAS 64
#define HILOS_POR_BLOQUE 512

// parametros
#define MAX_ITERACIONES 2000
#define ALPHA 2.2f
#define BETA 1.9f
#define EVAPORACION 0.7f
#define Q 1000.0f
#define PENALIZACION_B 1000.0f

// limites fisicos de memoria estatica
#define MAX_N 128
#define MAX_PASOS 1024

// estructura estatica para la gpu
struct Hormiga {
    int nodo_inicial;
    int nodo_actual;
    float distancia_total;
    int aristas_unicas_visitadas;
    int pasos_dados;
    
    // arreglos aplanados de tamaño fijo
    int aristas_visitadas[MAX_N * MAX_N]; 
    int recorrido[MAX_PASOS];
};

// kernel 1: inicializar las semillas de aleatoriedad de cada hilo
__global__ void init_curand(curandState *state, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < NUM_HORMIGAS) {
        curand_init(seed, id, 0, &state[id]);
    }
}

// kernel 2: el cerebro de la hormiga
__global__ void mover_hormigas(Hormiga* colonia, const float* grafo, const float* feromonas, int N, int total_aristas, curandState* state) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h >= NUM_HORMIGAS) return; // evitamos hilos fantasma

    curandState local_state = state[h]; // copiamos estado a memoria local rapida
    Hormiga& hormiga = colonia[h];

    // reseteo de la memoria de la hormiga para esta iteracion
    hormiga.nodo_inicial = curand(&local_state) % N;
    hormiga.nodo_actual = hormiga.nodo_inicial;
    hormiga.distancia_total = 0.0f;
    hormiga.aristas_unicas_visitadas = 0;
    hormiga.pasos_dados = 0;
    
    for (int i = 0; i < N * N; i++) {
        hormiga.aristas_visitadas[i] = 0;
    }
    
    hormiga.recorrido[0] = hormiga.nodo_inicial;
    hormiga.pasos_dados++;

    // construccion de la ruta
    while ((hormiga.aristas_unicas_visitadas < total_aristas || hormiga.nodo_actual != hormiga.nodo_inicial) && hormiga.pasos_dados < MAX_PASOS) {
        int i = hormiga.nodo_actual;
        
        float probabilidades[MAX_N];
        float suma_probabilidades = 0.0f;

        // calculamos probabilidades
        for (int j = 0; j < N; j++) {
            probabilidades[j] = 0.0f;
            float peso_arista = grafo[i * N + j];
            
            if (peso_arista > 0.0f) {
                float tau = powf(feromonas[i * N + j], ALPHA);
                
                // penalizacion dinamica
                float veces_visitada = (float)hormiga.aristas_visitadas[i * N + j];
                float castigo = 1.0f + (veces_visitada * PENALIZACION_B);
                float eta = powf(1.0f / (peso_arista * castigo), BETA);

                probabilidades[j] = tau * eta;
                suma_probabilidades += probabilidades[j];
            }
        }

        // seleccion por ruleta en gpu
        float aleatorio = curand_uniform(&local_state); // decimal entre 0 y 1
        float limite = aleatorio * suma_probabilidades;
        float acumulado = 0.0f;
        int siguiente_nodo = -1;

        for (int j = 0; j < N; j++) {
            if (probabilidades[j] > 0.0f) {
                acumulado += probabilidades[j];
                if (acumulado >= limite) {
                    siguiente_nodo = j;
                    break;
                }
            }
        }

        // en caso de errores de precision
        if (siguiente_nodo == -1) {
            for (int j = 0; j < N; j++) {
                if (grafo[i * N + j] > 0.0f) { siguiente_nodo = j; break; }
            }
        }

        // actualizamos memoria de forma aplanada
        if (hormiga.aristas_visitadas[i * N + siguiente_nodo] == 0) {
            hormiga.aristas_unicas_visitadas++;
        }
        
        hormiga.aristas_visitadas[i * N + siguiente_nodo]++;
        hormiga.aristas_visitadas[siguiente_nodo * N + i]++; // ida y vuelta
        
        hormiga.distancia_total += grafo[i * N + siguiente_nodo];
        hormiga.nodo_actual = siguiente_nodo;
        
        hormiga.recorrido[hormiga.pasos_dados] = siguiente_nodo;
        hormiga.pasos_dados++;
    }
    
    state[h] = local_state; // guardamos semilla para la proxima iteracion
}

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        cout << "falta la ruta del archivo" << endl;
        return 1;
    }

    ifstream file(argv[1]);
    int N = 0;
    file >> N;
    
    if (N > MAX_N) {
        cout << "error: la matriz supera el limite de MAX_N" << endl;
        return 1;
    }

    // memoria en el host (cpu) - aplanada 1d
    vector<float> h_grafo(N * N, 0.0f);
    vector<float> h_feromonas(N * N, 0.1f);
    int total_aristas = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            file >> h_grafo[i * N + j];
            if (i < j && h_grafo[i * N + j] > 0) total_aristas++;
        }
    }
    file.close();

    // punteros para el device (gpu)
    float *d_grafo, *d_feromonas;
    Hormiga *d_colonia;
    curandState *d_state;

    // asignamos memoria en la gpu
    cudaMalloc(&d_grafo, N * N * sizeof(float));
    cudaMalloc(&d_feromonas, N * N * sizeof(float));
    cudaMalloc(&d_colonia, NUM_HORMIGAS * sizeof(Hormiga));
    cudaMalloc(&d_state, NUM_HORMIGAS * sizeof(curandState));

    // copiamos grafo inicial a la gpu
    cudaMemcpy(d_grafo, h_grafo.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // configuracion automatica de bloques
    int bloques = (NUM_HORMIGAS + HILOS_POR_BLOQUE - 1) / HILOS_POR_BLOQUE;

    // inicializamos generador de numeros aleatorios en la gpu
    init_curand<<<bloques, HILOS_POR_BLOQUE>>>(d_state, time(NULL));

    float mejor_distancia_global = 9999999.0f;
    vector<int> mejor_recorrido_global;

    vector<Hormiga> h_colonia(NUM_HORMIGAS); // buffer temporal para extraer datos

    // --- inicio cronometro ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int iter = 0; iter < MAX_ITERACIONES; iter++) {
        // pasamos feromonas actualizadas a la gpu
        cudaMemcpy(d_feromonas, h_feromonas.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

        // cada hilo es una hormiga buscando su camino
        mover_hormigas<<<bloques, HILOS_POR_BLOQUE>>>(d_colonia, d_grafo, d_feromonas, N, total_aristas, d_state);
        
        // traemos los resultados de las hormigas de vuelta a la cpu
        cudaMemcpy(h_colonia.data(), d_colonia, NUM_HORMIGAS * sizeof(Hormiga), cudaMemcpyDeviceToHost);

        // evaporacion global en cpu
        for (int i = 0; i < N * N; i++) {
            h_feromonas[i] *= (1.0f - EVAPORACION);
        }

        // evaluacion y deposito de feromonas en cpu
        for (int h = 0; h < NUM_HORMIGAS; h++) {
            Hormiga& hormiga = h_colonia[h];
            
            // si termino exitosamente
            if (hormiga.aristas_unicas_visitadas == total_aristas && hormiga.nodo_actual == hormiga.nodo_inicial) {
                
                // checamos si es la mejor historica
                if (hormiga.distancia_total < mejor_distancia_global) {
                    mejor_distancia_global = hormiga.distancia_total;
                    mejor_recorrido_global.clear();
                    for(int p = 0; p < hormiga.pasos_dados; p++) {
                        mejor_recorrido_global.push_back(hormiga.recorrido[p]);
                    }
                }

                // deposito
                float aporte = Q / hormiga.distancia_total;
                for (int p = 0; p < hormiga.pasos_dados - 1; p++) {
                    int desde = hormiga.recorrido[p];
                    int hasta = hormiga.recorrido[p+1];
                    h_feromonas[desde * N + hasta] += aporte;
                    h_feromonas[hasta * N + desde] += aporte;
                }
            }
        }

        if (iter % 10 == 0) cout << "Iteracion " << iter << " | Mejor distancia: " << mejor_distancia_global << endl;
    }

    // --- fin cronometro ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milisegundos = 0;
    cudaEventElapsedTime(&milisegundos, start, stop);

    cout << "\n--- RESULTADO FINAL - CUDA ---" << endl;
    cout << "Distancia minima encontrada: " << mejor_distancia_global << endl;
    cout << "Tiempo de ejecucion en GPU: " << milisegundos << " ms" << endl;
    cout << "Ruta: ";
    for (int nodo : mejor_recorrido_global) cout << nodo << " ";
    cout << endl;

    // limpieza de memoria de video
    cudaFree(d_grafo);
    cudaFree(d_feromonas);
    cudaFree(d_colonia);
    cudaFree(d_state);

    return 0;
}