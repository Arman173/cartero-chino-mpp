#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

// parametros
#define NUM_HORMIGAS 64
#define MAX_ITERACIONES 2000
#define ALPHA 2.2
#define BETA 1.9
#define EVAPORACION 0.7
#define Q 1000.0
#define PENALIZACION_B 1000.0

// estructura para la memoria de nuestra hormiga
struct Hormiga {
    int nodo_inicial;
    int nodo_actual;
    float distancia_total;
    int aristas_unicas_visitadas;
    vector<vector<int>> aristas_visitadas;
    vector<int> recorrido;
};

int main(int argc, char* argv[]) {

    // verificamos que se haya mandado la ruta
    if (argc <= 1) {
        cout << "No se paso la direccion del archivo" << endl;
        return 1;
    }

    ifstream file(argv[1]);
    if (!file.is_open()) {
        cout << "Error al abrir el archivo " << argv[1] << endl;
        return 1;
    }

    int N = 0;
    file >> N;
    
    // generamos el grafo y contamos aristas totales
    vector<vector<float>> grafo(N, vector<float>(N));
    int total_aristas_grafo = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            file >> grafo[i][j];
            if (i < j && grafo[i][j] > 0) {
                total_aristas_grafo++;
            }
        }
    }
    file.close();

    // matriz de feromonas
    vector<vector<float>> feromonas(N, vector<float>(N, 0.1f));

    srand(time(NULL));

    float mejor_distancia_global = 9999999.0f;
    vector<int> mejor_recorrido_global;

    auto start_time = chrono::high_resolution_clock::now();

    // ciclo principal del algoritmo
    for (int iter = 0; iter < MAX_ITERACIONES; iter++) {
        vector<Hormiga> colonia(NUM_HORMIGAS);

        // inicializamos a las hormigas
        for (int h = 0; h < NUM_HORMIGAS; h++) {
            colonia[h].nodo_inicial = rand() % N; // seleccion aleatoria
            colonia[h].nodo_actual = colonia[h].nodo_inicial;
            colonia[h].distancia_total = 0;
            colonia[h].aristas_unicas_visitadas = 0;
            colonia[h].aristas_visitadas = vector<vector<int>>(N, vector<int>(N, 0));
            colonia[h].recorrido.push_back(colonia[h].nodo_inicial);
        }

        // construccion de soluciones
        for (int h = 0; h < NUM_HORMIGAS; h++) {
            Hormiga& hormiga = colonia[h];

            // condicion: visitar todas las calles Y regresar al inicio
            while (hormiga.aristas_unicas_visitadas < total_aristas_grafo || hormiga.nodo_actual != hormiga.nodo_inicial) {
                int i = hormiga.nodo_actual;
                vector<float> probabilidades(N, 0.0f);
                float suma_probabilidades = 0.0f;

                // calculamos probabilidad para cada vecino
                for (int j = 0; j < N; j++) {
                    if (grafo[i][j] > 0) { // si hay calle
                        float tau = pow(feromonas[i][j], ALPHA);
                        
                        // aplicamos penalizacion por repeticion
                        float veces_visitada = hormiga.aristas_visitadas[i][j];
                        float castigo = 1.0f + (veces_visitada * PENALIZACION_B);
                        float eta = pow(1.0f / (grafo[i][j] * castigo), BETA);

                        probabilidades[j] = tau * eta;
                        suma_probabilidades += probabilidades[j];
                    }
                }

                // seleccion por ruleta
                float aleatorio_ruleta = (float)rand() / RAND_MAX;
                float limite = aleatorio_ruleta * suma_probabilidades;
                float acumulado = 0.0f;
                int siguiente_nodo = -1;

                for (int j = 0; j < N; j++) {
                    if (probabilidades[j] > 0) {
                        acumulado += probabilidades[j];
                        if (acumulado >= limite) {
                            siguiente_nodo = j;
                            break;
                        }
                    }
                }

                // por si hay problemas de precision de flotantes
                if (siguiente_nodo == -1) siguiente_nodo = i; 

                // actualizamos la memoria de la hormiga
                if (hormiga.aristas_visitadas[i][siguiente_nodo] == 0) {
                    hormiga.aristas_unicas_visitadas++; // descubrimos calle nueva
                }
                
                // marcamos de ida y vuelta porque el grafo es no dirigido
                hormiga.aristas_visitadas[i][siguiente_nodo]++;
                hormiga.aristas_visitadas[siguiente_nodo][i]++;
                
                hormiga.distancia_total += grafo[i][siguiente_nodo];
                hormiga.nodo_actual = siguiente_nodo;
                hormiga.recorrido.push_back(siguiente_nodo);
            }

            if (hormiga.distancia_total < mejor_distancia_global) {
                mejor_distancia_global = hormiga.distancia_total;
                mejor_recorrido_global = hormiga.recorrido;
            }
        }

        // evaporacion de feromonas
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                feromonas[i][j] *= (1.0f - EVAPORACION);
            }
        }

        // deposito de nuevas feromonas
        for (int h = 0; h < NUM_HORMIGAS; h++) {
            float aporte = Q / colonia[h].distancia_total;
            for (size_t k = 0; k < colonia[h].recorrido.size() - 1; k++) {
                int desde = colonia[h].recorrido[k];
                int hasta = colonia[h].recorrido[k+1];
                feromonas[desde][hasta] += aporte;
                feromonas[hasta][desde] += aporte;
            }
        }

        if (iter % 10 == 0) {
            cout << "Iteracion " << iter << " | Mejor distancia: " << mejor_distancia_global << endl;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> tiempo_ejecucion = end_time - start_time;

    // output
    cout << "\n--- RESULTADO FINAL ---" << endl;
    cout << "Distancia minima encontrada: " << mejor_distancia_global << endl;
    cout << "Tiempo de ejecucion: " << tiempo_ejecucion.count() << " ms" << endl;
    cout << "Ruta: ";
    for (int nodo : mejor_recorrido_global) {
        cout << nodo << " ";
    }
    cout << endl;

    return 0;
}