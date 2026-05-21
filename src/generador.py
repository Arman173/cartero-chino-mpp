import random
import sys

def generar_grafo_cpp(n, densidad, peso_maximo, nombre_archivo):
    # 1. Inicializar matriz vacía (llena de ceros)
    matriz = [[0 for _ in range(n)] for _ in range(n)]

    # 2. Garantizar conexidad (crear un anillo base que conecte todos los nodos)
    for i in range(n):
        siguiente = (i + 1) % n
        peso = random.randint(1, peso_maximo)
        matriz[i][siguiente] = peso
        matriz[siguiente][i] = peso

    # 3. Rellenar el resto basándonos en la probabilidad (densidad)
    for i in range(n):
        for j in range(i + 1, n):
            if matriz[i][j] == 0:  # Si no hay calle aún
                # Si el número aleatorio cae dentro de nuestra densidad, creamos la calle
                if random.random() < densidad:
                    peso = random.randint(1, peso_maximo)
                    matriz[i][j] = peso
                    matriz[j][i] = peso

    # 4. Guardar en el formato estricto que espera tu código en C++/CUDA
    with open(nombre_archivo, 'w') as f:
        f.write(f"{n}\n")
        for fila in matriz:
            f.write(" ".join(map(str, fila)) + "\n")

    print(f"✅ Grafo generado exitosamente en: '{nombre_archivo}'")
    print(f"   -> Nodos: {n}")
    print(f"   -> Densidad extra: {int(densidad * 100)}%")
    print(f"   -> Peso máximo por arista: {peso_maximo}")

if __name__ == "__main__":
    # --- PARÁMETROS DE CONFIGURACIÓN ---
    # ¡Ajusta estos valores para crear tus diferentes pruebas!
    
    N_NODOS = 40           # Coincide con tu MAX_N actual en CUDA
    DENSIDAD = 0.40        # 40% de probabilidad de crear calles extra (grafo poblado)
    PESO_MAXIMO = 15       # Las calles medirán entre 1 y 20 km
    ARCHIVO_SALIDA = "matriz_50_masiva.txt"

    generar_grafo_cpp(N_NODOS, DENSIDAD, PESO_MAXIMO, ARCHIVO_SALIDA)