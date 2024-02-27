import pandas as pd
import random

def leer_ArchivoCol(archivo):
    vertices = set()
    aristas = []

    with open(archivo, 'r') as f:
        for linea in f:
            # Ignorar comentarios
            if linea.startswith('c'):
                continue
            
            # Leer información de la instancia
            if linea.startswith('p'):
                _, _, n_vertices, n_aristas = linea.split()
                n_vertices = int(n_vertices)
                n_aristas = int(n_aristas)
            elif linea.startswith('e'):
                _, v1, v2 = linea.split()
                v1 = int(v1)
                v2 = int(v2)
                aristas.append((v1, v2))
                vertices.add(v1)
                vertices.add(v2)
    
    return n_vertices, n_aristas, list(vertices), aristas

#archivo = 'prueba1.col'
#n_vertices, n_aristas, vertices, aristas = leer_ArchivoCol(archivo)

#print("Número de vértices:", n_vertices)
#print("Número de aristas:", n_aristas)
#print("Vértices:", vertices)
#print("Aristas:", aristas)


class SColoracion:
    def __init__(self, n_vertices, colores_asignados=None):
        self.n_vertices = n_vertices
        if colores_asignados is None:
            self.colores_asignados = [-1] * n_vertices  # Inicialmente ningún vértice tiene color asignado
        else:
            self.colores_asignados = colores_asignados
    
    def asignar_color(self, vertice, color):
        self.colores_asignados[vertice - 1] = color  # Los índices de los vértices empiezan en 1
    
    def obtener_color(self, vertice):
        return self.colores_asignados[vertice - 1]
    
def zakharov(x):
    x = pd.Series(x)
    d = len(x)
    t1 = sum(x**2)
    t2 = 0.5*sum(range(d)*(x**2))
    y = t1 + (t2**2) + (t2**4)
    return y


def generar_solucion_aleatoria(n_vertices, n_colores):
    """
    Genera una solución aleatoria para el problema de coloración de grafos.

    Parámetros:
    n_vertices : int
        Número de vértices en el grafo.
    n_colores : int
        Número de colores disponibles para la coloración.

    Devuelve:
    SColoracion
        Una instancia de la clase SColoracion con una asignación aleatoria de colores a los vértices.
    """
    solucion = SColoracion(n_vertices)
    for vertice in range(1, n_vertices + 1):
        color_asignado = random.randint(1, n_colores)
        solucion.asignar_color(vertice, color_asignado)
    return solucion


def generar_vecino(solucion_actual, n_colores):
    """
    Genera una solución vecina modificando aleatoriamente un color de un vértice.

    Parámetros:
    solucion_actual : SColoracion
        Solución actual que se usará como base para generar la solución vecina.
    n_colores : int
        Número de colores disponibles para la coloración.

    Devuelve:
    SColoracion
        Una instancia de la clase SColoracion que representa una solución vecina.
    """
    # Seleccionar aleatoriamente un vértice y cambiar su color
    vertice_a_cambiar = random.randint(1, solucion_actual.n_vertices)
    color_asignado_actual = solucion_actual.obtener_color(vertice_a_cambiar)
    nuevo_color = color_asignado_actual
    while nuevo_color == color_asignado_actual:
        nuevo_color = random.randint(1, n_colores)
    # Crear una copia de la solución actual y modificar el color del vértice seleccionado
    nueva_solucion = SColoracion(solucion_actual.n_vertices, solucion_actual.colores_asignados[:])
    nueva_solucion.asignar_color(vertice_a_cambiar, nuevo_color)
    return nueva_solucion

def hill_climbing(archivo_instancia, n_colores, max_iter):
    """
    Algoritmo de búsqueda por escalada para el problema de coloración de grafos.

    Parámetros:
    archivo_instancia : str
        Ruta al archivo que contiene la instancia del problema.
    n_colores : int
        Número de colores disponibles para la coloración.
    max_iter : int
        Número máximo de iteraciones permitidas.

    Devuelve:
    SColoracion
        La mejor solución encontrada por el algoritmo.
    """
    # Leer la instancia del problema
    n_vertices, _, _, _ = leer_ArchivoCol(archivo_instancia)

    # Generar una solución aleatoria como punto de partida
    mejor_solucion = generar_solucion_aleatoria(n_vertices, n_colores)
    mejor_valor = zakharov(mejor_solucion.colores_asignados)

    iteracion = 0
    while iteracion < max_iter:
        # Generar un vecino de la solución actual
        vecino = generar_vecino(mejor_solucion, n_colores)
        valor_vecino = zakharov(vecino.colores_asignados)
        
        # Si el vecino es mejor que la solución actual, actualizar la mejor solución
        if valor_vecino < mejor_valor:
            mejor_solucion = vecino
            mejor_valor = valor_vecino

        iteracion += 1

    return mejor_solucion

# Ejemplo de uso del algoritmo de búsqueda por escalada
archivo_instancia = 'prueba1.col'
n_colores = 2
max_iter = 1000
mejor_solucion = hill_climbing(archivo_instancia, n_colores, max_iter)
print("Mejor solución encontrada:", mejor_solucion.colores_asignados)
print("Valor de la función de Zakharov:", zakharov(mejor_solucion.colores_asignados))