import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx

"""
# Definir la lista de colores
# Definir la lista de colores en códigos hexadecimales
colores_hex = ['#0000FF', '#00FFFF', '#000080', '#87CEEB', '#008B8B',
               '#FF0000', '#FF6347', '#8B0000', '#FF4500', '#DC143C',
               '#008000', '#008000', '#228B22', '#32CD32', '#2E8B57',
               '#FFFF00', '#FFFF00', '#FFD700', '#F0E68C', '#FFFFE0',
               '#FF00FF', '#FF00FF', '#800080', '#FF1493', '#FF00FF',
               '#FFA500', '#FF8C00', '#FF7F50', '#FF8C00', '#FF7F50',
               '#000000', '#808080', '#A9A9A9', '#808080', '#C0C0C0',
               '#FFFFFF', '#FFFFFF', '#FFFFF0', '#FFFAFA', '#F8F8FF']

# Crear un grafo vacío
G = nx.Graph()

# Agregar un vértice por cada color
for i, color in enumerate(colores_hex):
    G.add_node(i, color=color)

# Dibujar el grafo
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color=colores_hex, node_size=1000)
nx.draw_networkx_labels(G, pos)
plt.axis('off')
plt.show()
"""

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



def dibujar_grafo(vertices, aristas):
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(aristas)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='cyan', font_size=8, font_weight='bold')
    plt.show()

#archivo = 'prueba1.col'
#n_vertices, n_aristas, vertices, aristas = leer_ArchivoCol(archivo)

#print("Número de vértices:", n_vertices)
#print("Número de aristas:", n_aristas)
#print("Vértices:", vertices)
#print("Aristas:", aristas)

#dibujar_grafo(vertices, aristas)




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
    

# Funcion que mapea un numero entero a un color de los contenidos en nx.color
'''
Colores disponibles en NetworkX:
Azul: 'b', 'blue', 'navy', 'skyblue', 'cyan'
Rojo: 'r', 'red', 'darkred', 'tomato', 'crimson'
Verde: 'g', 'green', 'forestgreen', 'limegreen', 'seagreen'
Amarillo: 'y', 'yellow', 'gold', 'khaki', 'lightyellow'
Morado: 'm', 'magenta', 'purple', 'deeppink', 'fuchsia'
Naranja: 'o', 'orange', 'darkorange', 'burntorange', 'coral'
Negro: 'k', 'black', 'gray', 'darkgray', 'silver'
Blanco: 'w', 'white', 'ivory', 'snow', 'ghostwhite'
'''
def mapear_color(numero):
    colores = ['b', 'blue', 'navy', 'skyblue', 'cyan',
               'r', 'red', 'darkred', 'tomato', 'crimson',
               'g', 'green', 'forestgreen', 'limegreen', 'seagreen',
               'y', 'yellow', 'gold', 'khaki', 'lightyellow',
               'm', 'magenta', 'purple', 'deeppink', 'fuchsia',
               'o', 'orange', 'darkorange', 'burntorange', 'coral',
               'k', 'black', 'gray', 'darkgray', 'silver',
               'w', 'white', 'ivory', 'snow', 'ghostwhite'] 
    return colores[numero % len(colores)]

    
def zakharov(x):
    x = pd.Series(x)
    d = len(x)
    t1 = sum(x**2)
    t2 = 0.5*sum(range(d)*(x**2))
    y = t1 + (t2**2) + (t2**4)
    return y

# funcion auxiliar que devuelve una lista con los colores (en numero) de los vecinos de un vertice
# entrada: Solucion, vertice
# salida: lista con los colores de los vecinos
def coloresVecinos(solucion, vertice):
    vecinos = [v for v1, v2 in aristas if v1 == vertice for v in [v2]] + [v for v1, v2 in aristas if v2 == vertice for v in [v1]]
    colores = [solucion.obtener_color(v) for v in vecinos]
    return colores


# Vamos a recibir un archivo el cual al leer con la funcion leer_ArchivoCol
# generaremos una grafica y apartir de los elementos de esa grafica vamos a
# escoger un vertice aleatorio y asignarle un color aleatorio, despues vamos a 
# revisar quienes son sus vecinos y asignarles un color diferente (aleatorio) diferente al del vertice
# que ya tiene color asignado, vamos a repetir este proceso hasta que todos los
# vertices tengan color asignado
def colorearGraficaConNColores(archivo):
    n_vertices, n_aristas, vertices, aristas = leer_ArchivoCol(archivo)
    dibujar_grafo(vertices, aristas)
    n_colores = n_vertices # Como en el peor de los casos se necesitaran n_vertices colores, vamos a asignarle ese numero de colores
    solucion = SColoracion(n_vertices)
    vertices = list(range(1, n_vertices+1))

    #Ahora ordenamos la lista de vertices con los vertices de mayor a menor, iniciando por aquellos que tienen mas vecinos
    vertices.sort(key=lambda v: len([v1 for v1, v2 in aristas if v1 == v or v2 == v]), reverse=True)

    #vecinos = [v for v1, v2 in aristas if v1 == vertice for v in [v2]] + [v for v1, v2 in aristas if v2 == vertice for v in [v1]]

    for vertice in vertices:
        # Asignar un color aleatorio al vértice y verificar que ese color no esté asignado a ninguno de sus vecinos
        color = (random.randint(1, n_colores))
        while color in coloresVecinos(solucion, vertice):
            color = (random.randint(1, n_colores))
        solucion.asignar_color(vertice, color)
        
        #for vecino in vecinos:
        #    color = (random.randint(1, n_colores))
        #    solucion.asignar_color(vecino, color)
    return solucion

def dibujar_grafo_coloreado(solucion, vertices, aristas):
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(aristas)
    pos = nx.spring_layout(G)
    colores = [mapear_color(solucion.obtener_color(v)) for v in vertices]
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=colores, font_size=8, font_weight='bold')
    plt.show()


#archivo = 'prueba1.col'
#n_vertices, n_aristas, vertices, aristas = leer_ArchivoCol(archivo)

#print("Número de vértices:", n_vertices)
#print("Número de aristas:", n_aristas)
#print("Vértices:", vertices)
#print("Aristas:", aristas)

#dibujar_grafo(vertices, aristas)
#solucion = colorearGraficaConNColores(archivo)
#dibujar_grafo_coloreado(solucion, vertices, aristas)



def generar_vecino(solucion_actual):
    """
    Genera una solución vecina modificando aleatoriamente un color de un vértice.

    Parámetros:
    solucion_actual : SColoracion
        Solución actual que se usará como base para generar la solución vecina.
    Devuelve:
    SColoracion
        Una instancia de la clase SColoracion que representa una solución vecina.
     
    """
    vertice_a_cambiar = random.randint(1, solucion_actual.n_vertices)
    colores_vecinos = coloresVecinos(solucion_actual, vertice_a_cambiar)


    color_asignado_actual = solucion_actual.obtener_color(vertice_a_cambiar)
    nuevo_color = color_asignado_actual
    while nuevo_color == color_asignado_actual & nuevo_color not in colores_vecinos:
        nuevo_color = random.randint(1, solucion_actual.n_vertices)
    
    # Crear una copia de la solución actual y modificar el color del vértice seleccionado
    nueva_solucion = SColoracion(solucion_actual.n_vertices, solucion_actual.colores_asignados[:])
    nueva_solucion.asignar_color(vertice_a_cambiar, nuevo_color)
    return nueva_solucion

archivo = 'prueba1.col'
n_vertices, n_aristas, vertices, aristas = leer_ArchivoCol(archivo)

#print("Número de vértices:", n_vertices)
#print("Número de aristas:", n_aristas)
#print("Vértices:", vertices)
#print("Aristas:", aristas)

dibujar_grafo(vertices, aristas)
solucion = colorearGraficaConNColores(archivo)
solucionVecina = generar_vecino(solucion)
dibujar_grafo_coloreado(solucion, vertices, aristas)
dibujar_grafo_coloreado(solucionVecina, vertices, aristas)


"""
def hill_climbing(archivo_instancia, n_colores, max_iter):
    # 
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
    # 
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
n_colores = 4
max_iter = 1000
mejor_solucion = hill_climbing(archivo_instancia, n_colores, max_iter)
print("Mejor solución encontrada:", mejor_solucion.colores_asignados)
print("Valor de la función de Zakharov:", zakharov(mejor_solucion.colores_asignados))
"""