# Programa que realiza la Busqueda Aleatoria para problemas de optimizacion continua.

import numpy as np
import pandas as pd
import math

'''
El programa deberá recibir como parametros :

a) Funcion Objetivo (Esto puede pasarse como un numero o una cadena para elegir entre
sum_of_powers, zakharov o schwefel)
b) Dimension que se utilizara para definir el espacio de busqueda correspondiente
c) Numero Total de iteraciones a realizar
d) Intervalo de busqueda

El programa deberá devolver e imprimir, el resultado de la busqueda imprimiendo el valor de x
(la mejor solucion encontrada) asi como su evaluacion.

Ejemplo de ejecucion:

$ busqueda_aleatoria.py sphere 2 1000 [-5.12,5.12]  
Funcion Objetivo: sphere
Dimension del problema: 2
Total de iteraciones: 1000
Mejor Solucion encontrada:
x = [0.1 1.2]
f(x) = 300.86

'''


# Funciones Objetivo

def sum_of_powers(x):
    x = pd.Series(x)
    d = len(x)
    y = sum(abs(x)**(x+1))
    return y


def zakharov(x):
    x = pd.Series(x)
    d = len(x)
    t1 = sum(x**2)
    t2 = 0.5*sum(range(d)*(x**2))
    y = t1 + (t2**2) + (t2**4)
    return y

def schwefel(x):
    x = pd.Series(x)
    d = len(x)
    t1 = 418.9829*d
    t2 = sum(x*np.sin(abs(x)**(1/2)))
    y = t1-t2
    return y

# Funcion de Busqueda Aleatoria
def busqueda_aleatoria(funcion_objetivo, dimension, iteraciones, intervalo):
    mejor_solucion = np.random.uniform(intervalo[0], intervalo[1], dimension)
    mejor_evaluacion = funcion_objetivo(mejor_solucion)
    for i in range(iteraciones):
        solucion = np.random.uniform(intervalo[0], intervalo[1], dimension)
        evaluacion = funcion_objetivo(solucion)
        if evaluacion < mejor_evaluacion:
            mejor_solucion = solucion
            mejor_evaluacion = evaluacion
    print('Funcion Objetivo:', funcion_objetivo.__name__)
    print('Dimension del problema:', dimension)
    print('Total de iteraciones:', iteraciones)
    print('Mejor Solucion encontrada:')
    print('x =', mejor_solucion)
    print('f(x) =', mejor_evaluacion)

# Funcion main para leer la entrada por consola de comandos
# Ejemplos de uso con las funciones objetivo: sum_of_powers, zakharov, schwefel
    
# Ejemplo de uso: python busqueda_aleatoria.py sum_of_powers 2 1000 [-5.12,5.12]
# Ejemplo de uso: python busqueda_aleatoria.py zakharov 2 1000 [-5.12,5.12]
# Ejemplo de uso: python busqueda_aleatoria.py schwefel 2 1000 [-5.12,5.12]
 
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print('Error: Numero de argumentos invalido')
        print('Uso: busqueda_aleatoria.py <funcion_objetivo> <dimension> <iteraciones> <intervalo>')
        #print('Ejemplo: busqueda_aleatoria.py sphere 2 1000 [-5.12,5.12]')
        sys.exit(1)
    funcion_objetivo = sys.argv[1]
    dimension = int(sys.argv[2])
    iteraciones = int(sys.argv[3])
    intervalo = eval(sys.argv[4])
    if funcion_objetivo == 'sum_of_powers':
        funcion_objetivo = sum_of_powers
    elif funcion_objetivo == 'zakharov':
        funcion_objetivo = zakharov
    elif funcion_objetivo == 'schwefel':
        funcion_objetivo = schwefel
    else:
        print('Error: Funcion Objetivo invalida')
        sys.exit(1)
    busqueda_aleatoria(funcion_objetivo, dimension, iteraciones, intervalo)    

