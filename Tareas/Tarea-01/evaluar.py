
import numpy as np
import pandas as pd
import math


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


'''
Genera un programa que te permita evaluar las funciones del ejercicio anterior. El programa debe poder
ejecutarse desde consola, y recibir todos los par ́ametros al momento de la ejecuci ́on. La funci ́on a evaluar
puede pasarse como un n ́umero o una cadena, para elegir alguna de las descritas anteriormente.

✪ “evaluar” es el nombre del ejecutable
✪ “sphere” es el nombre de la funcion que queremos evaluar (de manera alternativa, pueden implementarlo
como un par ́ametro num ́erico, en cuyo caso, la linea de ejecucion seria algo como: $evaluar 1 2 2.1 − 0.1
[el 1 es un numero arbitrario, que debe definirse en el reporte para especificar que con ese numero
indicaremos la ejecucion de la funcion sphere ]
✪ 2 es la dimension del problema
✪ Despues de la dimension, siguen n numeros, correspondientes a los valores de xi

Ejemplo de ejecucion:
$ evaluar sphere 2 2.1 -0.1


'''


def evaluar(funcion_objetivo, dimension, *args):
    if funcion_objetivo == "sum_of_powers":
        print("Funcion Objetivo: sum_of_powers")
        print("Dimension del problema: ", dimension)
        print("Evaluacion: ", sum_of_powers(args))
    elif funcion_objetivo == "zakharov":
        print("Funcion Objetivo: zakharov")
        print("Dimension del problema: ", dimension)
        print("Evaluacion: ", zakharov(args))
    elif funcion_objetivo == "schwefel":
        print("Funcion Objetivo: schwefel")
        print("Dimension del problema: ", dimension)
        print("Evaluacion: ", schwefel(args))
    else:
        print("Funcion Objetivo no valida")


if __name__ == "__main__":
    import sys
    evaluar(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))

# Ejemplo de ejecucion:
# python evaluar sum_of_powers 2 2.1 -0.1
# python evaluar zakharov 2 2.1 -0.1
# python evaluar schwefel 2 2.1 -0.1
        
