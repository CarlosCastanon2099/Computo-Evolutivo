<div align="center">

# 🎒 **Tarea 01** 📊



# **Problemas de Optimización**


</div>



<div align="center">

[![](http://www.adventuresofyoo.com/wp-content/uploads/2016/12/tumblr_ogm7cjPqBy1t0x810o1_540.gif)](https://www.youtube.com/watch?v=8SbUC-UaAxE)

</div>

---

### **Tarea hecha por:**

```Haskell
\src> Carlos Emilio Castañon Maldonado
```

```Kotlin  
\src> Dana Berenice Hernandez Norberto
```

---

## **Requerimientos**

Para la presente implementacion se contemplaron las bibliotecas de **pandas** y de  **Pyarrow** , en caso de no tenerlas, se
debe correr por terminal los siguientes comandos:

```C
> pip install pandas
```

```C
> pip install Pyarrow
```

---

## **Las Funciones Implementadas en el presente fueron:**

### **Función de Suma Ponderada**

```Python
def sum_of_powers(x):
    x = pd.Series(x)
    d = len(x)
    y = sum(abs(x)**(x+1))
    return y
```

### **Función de Zakharov**

```Python
def zakharov(x):
    x = pd.Series(x)
    d = len(x)
    t1 = sum(x**2)
    t2 = 0.5*sum(range(d)*(x**2))
    y = t1 + (t2**2) + (t2**4)
    return y
```

### **Función de Schwefel**

```Python
def schwefel(x):
    x = pd.Series(x)
    d = len(x)
    t1 = 418.9829*d
    t2 = sum(x*np.sin(abs(x)**(1/2)))
    y = t1-t2
    return y
```

---

## **Uso**

Para correr el programa `evaluar` el cual  realiza la **evaluacion** de las anteriores funciones:
- Correr desde `src/`:

Linux  : 

```Haskell
\src> python3 evaluar.py <funcion> <dimension del problema> <valores de x_i>
```

Windows:  

```Python
\src> python3 evaluar.py <funcion> <dimension del problema> <valores de x_i>
```

En donde:

```Julia
<funcion> = Nombre de la funcion a evaluar, puede ser:
            'sum_of_powers', 'zakharov' o 'schwefel'
```

```Julia
<dimension del problema> = Numero entero de la dimension del problema
```

```Julia
<valores de x_i> = N numeros, correspondientes a los valores de x_i
```


Ejemplos de uso con el programa:

```Python
\src> python3 evaluar.py sum_of_powers 2 2.1 -0.1
Funcion Objetivo: sum_of_powers
Dimension del problema:  2
Evaluacion:  10.100132533838126

\src> python3 evaluar.py zakharov 2 2.1 -0.1
Funcion Objetivo: zakharov
Dimension del problema:  2
Evaluacion:  4.4200250006249995

\src> python3 evaluar.py schwefel 2 2.1 -0.1
Funcion Objetivo: schwefel
Dimension del problema:  2
Evaluacion:  835.9124200696331
```


-------------


Para correr el programa `busqueda_aleatoria` el cual realiza la **búsqueda aleatoria** para problemas de optimización continua:
- Correr desde `src/`:

Linux  : 

```Haskell
\src> python3 busqueda_aleatoria.py <funcion> <dimension del problema> <iteraciones> <intervalo de busqueda>
```

Windows:  

```Python
\src> python busqueda_aleatoria.py <funcion> <dimension del problema> <iteraciones> <intervalo de busqueda>
```

En donde:

```Julia
<funcion> = Nombre de la funcion a evaluar, puede ser:
            'sum_of_powers', 'zakharov' o 'schwefel'
```

```Julia
<dimension del problema> = Numero entero de la dimension del problema
```

```Julia
<iteraciones> = Numero de iteraciones a realizar
```

```Julia
<intervalo de busqueda> = Dupla de numeros en la que se realizará la busqueda
```


Ejemplos de uso con el programa:

```Python
\src> python3 busqueda_aleatoria.py sum_of_powers 2 1000 [-5.12,5.12]
Funcion Objetivo: sum_of_powers
Dimension del problema: 2
Total de iteraciones: 1000
Mejor Solucion encontrada:
x = [-4.59638578 -4.555646  ]
f(x) = 0.008701007303661846

\src> python3 busqueda_aleatoria.py schwefel 2 1000 [-5.12,5.12]
Funcion Objetivo: schwefel
Dimension del problema: 2
Total de iteraciones: 1000
Mejor Solucion encontrada:
x = [4.81525571 4.97287512]
f(x) = 830.1258116824008

\src> python3 busqueda_aleatoria.py zakharov 2 1000 [-5.12,5.12]
Funcion Objetivo: zakharov
Dimension del problema: 2
Total de iteraciones: 1000
Mejor Solucion encontrada:
x = [0.07419368 0.20817246]
f(x) = 0.04931019454632571
```


-------

Para consultar el Notebook en el que se ejecuto la búsqueda aleatoria para todas las funciones anteriores, considerando 1,000,000 
iteraciones en las dimensiones 2, 5 y 10 para saber el mejor caso de f(x), su caso promedio y el peor.

Consultar el Notebook [tablas.ipynb](./src/tablas.ipynb)


|  Funcion Objetivo  |  Dimension   |  Mejor f(x)    |  Promedio f(x)  |   Peor f(x)      |
|--------------------|--------------|----------------|-----------------|------------------|
|    sum_of_powers   |     2        |  2.098934e-09  |  2.978298e+09   |   1.952262e+11   |
|    sum_of_powers   |     5        |  4.351890e-08  |  7.384374e+09   |   2.500835e+11   |
|    sum_of_powers   |    10        |  1.212787e-03  |  1.481389e+10   |   2.852122e+11   |
|         zakharov   |     2        |  0.000082      |  6.977483e+05   |   6.252611e+06   |
|         zakharov   |     5        |  1.620490      |  2.145085e+09   |   5.364599e+10   |
|         zakharov   |    10        |  692966.098116 |  5.540794e+11   |   1.296510e+13   |
|         schwefel   |     2        |  830.075215    |  837.969796     |   845.856388     |
|         schwefel   |     5        |  2075.331918   |  2094.919634    |   2114.534886    |
|         schwefel   |    10        |  4155.145285   |  4189.813929    |   4224.956487    |

-------
