<div align="center">

# 🦾🤖 **Proyecto Final : Entrenamiento de redes neuronales con algoritmos genéticos​** 📟🧪

-------

## **Integrantes del equipo:**

### <br> <img src="https://media.tenor.com/m6cM9lV-doYAAAAi/batman-batman-beyond.gif" width="30"> **Carlos Emilio Castañon Maldonado** & **Dana Berenice Hérnandez Norberto** <img src="https://i.pinimg.com/originals/c2/00/92/c2009226c462e1fe82a19ca7cd206d1c.gif" width="30"> <br>



</div>


<div align="center">

[![](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExZWkwYml2cDgyOXQ4c2N5d3B0eHR4bXpoN3VhMm5hZGg1a3UwbDRqdCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/gR92EF4p9XyEHyD2n5/giphy.gif)](https://youtu.be/ABzh6hTYpb8?t=3)

</div>

---

## **Requerimientos**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)

Para la presente implementacion se contemplaron las bibliotecas adicionales de pytorch, scikit-learn, numpy, pandas, matplotlib y seaborn en caso de no tenerlas instaladas, ejecutar:

[Pytorch](https://pytorch.org/)

```C
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

[Scikit-learn](https://scikit-learn.org/stable/install.html)

```C
> pip install -U scikit-learn
```

[Numpy](https://numpy.org/install/)

```C
> pip install numpy
```

[Pandas](https://pandas.pydata.org/getting_started.html)

```C
> pip install pandas
```

[Matplotlib](https://matplotlib.org/)

```C
> pip install matplotlib
```

[Seaborn](https://seaborn.pydata.org/installing.html)

```C
> pip install seaborn
```

En caso de querer correr nuestra version del entrenamiento en el notebook.

Es importante recordar tambien que debemos asegurarnos de que tenemos instalado [Jupyter](https://jupyter.org/install).

```C
> pip install jupyterlab
```

```C
> pip install notebook
```
------

## **Contenido**

```julia
    - Genético.ipynb 
    - genetic_model.pth
    - genetico.py
    - game.py
    - /tic+tac+toe+endgame (data-set)
    - /styles
```


[Genético.ipynb](./src/Genético.ipynb) contiene el codigo necesario para poder generar una red neuronal genetica entrenada, [genetic_model.pth](./src/genetic_model.pth) contiene los datos
resultantes del entrenamiento hecho con el notebook anterior, [genetico.py](./src/genetico.py) contiene toda la informacion del notebook pero en python nativo (ademas de que al ejecutar
este script no entrenaremos otra red, el proposito de este archivo es el de exportar todo nuestro codigo a game.py), [game.py](./src/game.py) contiene todo el codigo necesario 
para cargar la red entrenada y poder usarla en un juego de gato.


---

## **Preliminares**

El presente proyecto se divide en dos, el notebook en el que definimos la arquitectura y entrenamiento de la red neuronal genetica [Genético.ipynb](./src/Genético.ipynb) y 
el programa [game.py](./src/game.py) que a raiz de los resultados obtenidos en el notebook anterior, este nos permite jugar partidas ilimitadas de Gato (tic-tac-toe) contra nuestra
red neuronal genetica.

Nota, nosotros ya hemos entrenado a la red y hemos guardado los datos de la red entrenada en [genetic_model.pth](./src/genetic_model.pth), es por esto que podemos correr [game.py](./src/game.py)
sin problemas, ya que no necesitamos entrenar a la red cada que queramos jugar con ella, ya que de eso se encarga [genetic_model.pth](./src/genetic_model.pth) y el siguiente fragmento de codigo
en el que cargamos el modelo y despues instanciamos una red genetica con esos datos para nuestro modelo (todo esto desde [game.py](./src/game.py) ).

```java
# Cargamos el modelo
model = torch.load('genetic_model.pth')
model.eval()
...
    # Definir las dimensiones del modelo
    input_size = board_tensor.shape[0]
    hidden_size1 = 10
    hidden_size2 = 10
    output_size = 1

    # Crear una nueva instancia del modelo
    model = Gen_net(input_size, hidden_size1, hidden_size2, output_size)
```

---

## **Uso**

Para correr el notebook que implementa la arquitectura, entrenamiento y optimizacion de una red neuronal genetica, se debe abrir el Jupyter Notebook en algun editor (como Jupyter nativo, VS Code, etc.).

[Genético.ipynb](./src/Genético.ipynb)

Para correr el programa con el que nos enfrentamos a la red neuronal genetica entrenada, debemos correr (dependiendo el Sistema Operativo) alguno de los siguientes comandos:

Nota, ya tenemos entrenado el modelo, no es necesario ejecutar [Genético.ipynb](./src/Genético.ipynb) a menos que se desee experimentar entrenar la red con otros hiperparametros.


Linux  : 

```Haskell
\Proyecto> python3 game.py
```

Windows:  

```Python
\Proyecto> python game.py
```

Una vez ejecutado alguno de los anteriores, tendremos:

```Python
\Proyecto> python game.py
Device:  cpu
Bienvenido al juego de Gato, elige una de las siguientes opciones:
1 - Jugar como primer jugador - x
2 - Jugar como segundo jugador - o
3 - Salir
Ingresa el número de la opción:
```

## **Ejemplo de uso**

Haremos el siguiente ejemplo en la modalidad en la que la red neuronal primero hace su movimiento y despues nosotros (la opcion 2).

```C
.................    
  0  |  1  |  2  
.................    
  3  |  4  |  5  
.................    
  6  |  7  |  8  
.................    
```

```Python
\Proyecto> python game.py
Device:  cpu
Bienvenido al juego de Gato, elige una de las siguientes opciones:
1 - Jugar como primer jugador - x
2 - Jugar como segundo jugador - o
3 - Salir
Ingresa el número de la opción: 2
x |   |
---------
  |   |
---------
  |   |

Ingresa una posicion entre (0-8): 3
x |   |
---------
o |   |
---------
  |   |

x | x |
---------
o |   |
---------
  |   |

Ingresa una posicion entre (0-8): 2
x | x | o
---------
o |   |
---------
  |   |

x | x | o
---------
o | x |
---------
  |   |

Ingresa una posicion entre (0-8): 8
x | x | o
---------
o | x |
---------
  |   | o

x | x | o
---------
o | x | x
---------
  |   | o

Ingresa una posicion entre (0-8): 7
x | x | o
---------
o | x | x
---------
  | o | o

```

Notemos como es que la red neuronal genetica aprendio demasiado bien a jugar Gato, sin embargo aunque la red nos acorralo en este ejemplo, esta opto mejor por
bloquear una posible victoria nuestra en vez de tirar la `x` en la posicion `7` para que pudiera ganar, lo cual esto derivo en que terminaramos en un empate.




