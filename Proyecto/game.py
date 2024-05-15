import genetico
from genetico import *

# Usamos al modelo para jugar tic-tac-toe con el modelo entrenado

# Cargamos el modelo
model = torch.load('genetic_model.pth')
model.eval()


# Función para tomar una decisión basada en el estado del tablero
def make_decision(board):
    # Convertir el tablero a una lista de números
    board_numeric = [1 if cell == 'x' else (2 if cell == 'o' else 0) for cell in board]
    # Convertir el tablero numérico en un tensor
    board_tensor = torch.tensor(board_numeric, dtype=torch.float32)

    # Definir las dimensiones del modelo
    input_size = board_tensor.shape[0]
    hidden_size1 = 40
    hidden_size2 = 40
    output_size = 1

    # Crear una nueva instancia del modelo
    model = Gen_net(input_size, hidden_size1, hidden_size2, output_size)

    # Pasar el tablero a través del modelo
    output = model(board_tensor.view(1, -1))  

    # Obtenemos la lista de las casillas vacías
    empty_cells = [i for i, cell in enumerate(board) if cell == ' ']

    # decision = torch.argmax(output).item()
    # Tomar la decisión basada en la salida del modelo
    # En caso de que esa casilla no forme parte de las casillas vacías, se toma la primera casilla vacía
    # Que el modelo haya decidido usar
    decision = empty_cells[output.argmax().item()] if output.argmax().item() in empty_cells else empty_cells[0]
    

    return decision



# Función para representar el tablero
def print_board(board):
    for i in range(3):
        print(' | '.join(board[i*3:(i+1)*3]))
        if i < 2:
            print('---------')

# Función para determinar si alguien ganó
def check_winner(board):
    # Revisamos las filas
    for i in range(3):
        if board[i*3] == board[i*3 + 1] == board[i*3 + 2] != ' ':
            return board[i*3]
    # Revisamos las columnas
    for i in range(3):
        if board[i] == board[i + 3] == board[i + 6] != ' ':
            return board[i]
    # Revisamos las diagonales
    if board[0] == board[4] == board[8] != ' ':
        return board[0]
    if board[2] == board[4] == board[6] != ' ':
        return board[2]
    return ' '

# Función para jugar, toma como primer jugador al usuario y como segundo jugador a el modelo
def play_game_A():
    # Inicializar el tablero
    board = [' ']*9
    # Inicializar el jugador
    player = 'x'
    # Jugar hasta que alguien gane o haya un empate
    while ' ' in board:
        print_board(board)
        print()
        # Si es el turno del jugador
        if player == 'x':
            # Pedir la posición al jugador
            position = int(input('Ingresa una posicion entre (0-8): '))
            # Si la posición está vacía
            if board[position] == ' ':
                # Colocar la marca
                board[position] = player
                # Revisar si alguien ganó
                winner = check_winner(board)
                if winner != ' ':
                    print_board(board)
                    print(f'{winner} Gano!')
                    return
                # Cambiar de jugador
                player = 'o'
        # Si es el turno de la computadora
        else:
            # Tomar la decisión de la computadora
            position = make_decision(board)
            # Colocar la marca
            board[position] = player
            # Revisar si alguien ganó
            winner = check_winner(board)
            if winner != ' ':
                print_board(board)
                print(f'{winner} Gano!')
                return
            # Cambiar de jugador
            player = 'x'
    # Si no hay ganador, es un empate
    print_board(board)
    print('Tenemos un empate!')


# Función para jugar, toma como primer jugador al modelo y como segundo jugador al usuario
def play_game_B():
    # Inicializar el tablero
    board = [' ']*9

    playerB = 'x'
    # Tomar la decisión de la computadora
    position = make_decision(board)
    # Colocar la marca
    board[position] = playerB
    playerB = 'o'
    # Jugar hasta que alguien gane o haya un empate, el primer movimiento (x) lo hace la computadora

    while ' ' in board:
        print_board(board)
        print()
        # Si es el turno de la computadora
        if playerB != 'o':
            # Tomar la decisión de la computadora
            position = make_decision(board)
            # Colocar la marca
            board[position] = playerB
            # Revisar si alguien ganó
            winner = check_winner(board)
            if winner != ' ':
                print_board(board)
                print(f'{winner} Gano!')
                return
            # Cambiar de jugador
            playerB = 'o'
        # Si es el turno de el jugador
        else:
            # Pedir la posición al jugador
            position = int(input('Ingresa una posicion entre (0-8): '))
            # Si la posición está vacía
            if board[position] == ' ':
                # Colocar la marca
                board[position] = playerB
                # Revisar si alguien ganó
                winner = check_winner(board)
                if winner != ' ':
                    print_board(board)
                    print(f'{winner} Gano!')
                    return
                # Cambiar de jugador
                playerB = 'x'



# Jugar
#play_game_A()
#play_game_B()


if __name__ == "__main__":
    while True:
        print("Bienvenido al juego de Gato, elige una de las siguientes opciones:")
        print("1 - Jugar como primer jugador - x")
        print("2 - Jugar como segundo jugador - o")
        print("3 - Salir")
        
        option = input("Ingresa el número de la opción: ")
        
        if option == "1":
            play_game_A()
        elif option == "2":
            play_game_B()
        elif option == "3":
            print("Adios! =D")
            break
        else:
            print("Opción no válida, debes ingresar un número del 1 al 3 para elegir una opción válida.")