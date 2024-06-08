import genetico
from genetico import *

import tkinter as tk
from tkinter import messagebox
#import customtkinter as ctk
#from customtkinter import *

#set_appearance_mode('dark')

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
    hidden_size1 = 10
    hidden_size2 = 10
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



class PlayerSelectionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Player Selection")

        self.label = tk.Label(master, text="Choose your player:")
        self.label.pack()

        self.player_var = tk.StringVar(value="x")
        self.radio_x = tk.Radiobutton(master, text="X", variable=self.player_var, value="x")
        self.radio_x.pack()
        self.radio_o = tk.Radiobutton(master, text="O", variable=self.player_var, value="o")
        self.radio_o.pack()

        self.confirm_button = tk.Button(master, text="Start Game", command=self.start_game)
        self.confirm_button.pack()

    def start_game(self):
        player_choice = self.player_var.get()
        self.master.destroy()
        root = tk.Tk()
        app = TicTacToeGUI(root, player_choice)
        root.mainloop()

class TicTacToeGUI:
    def __init__(self, master, player_choice):
        self.master = master
        self.master.title("Tic Tac Toe")

        self.board = [' ']*9
        self.player = player_choice

        self.buttons = []
        for i in range(3):
            for j in range(3):
                btn = tk.Button(master, text=' ', font=('Arial', 20), width=6, height=3,
                                command=lambda row=i, col=j: self.on_click(row, col))
                btn.grid(row=i, column=j)
                self.buttons.append(btn)

        self.model = torch.load('genetic_model.pth')
        self.model.eval()

        if self.player == 'o':
            self.make_computer_move()

    def on_click(self, row, col):
        index = row * 3 + col
        if self.board[index] == ' ':
            self.board[index] = self.player
            self.buttons[index].config(text=self.player)
            winner = check_winner(self.board)
            if winner != ' ':
                messagebox.showinfo("Winner", f"{winner} wins!")
                self.reset_board()
            elif ' ' not in self.board:
                messagebox.showinfo("Draw", "It's a draw!")
                self.reset_board()
            else:
                self.player = 'o' if self.player == 'x' else 'x'
                if self.player == 'o':
                    self.make_computer_move()

    def make_computer_move(self):
        position = make_decision(self.board)
        self.board[position] = self.player
        self.buttons[position].config(text=self.player)
        winner = check_winner(self.board)
        if winner != ' ':
            messagebox.showinfo("Winner", f"{winner} wins!")
            self.reset_board()
        elif ' ' not in self.board:
            messagebox.showinfo("Draw", "It's a draw!")
            self.reset_board()
        else:
            self.player = 'x'

    def reset_board(self):
        self.board = [' ']*9
        self.player = 'x'
        for btn in self.buttons:
            btn.config(text=' ')


def main():
    root = tk.Tk()
    app = PlayerSelectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()