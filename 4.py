import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw(board):
    for i in range(0, 7, 3):
        print(f' {board[i]} | {board[i+1]} | {board[i+2]} ')
        if i < 6:
            print('---|---|---')

def check_win(board):
    for i in range(0, 7, 3):
        if board[i] == board[i+1] == board[i+2] != ' ':
            return board[i]
    for i in range(3):
        if board[i] == board[i+3] == board[i+6] != ' ':
            return board[i]
    if board[0] == board[4] == board[8] != ' ' or board[2] == board[4] == board[6] != ' ':
        return board[4]
    return None

def take_input(board, turn):
    while True:
        try:
            pos = int(input(f"{turn}'s turn. Pick a position (1-9): ")) - 1
            if 0 <= pos < 9 and board[pos] == ' ':
                board[pos] = turn
                return 'O' if turn == 'X' else 'X'
            print("Invalid position. Try again.")
        except ValueError:
            print("Enter a valid number.")

def tic_tac_toe():
    board = [' '] * 9
    
    turn, winner, moves_left = 'X', None, 9

    while not winner and moves_left:
        clear_console()
        draw(board)
        turn = take_input(board, turn)
        winner = check_win(board)
        moves_left -= 1

    clear_console()
    draw(board)
    print(f"{winner} wins!" if winner else "It's a draw!")

if __name__ == "__main__":
    tic_tac_toe()
