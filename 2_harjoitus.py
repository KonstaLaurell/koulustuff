import chess
import torch
import torch.nn as nn

# Neural Network Definition
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(8*8*12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        return torch.tanh(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))))

def board_to_tensor(board):
    # Placeholder: This should convert a board to a tensor representation
    # You can replace this with the actual implementation
    return torch.randn(8*8*12)

def mcts_predict_move(board, model):
    legal_moves = list(board.legal_moves)
    best_move = None
    best_value = float('-inf')
    for move in legal_moves:
        board.push(move)
        board_tensor = board_to_tensor(board)
        value = model(board_tensor).item()
        board.pop()
        if value > best_value:
            best_value = value
            best_move = move
    return best_move

def play_against_ai(model_path="chess_model.pth"):
    model = ChessNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    board = chess.Board()
    print(board)

    while not board.is_game_over():
        if board.turn == chess.WHITE:  # Assume the user plays as WHITE
            move_uci = input("Enter your move in UCI format (e.g. e2e4): ")
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
            else:
                print("Illegal move. Try again.")
                continue
        else:  # AI's turn
            print("AI is thinking...")
            move = mcts_predict_move(board, model)
            print(f"AI's move: {move.uci()}")
            board.push(move)

        print(board)

    print("Game Over!")
    print(f"Result: {board.result()}")

if __name__ == "__main__":
    play_against_ai()
