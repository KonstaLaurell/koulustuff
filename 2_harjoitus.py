import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# ChessNet definition goes here (same as before)...
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8*8*128, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 8*8*128)
        x = torch.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))
# Load model
model = ChessNet()
model_path = "model.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

def board_to_tensor(board):
    # Convert the board state to tensor (same as before)...
    pass

def evaluate_position(board):
    tensor = board_to_tensor(board)
    with torch.no_grad():
        return model(tensor).item()

def mcts_search(board, playouts=1000):
    wins = {}
    plays = {}
    
    for _ in range(playouts):
        visited_states = set()
        current_board = board.copy()
        
        while not current_board.is_game_over():
            legal_moves = list(current_board.legal_moves)
            
            if tuple(current_board.board_fen()) not in plays:
                break
            
            move_visits = [
                plays.get((tuple(current_board.board_fen()), move.uci()), 0)
                for move in legal_moves
            ]
            
            if sum(move_visits) == 0:
                move = random.choice(legal_moves)
            else:
                move_scores = [
                    (wins[(tuple(current_board.board_fen()), move.uci())] / plays[(tuple(current_board.board_fen()), move.uci())])
                    for move in legal_moves
                ]
                move = legal_moves[move_scores.index(max(move_scores))]
                
            current_board.push(move)
            visited_states.add((tuple(current_board.board_fen()), move.uci()))
        
        winner = evaluate_position(current_board)
        
        for state in visited_states:
            plays[state] = plays.get(state, 0) + 1
            wins[state] = wins.get(state, 0) + winner
    
    legal_moves = list(board.legal_moves)
    move_scores = [(wins[(tuple(board.board_fen()), move.uci())] / plays[(tuple(board.board_fen()), move.uci())])
                   if (tuple(board.board_fen()), move.uci()) in plays else 0 for move in legal_moves]
    
    best_move = legal_moves[move_scores.index(max(move_scores))]
    return best_move

def play():
    board = chess.Board()
    
    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            move = input("Your move as WHITE (e.g. e2e4): ")
        else:
            print("AI thinking...")
            move = mcts_search(board).uci()
            print(f"AI plays: {move}")
        board.push(chess.Move.from_uci(move))
        
    print("Game Over.")
    print(board.result())

if __name__ == "__main__":
    play()
