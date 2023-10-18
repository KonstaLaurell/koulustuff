import torch
import torch.nn as nn
import torch.optim as optim
import chess
import numpy as np

# Neural Network for Policy and Value Estimation
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2*8*8, 4672)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        policy = torch.softmax(self.policy_head(x), dim=1)
        value = self.value_head(x)
        return policy, value

def board_to_input(board):
    labels = [
        chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING,
        -chess.PAWN, -chess.ROOK, -chess.KNIGHT, -chess.BISHOP, -chess.QUEEN, -chess.KING
    ]
    arr = np.zeros((12, 8, 8), dtype=int)
    for piece, value in zip(labels, range(12)):
        for i in board.pieces(piece, chess.WHITE if piece > 0 else chess.BLACK):
            arr[value][np.unravel_index(i, (8, 8))] = 1
    return torch.tensor(arr).float().unsqueeze(0)

# Training Loop
def self_play_and_train(model, optimizer, episodes=5, mcts_searches=10):
    for episode in range(episodes):
        board = chess.Board()
        value = 0.0
        while not board.is_game_over():
            policy, _ = model(board_to_input(board))
            move = chess.Move.from_uci(chess.SQUARE_NAMES[np.argmax(policy.detach().numpy())])
            if move not in board.legal_moves:
                move = np.random.choice(list(board.legal_moves))
            board.push(move)

            print(board)  # Printing the board after every move

        # Calculate the result of the game
        result = board.result()
        if result == "1-0":
            value = 1
        elif result == "0-1":
            value = -1
        else:
            value = 0  # draw
        
        policy, board_value = model(board_to_input(board))
        optimizer.zero_grad()
        loss = nn.MSELoss()(board_value, torch.tensor([[value]])) - torch.log(policy[0][chess.SQUARE_NAMES.index(move.uci())] + 1e-5)
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}, Value: {board_value.item()}, Loss: {loss.item()}")

# Save Model
def save_model(model, filepath="chess_model.pth"):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

# Load Model
def load_model(filepath="chess_model.pth"):
    model = ChessNet()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model

# Main
model = ChessNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
self_play_and_train(model, optimizer)
save_model(model)
