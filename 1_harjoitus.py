import chess
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import threading

# Neural Network
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(64 * 12, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.layers(x)

# Board to Tensor
def board_to_tensor(board):
    tensor = torch.zeros(12, 64)
    piece_dict = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }
    
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            tensor[piece_dict[str(piece)], i] = 1
    return tensor.view(-1)

# Basic MCTS (Placeholder)
def mcts(board, model, depth=3):
    return random.choice(list(board.legal_moves))

# Wrapper for the MCTS search that will be used in a thread
def mcts_thread(board, model, best_move, depth=3):
    best_move[0] = mcts(board, model, depth)

# Self-play loop with timeout mechanism
def self_play(model, num_games=100, think_time=2.0):
    data = []
    outcomes = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for _ in range(num_games):
        board = chess.Board()
        game_data = []
        while not board.is_game_over():
            best_move = [None]
            search_thread = threading.Thread(target=mcts_thread, args=(board, model, best_move))
            search_thread.start()
            search_thread.join(timeout=think_time)
            if search_thread.is_alive():
                search_thread.join()
            
            if best_move[0] is not None:
                board.push(best_move[0])

        result = board.result()
        outcomes[result] += 1
        for move in board.move_stack:
            board.pop()
            game_data.append((board.copy(), result))
        data.extend(game_data[::-1])
    return data, outcomes

# Training Loop
def train(model, data, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for board, result in data:
            tensor_input = board_to_tensor(board)
            optimizer.zero_grad()
            outputs = model(tensor_input)
            result_value = 1 if result == "1-0" else (-1 if result == "0-1" else 0)
            loss = criterion(outputs, torch.tensor([result_value], dtype=torch.float32))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        average_loss = total_loss / len(data)
        epoch_losses.append(average_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}")
    return epoch_losses

# Save and Load Model
def save_model(model, file_path='chess_model.pth'):
    torch.save(model.state_dict(), file_path)

def load_model(file_path='chess_model.pth'):
    model = ChessNet()
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

# Main
if __name__ == "__main__":
    model = ChessNet()
    data, outcomes = self_play(model, think_time=5.0)
    
    # Plotting game outcomes
    labels = list(outcomes.keys())
    values = [outcomes[key] for key in labels]
    plt.bar(labels, values, color=['white', 'gray', 'black'])
    plt.title("Game Outcomes")
    plt.ylabel("Number of Games")
    plt.show()
    
    # Training and plotting loss
    losses = train(model, data)
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()

    # Save the trained model
    save_model(model)
