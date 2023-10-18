import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import threading
import os

# ChessNet Neural Network
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        print("Initializing ChessNet...")
        self.fc1 = nn.Linear(8*8*12, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        print("ChessNet initialized.")

    def forward(self, x):
        return torch.tanh(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))))

# MCTS thread function
def mcts_thread(board, model, best_move):
    print("Starting MCTS...")
    legal_moves = list(board.legal_moves)
    if legal_moves:
        move = random.choice(legal_moves)
        best_move[0] = move
    print(f"MCTS completed. Selected move: {best_move[0]}")

# Convert board to tensor
def board_to_tensor(board):
    return torch.randn(8*8*12)

# Convert result string to float
def result_to_float(result):
    return {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0}.get(result, 0.0)

# Load a pre-trained model
def load_model(model, path="chess_model.pth"):
    if os.path.exists(path):
        print(f"Loading model from {path}...")
        model.load_state_dict(torch.load(path))
        model.eval()
        print("Model loaded successfully!")
    else:
        print("No pre-trained model found. Starting from scratch...")

# Training function
def train(model, data, lr=0.001):
    print("Training ChessNet...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    # Create an interactive plot
    plt.ion()
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('lightgray')
    ax.set_facecolor('lightgray')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

    epoch = 0
    try:
        while True:  # Infinite loop
            print(f"Starting epoch {epoch + 1}...")
            total_loss = 0
            for board, result in data:
                board_tensor = board_to_tensor(board)
                optimizer.zero_grad()
                output = model(board_tensor)
                target = result_to_float(result)
                loss = criterion(output, torch.tensor([target]))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data)
            losses.append(avg_loss)

            # Update the plot
            ax.clear()
            ax.plot(losses, color='blue')
            ax.set_title(f'Training Loss over Epochs (Current Epoch: {epoch + 1})')
            plt.draw()
            plt.pause(0.1)

            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            epoch += 1
    except KeyboardInterrupt:
        print("Training interrupted.")
    return losses

# Save games to PGN
def save_games_to_pgn(games, file_path='games.pgn'):
    print(f"Saving games to {file_path}...")
    with open(file_path, 'w') as pgn_file:
        for game in games:
            pgn_file.write(str(game) + "\n\n")
    print("Games saved successfully!")

# Self-play loop with timeout mechanism and PGN saving
def self_play(model, num_games=10, think_time=2.0):
    print(f"Starting self-play for {num_games} games...")
    data = []
    outcomes = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    games_pgn = []
    for i in range(num_games):
        print(f"Playing game {i + 1}/{num_games}...")
        board = chess.Board()
        game_pgn = chess.pgn.Game()
        while not board.is_game_over():
            best_move = [None]
            search_thread = threading.Thread(target=mcts_thread, args=(board, model, best_move))
            search_thread.start()
            search_thread.join(timeout=think_time)
            if best_move[0] and best_move[0] in board.legal_moves:
                board.push(best_move[0])
                game_pgn.add_variation(best_move[0])
        result = board.result()
        outcomes[result] += 1
        for move in board.move_stack:
            board.pop()
            data.append((board.copy(), result))
        games_pgn.append(game_pgn)
        print(f"Game {i + 1} completed with result: {result}")
    print(f"Self-play completed. Outcomes: {outcomes}")
    return data, outcomes, games_pgn

# Save model function
def save_model(model, path="chess_model.pth"):
    print(f"Saving trained model to {path}...")
    torch.save(model.state_dict(), path)
    print("Model saved successfully!")

# Main
if __name__ == "__main__":
    model = ChessNet()
    load_model(model)

    print("\n--- Starting Self-Play ---")
    data, outcomes, games_pgn = self_play(model)

    print("\n--- Plotting Game Outcomes ---")
    plt.ion()  # Turn on interactive mode
    labels = list(outcomes.keys())
    values = [outcomes[key] for key in labels]
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('lightgray')
    ax.set_facecolor('lightgray')
    ax.bar(labels, values, color=['white', 'gray', 'black'])
    ax.set_title("Game Outcomes")
    ax.set_ylabel("Number of Games")
    plt.draw()
    plt.pause(0.1)  # Pause to allow the plot to be drawn

    print("\n--- Starting Training ---")
    try:
        losses = train(model, data)
    except KeyboardInterrupt:
        print("Training stopped.")

    print("\n--- Saving Model and Games ---")
    save_model(model)
    save_games_to_pgn(games_pgn)

    plt.show()
