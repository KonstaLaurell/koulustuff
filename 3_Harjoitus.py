import torch
import torch.nn as nn
import chess

# ChessNet Neural Network Definition
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

def export_to_onnx(model_path="chess_model.pth", onnx_path="chess_model.onnx"):
    model = ChessNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = board_to_tensor(chess.Board())
    torch.onnx.export(model, dummy_input, onnx_path)
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
