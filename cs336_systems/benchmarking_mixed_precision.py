import torch
from torch import nn, optim

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False, device=device, dtype=dtype)
        self.ln = nn.LayerNorm(10, device=device, dtype=dtype)
        self.fc2 = nn.Linear(10, out_features, bias=False,device=device, dtype=dtype)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
    
    
if __name__ == "__main__":
    model = ToyModel(256, 1).to("cuda")
    inputs = torch.randn((32, 256), device="cuda")
    targets = torch.randn((32, 1), device="cuda")
    optimizer = optim.AdamW(model.parameters())
    loss_fn = nn.MSELoss()
    with torch.autocast("cuda", dtype=torch.bfloat16) as ctx:
        optimizer.zero_grad()
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        
        