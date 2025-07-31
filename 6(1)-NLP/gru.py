import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # 구현하세요!
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Update Gate Parameter
        self.W_z = nn.Linear(input_size, hidden_size, bias=True)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Reset Gate Parameter
        self.W_r = nn.Linear(input_size, hidden_size, bias=True)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Candidate Hidden state Parameter
        self.W_h = nn.Linear(input_size, hidden_size, bias=True)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.norm_z = nn.LayerNorm(hidden_size)
        self.norm_r = nn.LayerNorm(hidden_size)
        self.norm_h = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        z = torch.sigmoid(self.norm_z(self.W_z(x) + self.U_z(h)))
        r = torch.sigmoid(self.norm_r(self.W_r(x) + self.U_r(h)))
        # z = torch.sigmoid(self.W_z(x) + self.U_z(h))
        # r = torch.sigmoid(self.W_r(x) + self.U_r(h))
        preact = self.W_h(x) + self.U_h(r * h)
        h_tilde = torch.tanh(self.norm_h(preact))
        h_next = (1 - z) * h + z * h_tilde
        return h_next


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # 구현하세요!
        self.num_layers = 2
        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.cells.append(GRUCell(input_size, hidden_size))
            else:
                self.cells.append(GRUCell(hidden_size, hidden_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        batch_size, seq_len, _ = inputs.size()
        h_layers = [torch.zeros(batch_size, self.hidden_size, device=inputs.device, dtype=inputs.dtype)
                    for _ in range(self.num_layers)]
        outputs = []  # 각 time step의 top layer hidden state 저장용
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            for layer_idx in range(self.num_layers):
                h_layers[layer_idx] = self.cells[layer_idx](x_t, h_layers[layer_idx])
                x_t = self.dropout(h_layers[layer_idx]) 
            outputs.append(h_layers[-1])  # 마지막 layer의 hidden state만 저장
        return torch.mean(torch.stack(outputs, dim=1), dim=1)  # [B, seq_len, H] → [B, H]
