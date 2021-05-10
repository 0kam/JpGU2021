from torch import nn
import torch
from torch.nn import functional as F

from pixyz.distributions import Normal, RelaxedCategorical, Deterministic

## inference model q(z|x)
class Inference(Normal):
    def __init__(self, x_dim, y_dim, z_dim, drop_out_rate=0.0):
        super().__init__(cond_var=["x", "y"], var=["z"], name="q")

        self.h_dim = int((x_dim + z_dim) / 2)
        self.gru = nn.GRU(input_size=x_dim, hidden_size=self.h_dim, batch_first=True, dropout=drop_out_rate)
        # self.fc11 = nn.Linear(self.h_dim, z_dim)
        #self.fc12 = nn.Linear(self.h_dim, z_dim)
        self.fc11 = nn.Linear(self.h_dim+1, z_dim)
        self.fc12 = nn.Linear(self.h_dim+1, z_dim)

    def forward(self, x, y):
        _, h = self.gru(x)
        h = h.contiguous().view(-1, self.h_dim)
        y = y.argmax(1) / y.shape[1]
        y = y.unsqueeze(1)
        h = torch.cat([h, y], 1)
        return {"loc": self.fc11(h), "scale": F.softplus(self.fc12(h))}

# generative model p(x|z) 
class Generator(Normal):
    def __init__(self, x_dim, z_dim, seq_length, device):
        super().__init__(cond_var=["z"], var=["x"], name="p")

        h_dim = int((x_dim + z_dim) / 2)
        self.device = device
        self.seq_length = seq_length
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.tanh = nn.Tanh()
        self.gru = nn.GRUCell(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, x_dim)
        self.fc22 = nn.Linear(h_dim, x_dim)

    def forward(self, z):
        h = self.tanh(self.fc1(z))
        x1 = torch.zeros(h.shape).to(self.device)
        hidden_state = h
        x = []
        for _ in range(self.seq_length):
            h = self.gru(x1, hidden_state)
            x1 = h
            hidden_state = h
            x.append(x1)
        x = torch.stack(x, dim=1).squeeze()

        return {"loc": self.fc21(x), "scale": F.softplus(self.fc22(x))}

# classifier q(y|x)
class Classifier(RelaxedCategorical):
    def __init__(self, x_dim, y_dim, drop_out_rate):
        super().__init__(cond_var=["x"], var=["y"], name="q")
        self.h_dim = int((x_dim + y_dim) / 2)

        self.gru = nn.GRU(input_size=x_dim, hidden_size=self.h_dim, batch_first=True, dropout=drop_out_rate)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(self.h_dim, y_dim)

    def forward(self, x):
        _, h = self.gru(x)
        h = h.contiguous().view(-1, self.h_dim)
        h = F.softmax(self.fc2(h), dim=1)
        h = h + 1e-9
        return {"probs": h}

# prior model p(z|y)
class Prior(Normal):
    def __init__(self, z_dim, y_dim):
        super().__init__(var=["z"], cond_var=["y"], name="p_{prior}")

        self.fc11 = nn.Linear(y_dim, z_dim)
        self.fc12 = nn.Linear(y_dim, z_dim)
    
    def forward(self, y):
        return {"loc": self.fc11(y), "scale": F.softplus(self.fc12(y))}

class Prior_Q(Normal):
    def __init__(self, z_dim, y_dim):
        super().__init__(var=["z"], cond_var=["y"], name="p_{prior}")
        self.z_dim = z_dim
    def forward(self, y):
        y = y.argmax(1)
        y = y.repeat(self.z_dim, 1).transpose(0,1).to("cuda")
        sigma = torch.ones(y.shape).to("cuda")
        return {"loc": y, "scale": sigma}

class Discriminator(Deterministic):
    def __init__(self, x_dim, y_dim):
        super().__init__(var=["t"], cond_var=["x", "y"], name="d")
        h_dim = int((x_dim + y_dim) / 2)
        self.fc1 = nn.Linear(x_dim+y_dim, h_dim)
        self.prelu1 = nn.PReLU()
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.prelu2 = nn.PReLU()
        self.fc3 = nn.Linear(h_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, y):
        h = self.prelu1(self.fc1(x))
        h = self.prelu2(self.fc2(h))
        t = self.sigmoid(self.fc3(h))
        return {"t": t}

class PData(Deterministic):
    def __init__(self):
        super().__init__(var=["x"], cond_var=[], name="p")
    
    def forward(self, x):
        return {"x": x}