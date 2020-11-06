import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

num_epochs = 10
print(f'This script will simulate the process of training a model with pytorch. The model will train for {num_epochs} iterations.')

X1 = torch.randn(1000, 50)
X2 = torch.randn(1000, 50) + 1.5
X = torch.cat([X1, X2], dim=0)
Y1 = torch.zeros(1000, 1)
Y2 = torch.ones(1000, 1)
Y = torch.cat([Y1, Y2], dim=0)

if torch.cuda.is_available():
    print('GPU is available and will be used.')
    device = 'cuda'
else:
    print('GPU is not available. CPU will be used')
    device = 'cpu'

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y
    
net = Net().to(device)
opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.BCELoss()

def train_epoch(model, opt, criterion, batch_size=50):
    model.train()
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i:beg_i + batch_size, :]
        y_batch = Y[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = net(x_batch.to(device))
        # (2) Compute diff
        loss = criterion(y_hat, y_batch.to(device))
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        losses.append(loss.data.cpu().numpy())
    return losses

e_losses = []

for epoch in range(num_epochs):
    loss = train_epoch(net, opt, criterion)
    e_losses += loss
    loss = torch.stack([torch.tensor(l) for l in loss])
    print(f'At iteraton {epoch} the error was {round(loss.mean().item(), 2)}')
    
print('done!')