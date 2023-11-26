from model import Net
import torch.nn as nn
import torch.optim as optim
from data import batch

load = False
net = Net()

if load:
    net.load_state_dict(torch.load("gdrive/MyDrive/10k_iterations"))

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

running_loss = 0
for i in range(1,10001):
    inputs, labels = batch(32)

    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if i > 1000:
        optimizer.param_groups[0]['lr'] = 0.0001

    if i > 3000:
        optimizer.param_groups[0]['lr'] = 0.00001

    running_loss += loss.item()
    # print statistics
    if i % 10 == 0:
        print("Iteration:",i,"Loss:",running_loss/10)
        running_loss = 0

print('Finished Training')