from model import Net
import torch.nn as nn
import torch.optim as optim
from data import batch
import torch

def train(model_path=None, n_iterations=10000, b_size=32, silent=False):
    net = Net()

    if model_path:
        net.load_state_dict(torch.load(model_path))

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0
    losses = []

    for i in range(1,n_iterations+1):
        inputs, labels = batch(b_size)

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

        if not silent and i % 100 == 0:
            print(f"Iteration: {i}, Loss: {running_loss/100}")
            losses.append(running_loss/100)
            running_loss = 0

    print('Finished Training')
    return net