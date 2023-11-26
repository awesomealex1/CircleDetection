from model import Net
import torch.nn as nn
import torch.optim as optim
from data import batch

load = False
net = Net()

if load:
    net.load_state_dict(torch.load("gdrive/MyDrive/10k_iterations"))

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

running_loss = 0
for epoch in range(1000):  # loop over the dataset multiple times
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = batch(32)

    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if epoch > 40000000:
        optimizer.param_groups[0]['lr'] = 0.0001

    if epoch > 40000000:
        optimizer.param_groups[0]['lr'] = 0.00001

    running_loss += loss.item()
    # print statistics
    if epoch % 10 == 0:
      print("Epoch:",epoch,"Loss:",running_loss/10)
      running_loss = 0

print('Finished Training')