class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool1 = nn.AvgPool2d(3,1, padding=1)
        self.L1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=2), nn.BatchNorm2d(32), nn.Conv2d(32, 32, 3, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2))
        self.L2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=2), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2))
        self.L3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=2), nn.BatchNorm2d(128), nn.Conv2d(128, 128, 3, padding=2), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2))
        self.L4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=2), nn.BatchNorm2d(256), nn.Conv2d(256, 256, 3, padding=2), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2,2))
        self.L5 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=2), nn.BatchNorm2d(128), nn.Conv2d(128, 128, 3, padding=2), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2))
        self.L6 = nn.Sequential(nn.Conv2d(128, 16, 3, padding=2), nn.BatchNorm2d(16), nn.ReLU())
        self.flat = nn.Flatten()
        self.FC = nn.Sequential(nn.Linear(1296, 256), nn.Linear(256, 128), nn.Linear(128, 3))

    def forward(self, x):
        x = self.avgpool1(x)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        x = self.L6(x)
        x = self.flat(x)
        x = self.FC(x)
        return x