from train import train
from evaluation import evaluate
import torch

if __name__ == "main":
    net = train()
    torch.save(net.state_dict(), "circle_detector")
    evaluate(net)