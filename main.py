from train import train
from evaluation import evaluate

if __name__ == "main":
    net = train()
    evaluate(net)