from circle_detection import CircleParams
from circle_detection import iou
from data import batch

def avg_iou(model, batch_size):
    total_iou = 0
    for i in range(batch_size):
        image, label = batch(1)
        label = label.detach().numpy()
        pred = model(image).detach().numpy()
        pred_params = CircleParams(pred[0][0], pred[0][1], pred[0][2])
        actual_params = CircleParams(label[0][0], label[0][1], label[0][2])
        total_iou += iou(pred_params, actual_params)
    return total_iou/batch_size

def baseline_iou(batch_size):
    total_iou = 0
    for i in range(batch_size):
        image, label = batch(1)
        label = label.detach().numpy()
        pred_params = CircleParams(0.5, 0.5, 0.3)
        actual_params = CircleParams(label[0][0], label[0][1], label[0][2])
        total_iou += iou(pred_params, actual_params)
    return total_iou/batch_size

def perfect_iou(batch_size):
    total_iou = 0
    for i in range(batch_size):
        image, label = batch(1)
        label = label.detach().numpy()
        actual_params = CircleParams(label[0][0], label[0][1], label[0][2])
        total_iou += iou(actual_params, actual_params)
    return total_iou/batch_size

def evaluate(model):
    radii = [(5,10), (10,25), (25,50), (50,99.9), (5,99.9)]
    thresholds = [0.5, 0.75, 0.9, 0.95]
    results = [[0 for t in thresholds] for r in radii]
    for k in range(len(radii)):
        for i in range(1000):
          image, label = batch(1, radii[k][0], radii[k][1])
          label = label.detach().numpy()
          pred = model(image).detach().numpy()
          pred_params = CircleParams(pred[0][0], pred[0][1], pred[0][2])
          actual_params = CircleParams(label[0][0], label[0][1], label[0][2])
          result = iou(pred_params, actual_params)
          for j in range(len(thresholds)):
              if thresholds[j] <= result:
                  results[k][j] += 1
    return results