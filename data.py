from circle_detection import generate_examples
import numpy as np
import torch

def batch(batch_size, min_radius=None, max_radius=None):
    generator = generate_examples(min_radius=min_radius, max_radius=max_radius)
    images = np.zeros((batch_size, 1, 100, 100))
    labels = np.zeros((batch_size, 3))
    for i in range(batch_size):
        example = next(generator)
        images[i] = example[0] / 100
        labels[i] = [example[1].row / 100, example[1].col / 100, example[1].radius / 100]
    tensor_images = torch.Tensor(images)
    tensor_labels = torch.Tensor(labels)
    return (tensor_images, tensor_labels)