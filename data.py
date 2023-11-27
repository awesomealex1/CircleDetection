from circle_detection import generate_examples
import numpy as np
import torch

def batch(batch_size, min_radius=None, max_radius=None, noise_level=None, as_np=False):
    generator = generate_examples(min_radius=min_radius, max_radius=max_radius, noise_level=noise_level)
    images = np.zeros((batch_size, 1, 100, 100))
    labels = np.zeros((batch_size, 3))
    for i in range(batch_size):
        example = next(generator)
        images[i] = example[0] / 100
        labels[i] = [example[1].row / 100, example[1].col / 100, example[1].radius / 100]
    if not np:
        images = torch.Tensor(images)
        labels = torch.Tensor(labels)
    return (images, labels)