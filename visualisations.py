from matplotlib import pyplot as plt
from data import batch

def save_sample_circle(radius=30, noise_level=0.5):
    circle, label = batch(batch_size=1, as_np=True, min_radius=radius, max_radius=radius+1, noise_level=noise_level)
    fig, ax = plt.subplots()
    ax.imshow(circle[0][0], cmap='gray')
    ax.set_title(f'Circle with radius {radius} and noise level {noise_level}')
    plt.savefig(f'circle_{radius}_{noise_level}.png')