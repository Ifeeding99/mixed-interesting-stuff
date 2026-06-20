import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class GrayScottDiffusion:
    def __init__(self, grid_size, d_a, d_b, feed_rate, kill_rate, laplacian_a, laplacian_b, time_step):
        self.grid = np.random.random(size=[grid_size, grid_size])
        self.d_a = d_a
        self.d_b =d_b
        self.feed_rate = feed_rate
        self.kill_rate = kill_rate
        self.laplacian_a = laplacian_a
        self.laplacian_b = laplacian_b
        self.time_step = time_step

    def convolution(self, matrix, kernel): # kernel and matrix are assumed to be square matrices
        excess_width = len(kernel) // 2
        kernel_size = len(kernel)
        # padding
        padded_matrix = np.pad(matrix, pad_width=excess_width, mode='constant', constant_values=0)
        result = np.zeros_like(matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                neighborhood = padded_matrix[i: i+kernel_size, j: j+kernel_size]
                n = (neighborhood * kernel).sum()
                result[i,j] = n

        return result