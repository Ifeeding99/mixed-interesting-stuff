import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# inspiration from https://karlsims.com/rd.html


laplacian_a = np.array([
    [0.05, 0.2, 0.05],
    [0.2, -1, 0.2],
    [0.05, 0.2, 0.05]], dtype=np.float32)

laplacian_b = np.array([
    [0.05, 0.2, 0.05],
    [0.2, -1, 0.2],
    [0.05, 0.2, 0.05]], dtype=np.float32)



class GrayScottDiffusion:
    def __init__(self, grid_size, d_a, d_b, feed_rate, kill_rate, laplacian_a, laplacian_b, time_step):
        self.grid = np.random.random(size=[grid_size, grid_size,2]) # the last dimensions serves only to store A and B in the same array
        self.d_a = d_a                                                # the first of the 2 dimensions is A and the second B
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
    

    def single_step_update(self):
        current_a = self.grid[:,:,0].copy()
        current_b = self.grid[:,:,1].copy()
        new_A_laplacian = self.convolution(current_a, self.laplacian_a)
        new_B_laplacian = self.convolution(current_b, self.laplacian_b)
        new_A = current_a + (self.d_a * new_A_laplacian - current_a * current_b**2 + self.feed_rate * (1-current_a)) * self.time_step
        new_B = current_b + (self.d_b * new_B_laplacian + current_a * current_b**2 - (self.kill_rate + self.feed_rate) * current_b) * self.time_step
        self.grid[:,:,0] = new_A
        self.grid[:,:,1] = new_B
        return self.grid