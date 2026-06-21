import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy.signal import convolve2d
matplotlib.use('TkAgg')
# inspiration from https://karlsims.com/rd.html
# nice website for visualisation: https://karlsims.com/rdtool.html?s=b3UrV0sMc


laplacian_a = np.array([
    [0.05, 0.2, 0.05],
    [0.2, -1, 0.2],
    [0.05, 0.2, 0.05]], dtype=np.float32)

laplacian_b = np.array([
    [0.05, 0.2, 0.05],
    [0.2, -1, 0.2],
    [0.05, 0.2, 0.05]], dtype=np.float32)

d_a = 1.0
d_b = 0.5
f = 0.055
k = 0.062


class GrayScottDiffusion:
    def __init__(self, grid_size, d_a, d_b, feed_rate, kill_rate, laplacian_a, laplacian_b, time_step, random = False, fast_conv = True):
        if random:
            self.grid = np.random.random([grid_size, grid_size,2])
        else:
            self.grid = np.zeros([grid_size, grid_size,2]) # the last dimensions serves only to store A and B in the same array
            self.grid[:,:,0] = np.ones([grid_size, grid_size], dtype=np.float32) # the first of the 2 dimensions is A and the second B
        self.d_a = d_a                                                
        self.d_b =d_b
        self.feed_rate = feed_rate
        self.kill_rate = kill_rate
        self.laplacian_a = laplacian_a
        self.laplacian_b = laplacian_b
        self.time_step = time_step
        self.fast_conv = fast_conv

    def convolution(self, matrix, kernel): # kernel and matrix are assumed to be square matrices
        excess_width = len(kernel) // 2
        kernel_size = len(kernel)
        # padding
        padded_matrix = np.pad(matrix, pad_width=excess_width, mode='reflect')
        result = np.zeros_like(matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                neighborhood = padded_matrix[i: i+kernel_size, j: j+kernel_size]
                n = (neighborhood * kernel).sum()
                result[i,j] = n

        return result
    
    def fast_convolution(self, matrix, kernel):
        result = convolve2d(matrix, kernel, boundary='symm', mode='same')
        return result
    

    def single_step_update(self):
        current_a = self.grid[:,:,0].copy()
        current_b = self.grid[:,:,1].copy()
        if self.fast_conv:
            new_A_laplacian = self.fast_convolution(current_a, self.laplacian_a)
            new_B_laplacian = self.fast_convolution(current_b, self.laplacian_b)
        else:
            new_A_laplacian = self.convolution(current_a, self.laplacian_a)
            new_B_laplacian = self.convolution(current_b, self.laplacian_b)
        new_A = current_a + (self.d_a * new_A_laplacian - current_a * current_b**2 + self.feed_rate * (1-current_a)) * self.time_step
        new_B = current_b + (self.d_b * new_B_laplacian + current_a * current_b**2 - (self.kill_rate + self.feed_rate) * current_b) * self.time_step
        self.grid[:,:,0] = np.clip(new_A,min=0)
        self.grid[:,:,1] = np.clip(new_B, min=0, max=1)
        return self.grid


    def loop(self, n_iterations, interval = 50):
        fig, ax = plt.subplots(figsize=(15,15))
        RGB = np.stack([self.grid[:,:,0], np.zeros_like(self.grid[:,:,0]), self.grid[:,:,1]], axis=-1)
        RGB = np.clip(RGB * 255, 0, 255).astype(np.uint8)
        img = ax.imshow(RGB)
        title = ax.text(0.5, 1.01, "Step 0", transform=ax.transAxes,
                     ha="center", va="bottom")

        def update_plot(step):
            self.single_step_update()
            new_data = np.stack([self.grid[:,:,0], np.zeros_like(self.grid[:,:,0]), self.grid[:,:,1]], axis=-1)
            new_data = np.clip(new_data * 255, 0, 255).astype(np.uint8)
            img.set_data(new_data)
            title.set_text(f"Step {step}")
            return [img, title]
        
        self.ani = FuncAnimation(fig=fig, func = update_plot, frames=n_iterations, interval = interval, blit=False)
        plt.show()


    def draw_square_B(self, size, center):
        ones = np.ones((size,size), dtype=np.float32)
        self.grid[center-size//2:center+size//2, center-size//2:center+size//2,1] = ones

    def draw_noise_B(self, amplitude=0.2):
        self.grid[:,:,1] = np.random.uniform(low=0, high=amplitude, size=self.grid[:,:,1].shape)




if __name__ == '__main__':
    s = GrayScottDiffusion(grid_size=200, d_a=d_a, d_b=d_b,
                           feed_rate=f, kill_rate=k,
                           laplacian_a=laplacian_a, laplacian_b=laplacian_b,
                           time_step=1, random = False, fast_conv=True)
    
    s.draw_square_B(size=10, center=50)

    s.loop(n_iterations=5000000, interval=1)
