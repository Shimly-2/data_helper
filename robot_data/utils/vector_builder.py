import numpy as np
import matplotlib.pyplot as plt

class vector_builder:
    def __init__(self):
        self.depth = [0.20, 0.40, 0.53, 0.78, 1.02, 1.25, 1.44, 1.50, 1.66, 1.88] # mm
        self.normal = [0.00, 0.36, 0.90, 1.99, 4.28, 9.44, 16.66, 18.68, 27.56, 41.66]

    def vector_fitting(self):
        parameter = np.polyfit(self.depth, self.normal, 3)
        func = np.poly1d(parameter)
        return func
    
    def rebuild_depth_with_force(self, depth):
        force_map = self.vector_fitting()
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                depth[i][j] = force_map(depth[i][j])
        return depth

if __name__ == "__main__":
    fitting = vector_builder()
    parameter = fitting.vector_fitting()
    xn = np.linspace(0, 2, 100)
    yn = np.poly1d(parameter)
    plt.plot(xn, yn(xn), fitting.depth, fitting.normal, 'o')
    plt.show()