import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

class MachineNormal:

    MEAN_MAX = 20
    MEAN_MIN = 100

    SIGMA_MAX = 5
    SIGMA_MIN = 0
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        self.mean = np.random.uniform(MachineNormal.MEAN_MIN, MachineNormal.MEAN_MAX)
        self.standard_deviation = np.random.uniform(MachineNormal.SIGMA_MIN, MachineNormal.SIGMA_MAX)

    def generate_random_number(self, lower=0):
        return max(np.random.normal(self.mean, self.standard_deviation), lower)
            
    def __repr__(self):
        return f"[μ{self.mean:.2f} σ{self.standard_deviation:.2f}]"

    def __str__(self):
            return self.__repr__()
    
    def visualize_generation(self, estimation=1000):
        generated_numbers = [self.generate_random_number() for _ in range(estimation)]
        
        plt.figure(figsize=(8, 6))
        plt.hist(generated_numbers, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        
        plt.title(f'{self.x} {self.y}: Distribution of Generated Numbers (n={estimation})', fontsize=14)
        plt.xlabel('Generated Values', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.show()

class Space:
    def __init__(self, name, sizeX, sizeY, class_machine):
        self.name = name
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.class_machine = class_machine
        self.grid = self.generate_grid()

    def generate_grid(self, other_class=None):
        grid = [[] for _ in range(self.sizeX)]
        
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                grid[i].append(self.class_machine(i, j) if not other_class else other_class(i, j))

        return grid
    
    def visualize_generation_of(self, x, y):
        machine = self.grid[y][x]
        machine.visualize_generation()
    
    def get_gain_of(self, x, y, number=1):
        return sum(self.grid[y][x].generate_random_number() for _ in range(number))

    def __repr__(self):
        representation = ""
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                representation += str(self.grid[i][j]) + " "
            representation += "\n"

        return representation
    
    def __str__(self):
        return self.__repr__()
    
    def best_machine(self):
        self.best_machine_mean = float('-inf')
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                if self.grid[i][j].mean > self.best_machine_mean:
                    self.best_machine_mean = self.grid[i][j].mean
                    self.best_machine_x = j
                    self.best_machine_y = i

        return self.best_machine_x, self.best_machine_y