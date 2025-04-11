import numpy as np

class NormalLaw:

    MIN_MEAN = 10
    MAX_MEAN = 20

    MIN_STD_DEV = 1
    MAX_STD_DEV = 6

    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev
    
    def __call__(self):
        return np.random.normal(self.mean, self.std_dev)
    
    def __repr__(self):
        return f"μ{self.mean:.2f} σ{self.std_dev:.2f}"
    
    def __str__(self):
        return self.__repr__()
    
    @staticmethod
    def generate_law():
        mean = np.random.uniform(NormalLaw.MIN_MEAN, NormalLaw.MAX_MEAN)
        std_dev = np.random.uniform(NormalLaw.MIN_STD_DEV, NormalLaw.MAX_STD_DEV)
        return NormalLaw(mean, std_dev)
    
    @staticmethod
    def greater(law1, law2):
        return law1.mean > law2.mean
    
class Cell:
    def __init__(self, space, x, y, law_class):
        self.space = space
        self.x = x
        self.y = y
        self.law_class = law_class
        self.law = law_class.generate_law()

    def get_gain(self):
        return self.law()
    
    def __repr__(self):
        return f"[{self.law}]"
    
    def __str__(self):
        return self.__repr__()

    def greater(self, other):
        return self.law_class.greater(self, other)

class Space:
    def __init__(self, sizeX, sizeY, law_class):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.law_class = law_class
        self.cell_grid = [[Cell(self, x, y, law_class) for y in range(sizeY)] for x in range(sizeX)]

    def get_cell(self, x, y):
        if (not (0 <= x < self.sizeX)) or (not (0 <= y < self.sizeY)):
            raise IndexError(f"Cell ({x}, {y}) is out of bounds.")
        return self.cell_grid[y][x]

    def get_gain_of(self, x, y):
        return self.get_cell(x, y).get_gain()
    
    def get_best_cell(self):
        best_cell = self.get_cell(0, 0)
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                if self.get_cell(i, j).greater(best_cell):
                    best_cell = self.get_cell(i, j)
        return best_cell
    
    def __repr__(self):
        representation = ""
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                representation += f"{self.get_cell(x, y)} "
            representation += "\n"
        return representation
    
