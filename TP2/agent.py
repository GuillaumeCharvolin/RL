import numpy as np

class CellInfo:
    def __init__(self, agent, x, y, cell):
        self.agent = agent
        self.x = x
        self.y = y
        self.cell = cell
        self.use_number = 0
        self.sum_gain = 0
        self.average_gain = 0
        self.estimated_gain = 0

    def get_gain(self):
        gain = self.cell.get_gain()
        self.use_number += 1
        self.sum_gain += gain
        self.average_gain = self.sum_gain / self.use_number
        return gain
    
    def __repr__(self):
        #return f"[n{self.use_number:5.2e} s{self.sum_gain:5.2e} a{self.average_gain:5.2e}]"
        return f"[x{self.x} y{self.y} q{self.estimated_gain:05.2f}]"

    def __str__(self):
        return self.__repr__()
    
class SpaceInfo:
    def __init__(self, space, agent):
        self.agent = agent
        self.space = space
        self.sizeX = space.sizeX
        self.sizeY = space.sizeY
        self.cell_grid = [[CellInfo(self.agent, x, y, space.get_cell(x, y)) for y in range(self.sizeY)] for x in range(self.sizeX)]
    
    def get_cell(self, x, y):
        return self.cell_grid[y][x]
    
    def get_gain_of(self, x, y):
        return self.get_cell(x, y).get_gain()
    
    def get_best_estimated_cell(self):
        best_cell_info = self.get_cell(0, 0)
        for i in range(self.sizeX):
            for j in range(self.sizeY):
                if self.get_cell(i, j).estimated_gain > best_cell_info.estimated_gain:
                    best_cell_info = self.get_cell(i, j)
        return best_cell_info.cell

    def get_real_best_cell(self):
        return self.space.get_best_cell()
    
    def __repr__(self):
        representation = ""
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                representation += f"{self.get_cell(x, y)} "
            representation += "\n"
        return representation
    
class Policy:
    def __call__(self, accessible_cell_list):
        return np.random.choice(accessible_cell_list)
    
class E_Greedy(Policy):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, accessible_cell_list):
        if np.random.rand() < self.epsilon:
            return np.random.choice(accessible_cell_list)
        else:
            return max(accessible_cell_list, key=lambda cell: cell.estimated_gain)
        
class Softmax(Policy):
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, accessible_cell_list):
        gains = np.array([cell.estimated_gain for cell in accessible_cell_list])
        scaled_gains = gains / self.temperature
        probabilities = np.exp(scaled_gains) / np.sum(np.exp(scaled_gains))
        
        return np.random.choice(accessible_cell_list, p=probabilities)

class Agent:
    agentStore = []

    def __init__(self, name, space, policy, learning_rate=0.1, horizon=0.5, start_x=0, start_y=0, episode_number=1_000, step_number=1_000):
        self.name = name
        self.space = space
        self.space_info = SpaceInfo(space, self)
        self.episode_number = episode_number
        self.step_number = step_number
        self.gain_history = []
        self.current_x = start_x
        self.current_y = start_y
        self.start_x = start_x
        self.start_y = start_y
        self.policy = policy
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.estimated_best_gain_history = []
        self.real_best_gain_history = []
        Agent.agentStore.append(self)

    def get_gain_of(self, x, y):
        self.space_info.get_gain_of(x, y)

    def get_gain_current_cell(self):
        gain = self.space_info.get_gain_of(self.current_x, self.current_y)
        self.gain_history.append(gain)
        return gain

    def get_accessible_cell(self):
        accessible_cell_list = []
        for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = self.current_x + dx, self.current_y + dy
            if 0 <= nx < self.space.sizeX and 0 <= ny < self.space.sizeY:
                accessible_cell_list.append(self.space_info.get_cell(nx, ny))
        return accessible_cell_list
    
    def select_cell_to_go(self):
        accessible_cell_list = self.get_accessible_cell()
        cell_to_go = self.policy(accessible_cell_list)
        return cell_to_go

    def move(self, cell_to_go):
        self.current_x, self.current_y = cell_to_go.x, cell_to_go.y

    def run(self):
        for _ in range(self.episode_number):
            self.current_x = self.start_x
            self.current_y = self.start_y
            self.update_estimated_gain_strategy()

    def update_estimated_gain_strategy(self):
        pass

class SASRA(Agent):
    def update_estimated_gain_strategy(self):
        cell_to_go = self.select_cell_to_go()
        for _ in range(self.step_number):
            self.move(cell_to_go)
            gain = self.get_gain_current_cell()
            cell_choosen_after = self.select_cell_to_go()
            cell_to_go.estimated_gain += self.learning_rate * (gain + self.horizon * cell_choosen_after.estimated_gain - cell_to_go.estimated_gain)
            cell_to_go = cell_choosen_after