import numpy as np
import matplotlib.pyplot as plt
import math

class Agent:

    agentStore = []
    
    def __init__(self, name, space, trials_number=100):
        self.name = name
        self.space = space
        self.trials_number = trials_number
        self.gain_history = []
        self.optimal_gain_history = []
        Agent.agentStore.append(self)

    def select_position(self):
        return np.random.randint(0, self.space.sizeX), np.random.randint(0, self.space.sizeY)

    def play(self):
        for _ in range(self.trials_number):
            x, y = self.select_position()
            gain = self.space.get_gain_of(x, y)
            self.gain_history.append(gain)

    def visualize_gain_variability(self):
        if not len(self.gain_history):
            print("No gain history to visualize.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(self.gain_history)), self.gain_history, color='b', alpha=0.7, s=3)
        
        plt.title(f"{self.name}: {self.space.name}: Gain History Over Trials", fontsize=14)
        plt.xlabel("Trial Number", fontsize=12)
        plt.ylabel("Gain", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.show()

    def visualize_gain(self, show_optimal=False):
        if not self.gain_history:
            print("No gain history to visualize.")
            return
        
        cumulative_gain = np.cumsum(self.gain_history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_gain, marker='o', color='g', linestyle='-', alpha=0.7, label='Cumulative Gain', linewidth=0.5)
        
        if show_optimal and self.optimal_gain_history:
            cumulative_optimal_gain = np.cumsum(self.optimal_gain_history)
            plt.plot(cumulative_optimal_gain, marker='x', color='r', linestyle='--', alpha=0.7, label='Optimal Cumulative Gain', linewidth=0.5)
        
        plt.title("Cumulative Gain Over Trials", fontsize=14)
        plt.xlabel("Trial Number", fontsize=12)
        plt.ylabel("Cumulative Gain", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.show()

    def calculate_optimal_strategy_gain(self):
        x, y = self.space.best_machine()
        self.optimal_gain_history = []
        for _ in range(self.trials_number):
            gain = self.space.get_gain_of(x, y)
            self.optimal_gain_history.append(gain)

    def visualization_regret(self, x, y):
        if not self.optimal_gain_history:
            print("No optimal gain history to visualize regret.")
            return
        
        regret = np.array(self.gain_history) - np.array(self.optimal_gain_history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(regret, marker='o', color='b', linestyle='-', alpha=0.7, label='Regret', linewidth=0.5)
        
        plt.title(f"{self.name}: {self.space.name}: Regret Over Trials", fontsize=14)
        plt.xlabel("Trial Number", fontsize=12)
        plt.ylabel("Regret", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.show()

    def visualize_regrets(agentStore):
        if not agentStore:
            print("No agents in agentStore to visualize.")
            return

        plt.figure(figsize=(10, 6))

        for agent in agentStore:
            if not len(agent.gain_history) or not len(agent.optimal_gain_history):
                print(f"Agent {agent.name} does not have the required data (gain_history or optimal_gain_history).")
                continue

            # Calcul du regret à chaque itération
            regrets = np.cumsum(np.array(agent.optimal_gain_history) - np.array(agent.gain_history))
            
            # Trace les regrets
            plt.plot(regrets, label=agent.name, alpha=0.8)

        plt.title("Regrets of Agents Over Iterations", fontsize=14)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Cumulative Regret", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.5, linestyle="--")
        plt.show()

class AgentUniform(Agent):
    pass

class AgentTraining(Agent):
    def __init__(self, name, space, trials_number=1000, training_percent=0.2):
        super().__init__(name, space, trials_number)
        self.training_percent = training_percent
        self.space_information = self.space.generate_grid(lambda x, y: 0)

    def play(self):
        training_size = int(self.trials_number * self.training_percent)
        
        for _ in range(training_size):
            x, y = self.select_position()
            gain = self.space.get_gain_of(x, y)
            self.gain_history.append(gain)
            self.space_information[y][x] += gain

        self.probably_best_machine_y, self.probably_best_machine_x = self.get_best_machine_probably()

        for _ in range(self.trials_number - training_size):
            gain = self.space.get_gain_of(self.probably_best_machine_x, self.probably_best_machine_y)
            self.gain_history.append(gain)

    def get_best_machine_probably(self):
        best_x = 0
        best_y = 0
        best_gain = 0
        for i in range(self.space.sizeX):
            for j in range(self.space.sizeY):
                if self.space_information[j][i] > best_gain:
                    best_gain = self.space_information[j][i]
                    best_x = j
                    best_y = i
        return best_x, best_y
    
class AgentEpsilonGreedy(Agent):

    def __init__(self, name, space, trials_number=1000, epsilon=0.1, decaying=1):
        super().__init__(name, space, trials_number)
        self.epsilon = epsilon
        self.space_information = self.space.generate_grid(lambda x, y: [0, 0, 0])
        self.decaying = decaying

    def play(self):
        for _ in range(self.trials_number):

            if np.random.uniform(0, 1) < self.epsilon:
                x, y = self.select_position()
            else:
                x, y = self.get_greedybest_machine()

            gain = self.space.get_gain_of(x, y)
            self.gain_history.append(gain)
            self.space_information[y][x][0] += gain
            self.space_information[y][x][1] += 1
            self.space_information[y][x][2] = self.space_information[y][x][0] / self.space_information[y][x][1]

            self.epsilon *= self.decaying
        
    def get_greedybest_machine(self):
        best_x = 0
        best_y = 0
        best_mean = 0
        for i in range(self.space.sizeX):
            for j in range(self.space.sizeY):
                if self.space_information[j][i][2] > best_mean:
                    best_mean = self.space_information[j][i][2]
                    best_x = i
                    best_y = j
        return best_x, best_y
    
class AgentUCB(Agent):

    def __init__(self, name, space, trials_number=1000, exploration_coef=5):
        super().__init__(name, space, trials_number)
        self.space_information = self.space.generate_grid(lambda x, y: [0, 0, 0])
        self.exploration_coef = exploration_coef

    def play(self):
        for i in range(self.trials_number):

            if i == 0:
                x, y = self.select_position()
            else:
                x, y = self.get_ucb_machine(i)

            gain = self.space.get_gain_of(x, y)
            self.gain_history.append(gain)
            self.space_information[y][x][0] += gain
            self.space_information[y][x][1] += 1
            self.space_information[y][x][2] = self.space_information[y][x][0] / self.space_information[y][x][1]
        
    def get_ucb_machine(self, trial):
        best_x = 0
        best_y = 0
        best_ucb = float('-inf')
        for i in range(self.space.sizeX):
            for j in range(self.space.sizeY):
                ucb_ = self.space_information[j][i][2] + 5 * math.sqrt(2 * math.log(max(trial, 1)) / max(self.space_information[j][i][1], 0.01))
                if ucb_ > best_ucb:
                    best_ucb = ucb_
                    best_x = i
                    best_y = j
        return best_x, best_y