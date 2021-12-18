import numpy as np


class Bandit:
    def __init__(self, mu=None):
        if mu is None:
            self.mu = np.random.random()  # mean payoff of the machine
        else:
            self.mu = mu

    def play(self):
        return 1 if np.random.random() > self.mu else 0


class Casino:
    def __init__(self, nb=100):
        self.machines_count = nb
        self.machines = [Bandit() for _ in range(nb)]

    def play_machine(self, i):
        return self.machines[i].play()

    # We cheat if we use it. Must only be used to calculate the regret
    def real_best_choice(self):
        return min(self.machines, key=lambda m: m.mu)
