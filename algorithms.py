import numpy as np
from abc import ABC, abstractmethod


# Abstract class for all the algorithms
class Algorithm(ABC):
    def __init__(self, casino, initial_credit) -> None:
        np.random.seed(1000)
        self._casino = casino
        self._initialCredit = initial_credit
        self._runs = None
        self._rewards = None
        self.total_reward = 0
        self.avg_reward = []
        self._n = 0

    def initialize(self):
        self._rewards = np.zeros(self._casino.machines_count)
        self._runs = np.zeros(self._casino.machines_count)
        self.total_reward = 0
        self.avg_reward = []
        self._n = 0

    def update_values(self, machine_index, reward):
        # print("Algorithm = %s : reward = %.2f" % (self.__class__.__name__, reward))
        self._n += 1
        self._runs[machine_index] += 1
        self.total_reward += reward
        self.avg_reward.append(self.total_reward / self._n)


# Abstract class for all glouton algorithms
class AbstractGlouton(Algorithm):
    def __init__(self, casino, initial_credit, test_iterations):
        super().__init__(casino, initial_credit)
        self._testIterations = test_iterations

    def do_abstract_glouton(self):
        self.initialize()
        for _ in range(self._testIterations):
            for machine_index in range(self._casino.machines_count):
                reward = self._casino.play_machine(machine_index)
                self._rewards[machine_index] += reward
                self.update_values(machine_index, reward)


# Implementing the Glouton method
class Glouton(AbstractGlouton):
    def __init__(self, casino, test_iterations=3, initial_credit=1000) -> None:
        super().__init__(casino, initial_credit, test_iterations)

    def do_glouton(self):
        super().do_abstract_glouton()
        remaining_play_credit = (
            self._initialCredit - self._testIterations * self._casino.machines_count
        )
        best_machine = np.argmax(self._rewards)
        while remaining_play_credit > 0:
            reward = self._casino.play_machine(best_machine)
            self._rewards[best_machine] += reward
            self.update_values(best_machine, reward)
            remaining_play_credit -= 1
        return self._rewards.sum()


# Implementing the epsilon-Glouton method
class EpsilonGlouton(AbstractGlouton):
    def __init__(
        self,
        casino,
        initial_credit=10000,
        test_iterations=5,
        epsilon=0.1,
    ) -> None:
        super().__init__(casino, initial_credit, test_iterations)
        self._epsilon = epsilon

    def do_epsilon_glouton(self):
        super().do_abstract_glouton()
        remaining_credit = (
            self._initialCredit - self._testIterations * self._casino.machines_count
        )
        best_machine_index = np.argmax(self._rewards)
        while remaining_credit > 0:
            if np.random.random() > self._epsilon:
                chosen_index = best_machine_index
            else:
                chosen_index = np.random.randint(0, self._casino.machines_count)
            reward = self._casino.play_machine(chosen_index)
            self._rewards[chosen_index] += reward
            self.update_values(chosen_index, reward)
            remaining_credit -= 1
        return self._rewards.sum()


# Implementing the UCB method
class AbstractUCB(Algorithm, ABC):
    def __init__(self, casino, initial_credit, confidence):
        super().__init__(casino, initial_credit)
        self.confidence = confidence

    @abstractmethod
    def confidence_bound(self):
        pass

    def _maximal_reward_arg(self):
        if self._n == 0:
            return np.random.choice(len(self._casino.machines))
        else:
            return np.argmax(self.confidence_bound())


class UCB(AbstractUCB):
    def __init__(
        self,
        casino,
        initial_credit=10000,
        confidence=1,
    ) -> None:
        super().__init__(casino, initial_credit, confidence)

    def confidence_bound(self):
        return self._rewards + self.confidence * np.sqrt(
            np.log(self._n) / (self._runs + 0.1)
        )

    def do_ucb(self):
        self.initialize()
        for i in range(self._initialCredit):
            arg = self._maximal_reward_arg()
            reward = self._casino.play_machine(arg)
            self._rewards[arg] = (self._runs[arg] * self._rewards[arg] + reward) / (
                self._runs[arg] + 1
            )
            self.update_values(arg, reward)
        assert self._runs.sum() == self._initialCredit
        return self.total_reward


class ImprovedUCB(AbstractUCB):
    def __init__(self, casino, initial_credit=10000, confidence=1, lr=0.1) -> None:
        super().__init__(casino, initial_credit, confidence)
        self.lr = lr

    def confidence_bound(self):
        delta = 2 * np.log(self._n * self._runs + 0.1) / ((self._runs + 0.1) ** 2)
        delta[delta < 0] = 0
        return self._rewards + self.confidence * np.sqrt(delta)

    def do_improved_ucb(self):
        self.initialize()
        for i in range(self._casino.machines_count):
            reward = self._casino.play_machine(i)
            self.update_values(i, reward)
        for i in range(self._initialCredit - self._casino.machines_count):
            arg = self._maximal_reward_arg()
            reward = self._casino.play_machine(arg)
            self._rewards[arg] += self.lr * (reward - self._rewards[arg])
            self.update_values(arg, reward)
        assert self._runs.sum() == self._initialCredit
        return self.total_reward


class OptimalPlayer(Algorithm):
    """
    This player cheats! it uses the function real_best_choice of Casino
    to play the best bandit every time
    """

    def __init__(self, casino, initial_credit=1000) -> None:
        super().__init__(casino, initial_credit)

    def do_optimal(self):
        self.initialize()
        best_machine, _ = self._casino.real_best_choice()
        for _ in range(self._initialCredit):
            reward = self._casino.play_machine(best_machine)
            self.update_values(best_machine, reward)
