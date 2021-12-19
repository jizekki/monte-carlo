import numpy as np

from abc import ABC

# Abstract class for all the algorithms
class Algorithm(ABC):
    def __init__(self, casino, initialCredits) -> None:
        self._casino = casino
        self._initialCredits = initialCredits
        self._runs = None
        self._rewards = None
        self.initialize()

    def initialize(self):
        self._runs = np.zeros(self._casino.machines_count)
        self._rewards = np.zeros(self._casino.machines_count)


# Abstract class for all glouton algorithms
class AbstractGlouton(Algorithm):
    def __init__(
        self,
        casino,
        initialCredits=10000,
        testIterations=5,
    ) -> None:
        super().__init__(casino, initialCredits)
        self._testIterations = testIterations

    def do_abstract_glouton(self):
        self.initialize()
        for _ in range(self._testIterations):
            for machineIndex in range(self._casino.machines_count):
                self._rewards[machineIndex] += self._casino.machines[
                    machineIndex
                ].play()
                self._runs[machineIndex] += 1


# Implementing the Glouton method
class Glouton(AbstractGlouton):
    def __init__(self, casino, testIterations=5, initialCredits=10000) -> None:
        super().__init__(casino, initialCredits, testIterations)

    def do_glouton(self):
        super().do_abstract_glouton()
        remaining_play_credits = (
            self._initialCredits - self._testIterations * self._casino.machines_count
        )
        best_machine_index = np.argmax(self._rewards)
        while remaining_play_credits > 0:
            self._rewards[best_machine_index] += self._casino.machines[
                best_machine_index
            ].play()
            self._runs[best_machine_index] += 1
            remaining_play_credits -= 1
        return self._rewards.sum()


# Implementing the epsilon-Glouton method
class EpsilonGlouton(AbstractGlouton):
    def __init__(
        self,
        casino,
        initialCredits=10000,
        testIterations=5,
        epsilon=0.1,
    ) -> None:
        super().__init__(casino, initialCredits, testIterations)
        self._epsilon = epsilon

    def do_epsilon_glouton(self):
        super().do_abstract_glouton()
        remaining_play_credits = (
            self._initialCredits - self._testIterations * self._casino.machines_count
        )
        best_machine_index = np.argmax(self._rewards)
        while remaining_play_credits > 0:
            if np.random.random() > self._epsilon:
                self._rewards[best_machine_index] += self._casino.machines[
                    best_machine_index
                ].play()
                self._runs[best_machine_index] += 1
            else:
                random_index = np.random.randint(0, self._casino.machines_count - 1)
                self._rewards[random_index] += self._casino.machines[
                    random_index
                ].play()
                self._runs[random_index] += 1
            remaining_play_credits -= 1
        return self._rewards.sum()


# Implementing the UCB method
class UCB(Algorithm):
    def __init__(self, casino, initialCredits=10000) -> None:
        super().__init__(casino, initialCredits)
        self._n = None
        self.initialize()

    def initialize(self):
        super().initialize()
        self._n = 0

    def _maximal_reward_arg(self, c=1):
        """
        Gets the machine with the highest reward so far.
        If a machine that has never been tried is found, it will be returned
        """
        arg = None
        maximal_reward = 0
        for a in range(self._casino.machines_count):
            if self._runs[a] == 0:
                return a
            else:
                delta = np.sqrt(2 * np.log(self._n) / self._runs[a])
                reward = self._rewards[a] + c * delta
                if reward > maximal_reward:
                    maximal_reward = reward
                    arg = a
        return arg

    def do_one_step_ucb(self, c=1):
        if self._n >= self._initialCredits:
            return None, self._n
        arg = self._maximal_reward_arg(c)
        reward = self._casino.machines[arg].play()
        self._rewards[arg] = (self._runs[arg] * self._rewards[arg] + reward) / (
            self._runs[arg] + 1
        )
        self._runs[arg] += 1
        self._n += 1
        return reward, self._n - 1

    def do_ucb(self, c=1):
        self.initialize()
        v = 0
        for i in range(self._initialCredits):
            reward, _ = self.do_one_step_ucb(c)
            v += reward
        assert self.do_one_step_ucb(c)[0] is None
        return v
