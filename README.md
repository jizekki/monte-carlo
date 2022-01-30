# monte-carlo
Studying and improving the Monte Carlo tree search algorithm

# Improvements and functionalities

- Implemented the different algorithms
- Experimented using different parameters
- Implemented an improved version of UCB.

## File contents:
- The `casino.py` file defines the architecture of the casino. It was provided at the beginning of the TP. The *rewards* calculation method has been changed.
- The `algorithms.py` file contains the implementation of the different algorithms: Glutton, Epsilon-Glouton, UCB and its improved version as well as an
algorithm that always plays the machine with the best *payoff*.
- The `notebook.ipynb` file and its html version contain the different experiments carried out.

# Additional changes as to the initial version
In addition to all the previous functionalities :
- The code has been cleaned to follow the `PEP8` style guide.
- The code has been organized into classes. Duplicated code has been abstracted.
