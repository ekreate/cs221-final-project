import collections, random
from typing import List, Tuple, Dict, Any
from main import *

# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    # Return the start state.
    def startState(self) -> Tuple: raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state: Tuple) -> List[Any]: raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state: Tuple, action: Any) -> List[Tuple]: raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        # print ("%d states" % len(self.states))
        # print (self.states)

############################################################

# A simple example of an MDP where states are integers in [-n, +n].
# and actions involve moving left and right by one position.
# We get rewarded for going to the right.

############################################################

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state: Tuple) -> Any: raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state: Tuple, action: Any, reward: int, newState: Tuple): raise NotImplementedError("Override me")

# An RL algorithm that acts according to a fixed policy |pi| and doesn't
# actually do any learning.

############################################################

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(rl: RLAlgorithm, numTrials=50000, maxIterations=200, verbose=False, screen_en=False):
    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        field = Field(7)
        state = Drop7()
        state.set_field(field)
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        for _ in range(maxIterations):
            # saved_state = deepcopy(state)
            action = rl.getAction(state)
            if screen_en == False:
                new_state = playDrop7(None, state, action)
            else:
                curses.wrapper(playDrop7, state, action)
                new_state = state
            if new_state.endGame == True:
                rl.incorporateFeedback(state, action, 0, None)
                break
            rl.incorporateFeedback(state, action, 1, new_state)
            totalReward +=  1
            state =  new_state
        if verbose:
            print(("Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)))
        totalRewards.append(state.iteration)
    return totalRewards
