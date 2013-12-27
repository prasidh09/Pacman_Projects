__author__ = 'prasidh09'
# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0

    "*** YOUR CODE HERE ***"
    #Compute the values for each iteration
    for i in range(self.iterations):
      #For each state available in the MDP we find the best action's probability and store it as the value of that state.
      next_val = self.values.copy()
      for state in self.mdp.getStates():
        best_val = None
        #Each state can have several possible actions, so find the best possible action
        for action in self.mdp.getPossibleActions(state):
          temp_value = 0
          for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
            #print nextState,prob
            #print ""
            temp_value += prob * (self.mdp.getReward(state, action, nextState) + self.discount * next_val[nextState])
          #Pick the max of all the values for each action.
          best_val = max(temp_value, best_val)
        if self.mdp.isTerminal(state):
          self.values[state] = 0
        else:
          self.values[state] = best_val


  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    #Compute the expected utility having taken action from a state
    QValue = 0
    #print self.mdp.getTransitionStatesAndProbs(state,action)
    for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
      QValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
    return QValue

    util.raiseNotDefined()

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    #Optimal action from each state is our desired policy
    possibleActions = self.mdp.getPossibleActions(state)
    if len(possibleActions) == 0:
      return None
    else:
      Q = [(self.getQValue(state, action), action) for action in possibleActions]
      #print Q
      #print max(Q)[1]
      return max(Q)[1]
    util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)

