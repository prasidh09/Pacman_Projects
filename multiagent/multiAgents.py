# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        import re
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        oldGhostStates = currentGameState.getGhostStates()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        oldGhostDist = 0
        newGhostDist = 0
        score = 0
        #print oldGhostStates[0]
        #sum of ghost distances in current state
        for state in oldGhostStates:
            state = str(state)[13:]
            print str(state)
            x = re.split(',', str(state))
            #print x
            oldGhostDist += manhattanDistance((int(x[0][1]), int(x[1][1])), oldPos)
        #sum of ghost distances in new state
        for state in newGhostStates:
            state = str(state)[13:]
            x = re.split(',', str(state))
            #print newPos, (int(float(x[0][1:])), int(float(x[1][1:-1]))), "state=", state
            if newPos == (int(float(x[0][1:])), int(float(x[1][1:-1]))):
                return float("-inf")            #-inf is returned if there is a chance of the Pacman getting caught
            newGhostDist += manhattanDistance((int(float(x[0][1:])), int(float(x[1][1:-1]))), newPos)
        if newScaredTimes == [0] and newGhostDist < 3:
            score -= 7*(oldGhostDist - newGhostDist)    #this is the measure of how close the Pacman is to the ghost
        minOld = minNew = 0
        #distance to closest food in current state
        oldFoodList = oldFood.asList()
        flag = 0
        for food in oldFoodList:
            dist = manhattanDistance(food, oldPos)
            if flag == 0:
                minOld = dist
                flag = 1
            else:
                if dist < minOld:
                    minOld = dist
        #distance to closest food in new state
        newFoodList = newFood.asList()
        flag = 0
        for food in newFoodList:
            dist = manhattanDistance(food, newPos)
            if flag == 0:
                minNew = dist
                flag = 1
            else:
                if dist < minNew:
                    minNew = dist
        if newGhostDist > 2:
            score -= 10*(minNew - minOld)       #measure of how close the pacman is to the food
            score -= 1000*(len(newFoodList) - len(oldFoodList)) #if the pacman is next to 1 food then eat it
        else:
            score -= 5*(minNew - minOld)    #if pacman is close to food but ghost is also close
            score -= 6*(len(newFoodList) - len(oldFoodList))    #if pacman is next to food but ghost is also close
        #print len(newFoodList), len(oldFoodList)
        #score -= 13*(len(newFoodList) - len(oldFoodList))
        "return successorGameState.getScore()"
        #print score
        return score
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


def MiniMax(self, node, depth, agent, TotalAgents):
    f = 0
    for item in str(node):
        if item == 'V' or item == 'v' or item == '^' or item == '<' or item == '>':
            f = 1
    #print node, f
    if depth == 0 or f == 0:
        #print agent, TotalAgents
        return self.evaluationFunction(node)
    if agent == 0:
        bestValue = float("-inf")
        #bestValue = -99999
        actions = node.getLegalActions(agent)
        for action in actions:
            childState = node.generateSuccessor(agent, action)
            val = MiniMax(self, childState, depth - 1, (agent + 1)%TotalAgents, TotalAgents)
            bestValue = max(bestValue, val)
        return bestValue
    else:
        bestValue = float("inf")
        #bestValue = 99999
        actions = node.getLegalActions(agent)
        for action in actions:
            childState = node.generateSuccessor(agent, action)
            val = MiniMax(self, childState, depth - 1, (agent + 1)%TotalAgents, TotalAgents)
            bestValue = min(bestValue, val)
        return bestValue


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        agents = gameState.getNumAgents()
        d = self.depth*agents - 1
        actions = gameState.getLegalActions(0)
        flag = 0
        move = ''
        for action in actions:          #find the best Minimax child
            child = gameState.generateSuccessor(0, action)
            val = MiniMax(self, child, d - 1, 1, agents)
            print
            if flag == 0:
                m = val
                move = action
                flag = 1
            else:
                if val > m:
                    m = val
                    move = action
        return move
        #print self.evaluationFunction(gameState)
        #print gameState

def alpha_beta(self, node, depth, a, b, agent, TotalAgents):
    f = 0
    for item in str(node):
        if item == 'V' or item == 'v' or item == '^' or item == '<' or item == '>':
            f = 1
    if depth == 0 or f == 0:
        return self.evaluationFunction(node)
    if agent == 0:
        actions = node.getLegalActions(agent)
        for action in actions:
            childState = node.generateSuccessor(agent, action)
            a = max(a, alpha_beta(self, childState, depth - 1, a, b, (agent + 1)%TotalAgents, TotalAgents))
            if b <= a:
                break   #b pruning
        return a
    else:
        actions = node.getLegalActions(agent)
        for action in actions:
            childState = node.generateSuccessor(agent, action)
            b = min(b, alpha_beta(self, childState, depth - 1, a, b, (agent + 1)%TotalAgents, TotalAgents))
            if b <= a:
                break   #a pruning
        return b

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        agents = gameState.getNumAgents()
        d = self.depth*agents - 1
        actions = gameState.getLegalActions(0)
        flag = 0
        move = ''
        for action in actions:              #find the best minimax child
            child = gameState.generateSuccessor(0, action)
            val = alpha_beta(self, child, d - 1, float("-inf"), float("inf"), 1, agents)
            #print val
            if flag == 0:
                m = val
                move = action
                flag = 1
            else:
                if val > m:
                    m = val
                    move = action
        return move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

