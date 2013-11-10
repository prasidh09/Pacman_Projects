# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]


def Compare(v1,v2):
    return ((int(v1[0])-int(v2[0]))*(int(v1[0])-int(v2[0])))+((int(v1[1])-int(v2[1]))*(int(v1[1])-int(v2[1])))


"""def depthFirstSearch(problem):
  ""This version of the DFS traverses the Pacman through the entire DFS path""
  from game import Directions   # To make use of the directions to navigate
  from util import Stack   # To make use of the stack data structure to store nodes of the DFS Tree
  s = Directions.SOUTH
  w = Directions.WEST
  e = Directions.EAST
  n = Directions.NORTH
  DFS = Stack()   # Instantiate the stack
  Moves = []   # This list stores the moves that will be returned at the end of the DFS
  Parent = []
  Child = []
  Visited = [problem.getStartState()]   # List that stores the nodes visited by DFS
  LastState = [1,[problem.getStartState(),'',1]]   # Holds the list of the last state of the DFS
  for item in enumerate(problem.getSuccessors(problem.getStartState())):
      DFS.push(item)
  while not DFS.isEmpty():
    CurrentState = DFS.pop()
    Visited.append(CurrentState[1][0])
    #print Moves
    #print "-", CurrentState
    if Compare(CurrentState[1][0], LastState[1][0]) == 1:
        Moves.append(CurrentState[1][1])
        #print "last=",LastState,"      Current=",CurrentState
        Parent.append(LastState[1][0])
        Child.append(CurrentState[1][0])
    else:
        #print "last=",LastState
        xy = Parent[Child.index(LastState[1][0])]
        LastState1 = LastState
        while True:
            if LastState1[1][0][0] > xy[0]:
                Moves.append(w)
            elif LastState1[1][0][0] < xy[0]:
                Moves.append(e)
            elif LastState1[1][0][1] > xy[1]:
                Moves.append(s)
            elif LastState1[1][0][1] < xy[1]:
                Moves.append(n)
            #print "xy=", xy
            if Compare(CurrentState[1][0], xy) == 1:
                Parent.append(xy)
                Child.append(CurrentState[1][0])
                break
            LastState1 = [1,[xy,'',1]]
            xy = Parent[Child.index(xy)]
            #print Moves
        if CurrentState[1][0][0] > xy[0]:
            Moves.append(e)
        if CurrentState[1][0][0] < xy[0]:
            Moves.append(w)
        if CurrentState[1][0][1] > xy[1]:
            Moves.append(n)
        if CurrentState[1][0][1] < xy[1]:
            Moves.append(s)
    LastState = CurrentState
    if problem.isGoalState(CurrentState[1][0]):
        Moves.append(Directions.STOP)
        break
    for item in enumerate(problem.getSuccessors(CurrentState[1][0])):
        if Visited.count(item[1][0]) == 0:
            DFS.push(item)
  #Moves.remove('Start')
  return Moves"""


def depthFirstSearch(problem):
    "Search the shallowest nodes in the search tree first. [p 81]"
    from util import Stack
    BFS1 = Stack()
    Moves = []
    Visited = []
    Final = []
    NewState = (0, (problem.getStartState(), 'Start', 0))
    #print CurrentState
    BFS1.push([NewState])
    while not BFS1.isEmpty():
        NewState = BFS1.pop()
        if problem.isGoalState(NewState[0][1][0]):
            Final = NewState
            break
        if Visited.count(NewState[0][1][0]) == 0:
            #print NewState
            for item in enumerate(problem.getSuccessors(NewState[0][1][0])):
                #print item
                BFS1.push([item] + NewState)
        Visited.append(NewState[0][1][0])
    for nodes in Final:
        Moves.append(nodes[1][1])
    Moves.reverse()
    Moves.remove('Start')
    #print Moves
    return Moves


def breadthFirstSearch(problem):
    "Search the shallowest nodes in the search tree first. [p 81]"
    from util import Queue
    BFS1 = Queue()
    Moves = []
    Visited = []
    Final = []
    NewState = (0, (problem.getStartState(), 'Start', 0))
    #print CurrentState
    BFS1.push([NewState])
    while not BFS1.isEmpty():
        NewState = BFS1.pop()
        if problem.isGoalState(NewState[0][1][0]):
            Final = NewState
            break
        if Visited.count(NewState[0][1][0]) == 0:
            #print NewState
            for item in enumerate(problem.getSuccessors(NewState[0][1][0])):
                #print item
                BFS1.push([item] + NewState)
        Visited.append(NewState[0][1][0])
    for nodes in Final:
        Moves.append(nodes[1][1])
    Moves.reverse()
    Moves.remove('Start')
    #print Moves
    return Moves


    """def breadthFirstSearch(problem):
    ""This version of the BFS traverses the Pacman through the entire BFS path""
    from util import Queue
    BFS = Queue()
    Moves = []
    Visited = []
    LastState = (0, (problem.getStartState(), '', 0))
    BFS.push(LastState)
    while not BFS.isEmpty():
        #print BFS
        NewState = BFS.pop()
        Moves = Moves + BFS_Secondary(problem, LastState[1][0], NewState[1][0])
        LastState = NewState
        if problem.isGoalState(NewState[1][0]):
            Moves.append('Stop')
            break
        if Visited.count(NewState[1][0]) == 0:
            #Successors = problem.getSuccessors(NewState[1][0])
            print NewState
            for item in enumerate(problem.getSuccessors(NewState[1][0])):
                print item
                Succ = problem.getSuccessors(item[1][0])
                print "-", Succ
                flag = 0
                last = NewState[1]
                print "last= ", last
                while len(Succ) == 2:
                    flag = 1
                    #move = item[1][1]
                    if Succ[0][0] == last[0]:
                        move = Succ[1][1]
                    elif Succ[1][0] == last[0]:
                        move = Succ[0][1]
                    last = item
                    Visited.append(item[1][0])
                    if move == 'North':
                        y = int(item[1][0][1])
                        item = (item[0], ((item[1][0][0], y + 1), item[1][1], item[1][2]))
                    elif move == 'South':
                        y = int(item[1][0][1])
                        item = (item[0], ((item[1][0][0], y - 1), item[1][1], item[1][2]))
                    elif move == 'East':
                        x = int(item[1][0][0])
                        item = (item[0], ((x + 1, item[1][0][1]), item[1][1], item[1][2]))
                    elif move == 'West':
                        x = int(item[1][0][0])
                        item = (item[0], ((x - 1, item[1][0][1]), item[1][1], item[1][2]))
                    Succ = problem.getSuccessors(item[1][0])
                    print item[1][0], "->", Succ
                BFS.push(item)
        Visited.append(NewState[1][0])
    #print Moves
    return Moves"""


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    from game import Directions
    from util import PriorityQueue
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    UCS = PriorityQueue()
    Nodes = []
    Cost = []
    Moves = []
    Visited = []
    Final = []
    flag = 0
    CurrentState = (0, (problem.getStartState(), 'Start', 0))
    Nodes.append(problem.getStartState())
    Cost.append(0)
    UCS.push([CurrentState], 0)
    while not UCS.isEmpty():
        #print UCS
        NewState = UCS.pop()
        #print NewState
        #print Nodes, Cost
        if problem.isGoalState(NewState[0][1][0]):
            if flag == 0:
                Final = NewState
                C = Cost[Nodes.index(NewState[0][1][0])]
            else:
                flag = 1
                if Cost[Nodes.index(NewState[0][1][0])] < C:
                    Final = NewState
                    C = Cost[Nodes.index(NewState[0][1][0])]
        if Visited.count(NewState[0][1][0]) == 0:
            for item in enumerate(problem.getSuccessors(NewState[0][1][0])):
                priority = Cost[Nodes.index(NewState[0][1][0])] + item[1][2]
                UCS.push([item] + NewState, priority)
                Nodes.append(item[1][0])
                Cost.append(priority)
        Visited.append(NewState[0][1][0])
    #print Final
    for nodes in Final:
        Moves.append(nodes[1][1])
    if Moves.count('Start') > 0:
        Moves.remove('Start')
    Moves.reverse()
    print Moves
    return Moves


def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    from game import Directions
    from util import PriorityQueue
    Astar = PriorityQueue()
    Nodes = []
    Cost = []
    Moves = []
    Visited = []
    Final = []
    flag = 0
    CurrentState = (0, (problem.getStartState(), 'Start', 0))
    Nodes.append(problem.getStartState())
    Cost.append(0)
    Astar.push([CurrentState], 0)
    while not Astar.isEmpty():
        #print Astar
        NewState = Astar.pop()
        #print NewState
        #print Nodes, Cost
        if problem.isGoalState(NewState[0][1][0]):
             if flag == 0:
                Final = NewState
                break
        if Visited.count(NewState[0][1][0]) == 0:
            for item in enumerate(problem.getSuccessors(NewState[0][1][0])):
                #print heuristic(item[1][0], problem)
                priority = Cost[Nodes.index(NewState[0][1][0])] + int(item[1][2]) + heuristic(item[1][0], problem)
                Astar.push([item] + NewState, priority)
                Nodes.append(item[1][0])
                Cost.append(priority - heuristic(item[1][0], problem))
        Visited.append(NewState[0][1][0])
    #print Final
    for nodes in Final:
        Moves.append(nodes[1][1])
    if Moves.count('Start') > 0:
        Moves.remove('Start')
    Moves.reverse()
    return Moves

  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch