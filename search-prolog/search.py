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


def depthFirstSearch(problem):

    from spade import pyxf
    myXsb=pyxf.xsb("/home/prasidh/Downloads/XSB/bin/xsb")
    myXsb.load("/home/prasidh/Downloads/search-source1/search/dfs.P")
    myXsb.load("/home/prasidh/Downloads/search-source1/search/maze.P")
    param = "dfs1(c"+str(problem.startState[0])+"_"+str(problem.startState[1])+",goal,Solution)"
    #param calls the prolog query that returns the dfs path
    result1=myXsb.query(param)
    #print result1
    x=[]
    x.append(result1[0])
    xy1=[0]
    xy2=[0]
    moves=[]
    #Parse the values obtained from the dfs.P file and return the directions
    for i in x:
	#print i
	x=i.values()
	#print x
	z=x[0]
	#print z
	t=z.replace('[','')
	t=t.replace(']','')
	t=t.replace('c','')
	t=t.split(',')
	#print t
	for i in t:
		i=i.split('_')
		xy1.append(i[0])
		xy2.append(i[1])
		#print i[0]
		#print i[1]
		#print xy1
		#print xy2 
    #print xy1
    #print xy2
    length=len(xy1)-1

    for i in range(1,length):
	if int(xy1[i+1])-int(xy1[i])==1:
		moves.append('East')
	if int(xy2[i+1])-int(xy2[i])==1:
		moves.append('North')
	if int(xy1[i+1])-int(xy1[i])==-1:
		moves.append('West')
	if int(xy2[i+1])-int(xy2[i])==-1:
		moves.append('South')
    return moves


def breadthFirstSearch(problem):

    from spade import pyxf
    myXsb=pyxf.xsb("/home/prasidh/Downloads/XSB/bin/xsb")
    myXsb.load("/home/prasidh/Downloads/search-source1/search/bfs2.P")
    myXsb.load("/home/prasidh/Downloads/search-source1/search/maze.P")
    param = "bfs(c"+str(problem.startState[0])+"_"+str(problem.startState[1])+",Solution)."
    #param calls the prolog query that returns the bfs path
    result1=myXsb.query(param)
    #print result1
    length=len(result1)
    #print length
    list=[]
    m=0
    y='inf'
    minlist=0
    #Find the shortest path that bfs returns
    for i in range(0,length):
	list.append(result1[i])
	#print list
	for j in list:
		l=j.values()
		#print l
		z=l[0]
		m=z.count('_')
		#print m
		if m<y:
			y=m
			minlist=i
			#print index
		#print list
	list=[]
    
    x=[]
    x.append(result1[minlist])
    xy1=[0]
    xy2=[0]
    moves=[]
    #Parse the solution to return the moves for Breadth First Search
    for i in x:
	#print i
	x=i.values()
	#print x
	z=x[0]
	#print z
	t=z.replace('[','')
	t=t.replace(']','')
	t=t.replace('c','')
	t=t.split(',')
    
	#print t
	for i in t:
		i=i.split('_')
		xy1.append(i[0])
		xy2.append(i[1])
		#print i[0]
		#print i[1]
		#print xy1
		#print xy2 
    #print xy1
    #print xy2
    length=len(xy1)-1
    for i in range(1,length):
	if int(xy1[i+1])-int(xy1[i])==1:
		moves.append('West')
	if int(xy2[i+1])-int(xy2[i])==1:
		moves.append('South')
	if int(xy1[i+1])-int(xy1[i])==-1:
		moves.append('East')
	if int(xy2[i+1])-int(xy2[i])==-1:
		moves.append('North')

	#print int(xy2[i+1])-int(xy2[i])
	#print moves
    moves.reverse()
    #print moves    
    return moves    

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
