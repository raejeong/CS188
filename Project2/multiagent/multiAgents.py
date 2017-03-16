# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import pdb
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
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        newScore = 0
        import random
        for thisGhost in newGhostStates:
          if newPos == thisGhost.configuration.pos:
            return -10000
          if manhattanDistance(newPos, thisGhost.configuration.pos) < 10:
            return -10*(1/float(manhattanDistance(newPos, thisGhost.configuration.pos)))
        
        if newPos in newFood.asList():
          newScore += 10

        foodAvgLoc = [0,0]
        for thisFood in newFood.asList():
          foodAvgLoc[0] += float(thisFood[0])/len(newFood.asList())
          foodAvgLoc[1] += float(thisFood[1])/len(newFood.asList())

        if manhattanDistance(currentGameState.getPacmanPosition(),(foodAvgLoc[0],foodAvgLoc[1])) > manhattanDistance(newPos,(foodAvgLoc[0],foodAvgLoc[1])):
          newScore += random.randint(-1,100)

        return newScore + successorGameState.getScore()


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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def isTerminalState(self,gameState,plyDepth):
        return (plyDepth == 0 or gameState.isWin() or gameState.isLose())

    def maxValue(self,gameState,plyDepth,singlePlyDepth):
        plyDepth -= 1
        singlePlyDepth = 1
        if self.isTerminalState(gameState,plyDepth):
          return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(0)
        successorStates = [gameState.generateSuccessor(0,indexAction) for indexAction in legalActions]

        valueActionList = [self.minValue(indexGameState, plyDepth, singlePlyDepth) for indexGameState in successorStates]

        return max(valueActionList)

    def minValue(self,gameState,plyDepth,singlePlyDepth):
        if self.isTerminalState(gameState,plyDepth):
          return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(singlePlyDepth)
        successorStates = [gameState.generateSuccessor(singlePlyDepth,indexAction) for indexAction in legalActions]

        if singlePlyDepth == gameState.getNumAgents()-1 :
          valueActionList = [self.maxValue(indexGameState, plyDepth, singlePlyDepth) for indexGameState in successorStates]
        else:
          valueActionList = [self.minValue(indexGameState, plyDepth, singlePlyDepth+1) for indexGameState in successorStates]

        return min(valueActionList)

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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        #pdb.set_trace()
        
        legalActions = gameState.getLegalActions(0)
        successorStates = [gameState.generateSuccessor(0,indexAction) for indexAction in legalActions]

        plyDepth = self.depth
        singlePlyDepth = 1

        valueActionList = [self.minValue(indexGameState,plyDepth,singlePlyDepth) for indexGameState in successorStates]

        correctActionIndex = valueActionList.index(max(valueActionList))

        correctAction = legalActions[correctActionIndex]

        return correctAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def isTerminalState(self,gameState,plyDepth):
        return (plyDepth == 0 or gameState.isWin() or gameState.isLose())

    def maxValue(self,gameState,plyDepth,singlePlyDepth,alpha,beta):
        plyDepth -= 1
        singlePlyDepth = 1
        if self.isTerminalState(gameState,plyDepth):
          return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(0)

        valueActionList = []

        for indexAction in legalActions:
          indexGameState = gameState.generateSuccessor(0,indexAction)
          v = self.minValue(indexGameState, plyDepth, singlePlyDepth, alpha, beta)
          valueActionList.append(v)
          if v > beta:
            return max(valueActionList)
          alpha = max(alpha,v)

        return max(valueActionList)

    def minValue(self,gameState,plyDepth,singlePlyDepth,alpha,beta):
        if self.isTerminalState(gameState,plyDepth):
          return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(singlePlyDepth)

        valueActionList = []

        if singlePlyDepth == gameState.getNumAgents()-1 :

          for indexAction in legalActions:
            indexGameState = gameState.generateSuccessor(singlePlyDepth,indexAction)
            v = self.maxValue(indexGameState, plyDepth, singlePlyDepth, alpha, beta)
            valueActionList.append(v)
            if v < alpha:
              return min(valueActionList)
            beta = min(beta,v)
        else:

          for indexAction in legalActions:
            indexGameState = gameState.generateSuccessor(singlePlyDepth,indexAction)
            v = self.minValue(indexGameState, plyDepth, singlePlyDepth+1, alpha, beta)
            valueActionList.append(v)
            if v < alpha:
              return min(valueActionList)
            beta = min(beta,v)

        return min(valueActionList)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        #pdb.set_trace()
        
        legalActions = gameState.getLegalActions(0)

        plyDepth = self.depth
        singlePlyDepth = 1

        alpha = -9999999
        beta = 9999999
        valueActionList = []

        for indexAction in legalActions:
          indexGameState = gameState.generateSuccessor(0,indexAction)
          v = self.minValue(indexGameState, plyDepth, singlePlyDepth, alpha, beta)
          valueActionList.append(v)
          if v > beta:
            break
          alpha = max(alpha,v)

        correctActionIndex = valueActionList.index(max(valueActionList))

        correctAction = legalActions[correctActionIndex]

        return correctAction

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def isTerminalState(self,gameState,plyDepth):
        return (plyDepth == 0 or gameState.isWin() or gameState.isLose())

    def maxValue(self,gameState,plyDepth,singlePlyDepth):
        plyDepth -= 1
        singlePlyDepth = 1
        if self.isTerminalState(gameState,plyDepth):
          return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(0)
        successorStates = [gameState.generateSuccessor(0,indexAction) for indexAction in legalActions]

        valueActionList = [self.minValue(indexGameState, plyDepth, singlePlyDepth) for indexGameState in successorStates]

        return max(valueActionList)

    def minValue(self,gameState,plyDepth,singlePlyDepth):
        if self.isTerminalState(gameState,plyDepth):
          return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(singlePlyDepth)
        successorStates = [gameState.generateSuccessor(singlePlyDepth,indexAction) for indexAction in legalActions]

        if singlePlyDepth == gameState.getNumAgents()-1 :
          valueActionList = [self.maxValue(indexGameState, plyDepth, singlePlyDepth) for indexGameState in successorStates]
        else:
          valueActionList = [self.minValue(indexGameState, plyDepth, singlePlyDepth+1) for indexGameState in successorStates]

        return sum(valueActionList)/float(len(valueActionList))

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        #pdb.set_trace()
        
        legalActions = gameState.getLegalActions(0)
        successorStates = [gameState.generateSuccessor(0,indexAction) for indexAction in legalActions]

        plyDepth = self.depth
        singlePlyDepth = 1

        valueActionList = [self.minValue(indexGameState,plyDepth,singlePlyDepth) for indexGameState in successorStates]

        correctActionIndex = valueActionList.index(max(valueActionList))

        correctAction = legalActions[correctActionIndex]

        return correctAction

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    #pdb.set_trace()
    moved = currentGameState.data._agentMoved
    foodList = currentGameState.data.food.asList()
    capsuleList = currentGameState.data.capsules
    pacmanPostion = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()

    closetFoodDistance = 999999
    closetCapsuleDistance = 9999999
    closetGhostDistance = 9999999
    
    for indexFood in foodList:
      closetFoodDistance = float(min(closetFoodDistance, manhattanDistance(pacmanPostion,indexFood)))+1

    for indexCapsule in capsuleList:
      closetCapsuleDistance = float(min(closetCapsuleDistance, manhattanDistance(pacmanPostion,indexCapsule)))+1  

    for indexGhost in ghostPositions:
      closetGhostDistance = float(min(closetGhostDistance, manhattanDistance(pacmanPostion,indexGhost)))

    if closetGhostDistance < 6:
      ghostFactor = -100000
    else:
      ghostFactor = 0

    import random

    return random.randint(0,80)/closetFoodDistance + random.randint(0,100)/closetCapsuleDistance + ghostFactor + 10*currentGameState.getScore()
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

