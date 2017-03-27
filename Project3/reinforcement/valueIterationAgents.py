# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

import pdb

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
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a   dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # pdb.set_trace()

        for valueIterationIndex in range(0,self.iterations):
            valuesLast = self.values.copy()         
            for indexState in self.mdp.getStates():
                qValues = util.Counter()
                
                for indexAction in self.mdp.getPossibleActions(indexState):
                    
                    for indexSuccesorState in self.mdp.getTransitionStatesAndProbs(indexState, indexAction):
                        stateProbability = indexSuccesorState[1]
                        stateReward = self.mdp.getReward(indexState, indexAction, indexSuccesorState[0])
                        qValues[indexAction] += stateProbability * (stateReward + self.discount * valuesLast[indexSuccesorState[0]])
                if not self.mdp.isTerminal(indexState):
                    # pdb.set_trace()
                    self.values[indexState] = max(qValues.values())

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qValue = 0
        for indexSuccesorState in self.mdp.getTransitionStatesAndProbs(state, action):
            stateProbability = indexSuccesorState[1]
            stateReward = self.mdp.getReward(state, action, indexSuccesorState[0])
            qValue += stateProbability * (stateReward + self.discount * self.values[indexSuccesorState[0]])
        
        return qValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if not self.mdp.getPossibleActions(state):
            return None
        # pdb.set_trace()
        qValues = util.Counter()
        for indexAction in self.mdp.getPossibleActions(state):
            for indexSuccesorState in self.mdp.getTransitionStatesAndProbs(state, indexAction):
                stateProbability = indexSuccesorState[1]
                stateReward = self.mdp.getReward(state, indexAction, indexSuccesorState[0])
                qValues[indexAction] += stateProbability * (stateReward + self.discount * self.values[indexSuccesorState[0]])

        qValueMax = max(qValues.values())
        for qAction, qValue in qValues.iteritems():
            if qValue == qValueMax:
                return qAction

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # pdb.set_trace()
        # for valueIterationIndex in range(0,self.iterations):
        stateIndex = 0
        for valueIterationIndex in range(0, self.iterations):
            currentState = self.mdp.getStates()[stateIndex]
            stateIndex += 1
            if stateIndex == len(self.mdp.getStates()):
                stateIndex = 0
            # pdb.set_trace()
            qValues = util.Counter()
            
            for indexAction in self.mdp.getPossibleActions(currentState):
                
                for indexSuccesorState in self.mdp.getTransitionStatesAndProbs(currentState, indexAction):
                    stateProbability = indexSuccesorState[1]
                    stateReward = self.mdp.getReward(currentState, indexAction, indexSuccesorState[0])
                    qValues[indexAction] += stateProbability * (stateReward + self.discount * self.values[indexSuccesorState[0]])
            if not self.mdp.isTerminal(currentState):
                # pdb.set_trace()
                self.values[currentState] = max(qValues.values())


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        predecessors = util.Counter()
        for currentState in self.mdp.getStates():
            for currentAction in self.mdp.getPossibleActions(currentState):
                for currentSuccesorState in self.mdp.getTransitionStatesAndProbs(currentState, currentAction):
                    if not predecessors[currentSuccesorState[0]]:
                        predecessors[currentSuccesorState[0]] = set([])
                    predecessors[currentSuccesorState[0]].add(currentState)

        prioritizedStateQueue = util.PriorityQueue()

        # pdb.set_trace()
        for indexState in self.mdp.getStates():
            qValues = util.Counter()
            valuesLast = self.values.copy()         
            
            for indexAction in self.mdp.getPossibleActions(indexState):
                
                for indexSuccesorState in self.mdp.getTransitionStatesAndProbs(indexState, indexAction):
                    stateProbability = indexSuccesorState[1]
                    stateReward = self.mdp.getReward(indexState, indexAction, indexSuccesorState[0])
                    qValues[indexAction] += stateProbability * (stateReward + self.discount * valuesLast[indexSuccesorState[0]])
            if not self.mdp.isTerminal(indexState):
                # pdb.set_trace()
                diff = abs(self.values[indexState] - max(qValues.values()))
                prioritizedStateQueue.push(indexState,-diff)
                

        for valueIterationIndex in range(0, self.iterations):
            if prioritizedStateQueue.isEmpty():
                return

            currentState = prioritizedStateQueue.pop()
            if not self.mdp.isTerminal(currentState):
                qValues = util.Counter()
                valuesLast = self.values.copy()         
                for indexAction in self.mdp.getPossibleActions(currentState):

                    for indexSuccesorState in self.mdp.getTransitionStatesAndProbs(currentState, indexAction):
                        stateProbability = indexSuccesorState[1]
                        stateReward = self.mdp.getReward(currentState, indexAction, indexSuccesorState[0])
                        qValues[indexAction] += stateProbability * (stateReward + self.discount * valuesLast[indexSuccesorState[0]])
                    if not self.mdp.isTerminal(currentState):
                        self.values[currentState] = max(qValues.values())

            for predecessorState in predecessors[currentState]:             
                qValues = util.Counter()
                # valuesLast = self.values.copy()
                
                for indexAction in self.mdp.getPossibleActions(predecessorState):
                    
                    for indexSuccesorState in self.mdp.getTransitionStatesAndProbs(predecessorState, indexAction):
                        stateProbability = indexSuccesorState[1]
                        stateReward = self.mdp.getReward(predecessorState, indexAction, indexSuccesorState[0])
                        qValues[indexAction] += stateProbability * (stateReward + self.discount * self.values[indexSuccesorState[0]])
                if not self.mdp.isTerminal(predecessorState):
                    diff = abs(self.values[predecessorState] - max(qValues.values()))
                    if diff > self.theta:
                        prioritizedStateQueue.update(predecessorState,-diff)


