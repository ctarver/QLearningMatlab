classdef QLearningAgent
   %QLEARNING This is a class that will generate an object with an
   %environment, agent, and perform Q Learning to develop a policy for the
   %agent to learn to choose an ideal action for a given state in the
   %environment.
   
   properties
      QMatrix
      gamma
      learningRate
      explorationRate
      nActions
      nStates
   end
   
   methods
      function obj = QLearningAgent(gamma, learningRate, nStates, nActions)
         %QLEARNING Construct an instance of this class
         %
         % Syntax: learningObj = QLearning(gamma, learningRate, nStates, nActions));
         %
         % Inputs:
         %    gamma: discount factor [0,1] where closer to 0 prioritizes
         %      immeadiate reward and closer to 1 prioritizes future rewards
         %    learningRate: [0,1] where closer to 0 makes smaller changes
         %      averaging the new calculation with the previous value for an
         %      element of the Q table and closer to 1 replaces the old value
         %      of an element with a new one.
         %    nStates: number of states in the environment
         %    nActions: number of possible actions to take
         
         clc;close all;
         
         if nargin == 0
            obj.learningRate = 0.5;
            obj.gamma = 0.5;
            obj.nStates = 5;
            obj.nActions = 2;
            obj.explorationRate = 50;
         else
            obj.gamma = gamma;
            obj.learningRate = learningRate;
         end
         
         obj = obj.initializeQMatrix(obj.nStates, obj.nActions);
      end
      
      function obj = initializeQMatrix(obj, nStates, nActions)
         %initializeQMatrix We set up the Q table by initializing all
         %1 element to be 0. We don't know the numbe of states or actions.
         %
         % Syntax: obj = obj.initializeQMatrix(nStates, nActions);
         
         obj.QMatrix = zeros(nStates, nActions);
      end
      
      function nextAction = chooseAction(obj, currentState)
         % Choose an action either through random exploration or
         % according to the Q policy
         if randi(100) > obj.explorationRate
            nextAction = randi(obj.nActions);
         else
            [~, nextAction] = max(obj.QMatrix(currentState,:));
         end
      end
      
      function obj = updateQMatrix(obj,currentState, action, reward, nextState)
         % Make an update to the QTable based on expierence and rewards
         obj.QMatrix(currentState, action) = ...
            (1-obj.learningRate)*...
            obj.QMatrix(currentState, action) + ...
            obj.learningRate*(reward + ...
            obj.gamma * max(obj.QMatrix(nextState,:)));
      end
      
   end
end