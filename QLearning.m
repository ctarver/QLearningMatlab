classdef QLearning
   %QLEARNING This is a class that will generate an object with an
   %environment, agent, and perform Q Learning to develop a policy for the
   %agent to learn to choose an ideal action for a given state in the
   %environment.
   
   properties
      markovDecisionProcess
      QMatrix
      RMatrix
      gamma
      currentState
      learningRate
      desirableState
      undesirableState
   end
   
   methods
      function obj = QLearning(markovDecisionProcess, gamma, RMatrix, ...
            initialState, learningRate)
         %QLEARNING Construct an instance of this class
         %   Detailed explanation goes here
         
         if nargin == 0
            obj.learningRate = 0.5;
            obj.markovDecisionProcess.nStates = 5;
            obj.markovDecisionProcess.nActions = 2;
            obj.gamma = 0.5;
            
            obj.RMatrix = zeros(obj.markovDecisionProcess.nStates, ...
               obj.markovDecisionProcess.nActions);
            obj.currentState = randi(obj.markovDecisionProcess.nStates);
            obj.markovDecisionProcess.environment = randi([1 ...
               obj.markovDecisionProcess.nStates], ...
               obj.markovDecisionProcess.nStates,  ...
               obj.markovDecisionProcess.nActions);
            stateWithoutRewards = randi(obj.markovDecisionProcess.nStates);
            obj.RMatrix(stateWithoutRewards,:) = 0;
            undesirableState = randi(obj.markovDecisionProcess.nStates);
            obj.undesirableState = undesirableState;
            obj.RMatrix(obj.markovDecisionProcess.environment==...
               undesirableState) = -100;
            desirableState = randi(obj.markovDecisionProcess.nStates);
            while desirableState == undesirableState
               desirableState = randi(obj.markovDecisionProcess.nStates);
            end
            obj.desirableState = desirableState;
            obj.RMatrix(obj.markovDecisionProcess.environment==...
               desirableState) = 100;
         else
            obj.markovDecisionProcess.nStates = ...
               markovDecisionProcess.nStates;
            obj.markovDecisionProcess.nActions = ...
               markovDecisionProcess.nActions;
            obj.gamma = gamma;
            obj.RMatrix = RMatrix;
            obj.currentState = initialState;
            obj.learningRate = learningRate;
         end
         
         obj = obj.initializeQMatrix();
         obj = obj.performQLearning();
         
      end
      
      function obj = initializeQMatrix(obj)
         %initializeQMatrix We set up the Q table by initializing all
         %elements to be equal to 0
         %
         % Syntax: obj = obj.initializeQMatrix();
         
         obj.QMatrix = zeros(obj.markovDecisionProcess.nStates, ...
            obj.markovDecisionProcess.nActions);
      end
      
      function obj = performQLearning(obj,nIterations,explorationRate)
         %performQLearning Performs the Q learning on the object
         %
         % Syntax: obj = obj.performQLearning(nIterations,explorationRate);
         %
         % Inputs:
         %   nIterations     - Number of iterations used to perform Q learning
         %   explorationRate - Rate at which a random action is taken. Must
         %                     be between 0 and 100.
         if nargin == 1
            nIterations = 500;
            explorationRate = 50;
         end
         
         % Set up variables for analysis
         arrayForm = zeros(length(obj.QMatrix(:)),nIterations);
         stateArray = zeros(1,nIterations);
         
         % Start Iterations
         for i = 1:nIterations            
            % Choose an action either through random exploration or
            % according to the Q policy
            if randi(100) > explorationRate
               nextAction = randi(obj.markovDecisionProcess.nActions);
            else
               [~, nextAction] = max(obj.QMatrix(obj.currentState,:));
            end
            % Lookup next state for the current state and choosen action
            nextState = obj.markovDecisionProcess.environment(obj.currentState,nextAction);
            
            %Update Q Matrix
            obj.QMatrix(obj.currentState, nextAction) = ...
               (1-obj.learningRate)*...
               obj.QMatrix(obj.currentState, nextAction) + ...
               obj.learningRate*(obj.RMatrix(obj.currentState,nextAction) + ...
               obj.gamma * max(obj.QMatrix(nextState,:)));
            arrayForm(:,i) = obj.QMatrix(:);
            stateArray(i) = nextState;
            %Update current state
            obj.currentState = nextState;
         end
         
         %Figures
         figure
         plot(arrayForm');
         figure
         plot(stateArray);
         obj
      end
   end
end

