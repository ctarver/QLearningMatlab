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
      learningRate
      desirableState
      undesirableState
   end
   
   methods
      function obj = QLearning(markovDecisionProcess, gamma, RMatrix, ...
            learningRate)
         %QLEARNING Construct an instance of this class
         %
         % Syntax: learningObj = QLearning(markovDecisionProcess, gamma,
         %                           RMatrix, learningRate);
         % 
         % Inputs:
         %    markovDecisionProcess: strut with nStates, nActions, and
         %      environment defining the transitions for each state/action
         %    gamma: discount factor [0,1] where closer to 0 prioritizes
         %      immeadiate reward and closer to 1 prioritizes future rewards
         %    RMatrix: matrix of the rewards given for taking an action 
         %      at a state
         %    learningRate: [0,1] where closer to 0 makes smaller changes
         %      averaging the new calculation with the previous value for an
         %      element of the Q table and closer to 1 replaces the old value
         %      of an element with a new one.
         
         
         if nargin == 0
            obj.learningRate = 0.5;
            obj.markovDecisionProcess.nStates = 5;
            obj.markovDecisionProcess.nActions = 2;
            obj.gamma = 0.5;
            
            obj.RMatrix = zeros(obj.markovDecisionProcess.nStates, ...
               obj.markovDecisionProcess.nActions);
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
            currentState = randi(obj.markovDecisionProcess.nStates);
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
               [~, nextAction] = max(obj.QMatrix(currentState,:));
            end
            % Lookup next state for the current state and choosen action
            nextState = obj.markovDecisionProcess.environment(currentState,nextAction);
            
            % Update Q Matrix
            obj.QMatrix(currentState, nextAction) = ...
               (1-obj.learningRate)*...
               obj.QMatrix(currentState, nextAction) + ...
               obj.learningRate*(obj.RMatrix(currentState,nextAction) + ...
               obj.gamma * max(obj.QMatrix(nextState,:)));
            
            % Record Statistics
            arrayForm(:,i) = obj.QMatrix(:);
            stateArray(i) = nextState;
            
            % Update current state
            currentState = nextState;
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

