classdef QLearning
   %QLEARNING This is a class that will generate an object with an
   %environment, agent, and perform Q Learning to develop a policy for the
   %agent to learn to choose an ideal action for a given state in the
   %environment.
   
   properties
      nStates
      nActions
      environment
      QMatrix
      RMatrix
      gamma
      currentState
      learningRate
      desirableState
      undesirableState
   end
   
   methods
      function obj = QLearning(nStates, nActions, gamma, RMatrix, ...
            initialState, environment)
         %QLEARNING Construct an instance of this class
         %   Detailed explanation goes here
         
         learningRate = 0.5;
         if nargin == 0
            obj.nStates = 5;
            obj.nActions = 2;
            obj.gamma = 0.5;
            
            obj.RMatrix = zeros(obj.nStates, obj.nActions);
            obj.currentState = randi(obj.nStates);
            obj.environment = randi([1 obj.nStates], obj.nStates,  ...
               obj.nActions);
            stateWithoutRewards = randi(obj.nStates);
            obj.RMatrix(stateWithoutRewards,:) = 0;
            undesirableState = randi(obj.nStates);
            obj.undesirableState = undesirableState;
            obj.RMatrix(obj.environment==undesirableState) = -100;
            desirableState = randi(obj.nStates);
            while desirableState == undesirableState
               desirableState = randi(obj.nStates);
            end
            obj.desirableState = desirableState;
            obj.RMatrix(obj.environment==desirableState) = 100;
         else
            obj.nStates = nStates;
            obj.nActions = nActions;
            obj.gamma = gamma;
            obj.RMatrix = RMatrix;
            obj.currentState = initialState;
            obj.environment = environment;
         end
         
         obj.QMatrix = zeros(obj.nStates, obj.nActions);
         nIterations = 500;
         arrayForm = zeros(length(obj.QMatrix(:)),nIterations);
         stateArray = zeros(1,nIterations);
         explorationRate = 50;
         for i = 1:nIterations
            %Choose an action.
            [~, nextAction] = max(obj.QMatrix(obj.currentState,:));
            if randi(100) > explorationRate
               nextAction = randi(obj.nActions);
            end
            %Get next state
            nextState = obj.environment(obj.currentState,nextAction);
            
            %Update Q Matrix
            obj.QMatrix(obj.currentState, nextAction) = ...
               (1-learningRate)*...
               obj.QMatrix(obj.currentState, nextAction) + ...
               learningRate*(obj.RMatrix(obj.currentState,nextAction) + ...
               obj.gamma * max(obj.QMatrix(nextState,:)));
            arrayForm(:,i) = obj.QMatrix(:);
            stateArray(i) = nextState;
            %Update current state
            obj.currentState = nextState;
         end
         figure
         plot(arrayForm');
         figure
         plot(stateArray);
         obj
      end
      
      function outputArg = method1(obj,inputArg)
         %METHOD1 Summary of this method goes here
         %   Detailed explanation goes here
         outputArg = obj.Property1 + inputArg;
      end
   end
end

