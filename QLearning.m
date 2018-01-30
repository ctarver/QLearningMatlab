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
   end
   
   methods
      function obj = QLearning(nStates, nActions, gamma, RMatrix, ...
            initialState, environment)
         %QLEARNING Construct an instance of this class
         %   Detailed explanation goes here
         
         learningRate = 0.5;
         if nargin == 0
            obj.nStates = 4;
            obj.nActions = 4;
            obj.gamma = 0.8;
            obj.RMatrix = zeros(obj.nStates, obj.nActions);
            for i = 1:obj.nStates
               %Give a random action a positive reward
               obj.RMatrix(i,randi(obj.nActions)) = randi(100);
            end
            obj.RMatrix
            obj.currentState = randi(obj.nStates);
            obj.environment = randi([1 obj.nStates], obj.nStates,  ...
               obj.nActions);
         else
            obj.nStates = nStates;
            obj.nActions = nActions;
            obj.gamma = gamma;
            obj.RMatrix = RMatrix;
            obj.currentState = initialState;
            obj.environment = environment;
         end
         
         obj.environment
         obj.RMatrix
         obj.currentState
         
         obj.QMatrix = zeros(obj.nStates, obj.nActions);
         
         for i = 1:100
            %Choose an action.
            action = randi(obj.nActions);
            %Get next state
            nextState = obj.environment(obj.currentState,action);
            
            %Update Q Matrix
            obj.QMatrix(obj.currentState, action) = ...
               obj.RMatrix(obj.currentState,action) + ...
               obj.gamma * max(obj.QMatrix(nextState,:));
            
            %Update current state
            obj.currentState = nextState;
         end
         
         
      end
      
      function outputArg = method1(obj,inputArg)
         %METHOD1 Summary of this method goes here
         %   Detailed explanation goes here
         outputArg = obj.Property1 + inputArg;
      end
   end
end

