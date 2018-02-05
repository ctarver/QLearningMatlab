classdef Environment
   %Environment Class that is a space/action/reward space for an actor to
   %interface with.
   %   
   % TODO:
   % - Guarantee a fully connected environment
   % - Make it so that I an action isn't completely deterministic in
   %   where it sends the agents. 
   
   properties
      nStates
      nActions
      rewardMatrix
      transitionMatrix
      undesirableState
      desirableState
      currentState
   end
   
   methods
      function obj = Environment(nStates,nActions)
         %Environment Construct an instance of this class
         %   just radomly connect the states and actions.
         
         if nargin == 0
            obj.nStates = 5;
            obj.nActions = 2;
         else
            obj.nStates = nStates;
            obj.nActions = nActions;
         end
         
         obj.rewardMatrix = zeros(obj.nStates, ...
            obj.nActions);
         obj.transitionMatrix = randi([1 ...
            obj.nStates], ...
            obj.nStates,  ...
            obj.nActions);
         stateWithoutRewards = randi(obj.nStates);
         obj.rewardMatrix(stateWithoutRewards,:) = 0;
         undesirableState = randi(obj.nStates);
         obj.undesirableState = undesirableState;
         obj.rewardMatrix(obj.transitionMatrix == ...
            undesirableState) = -100;
         desirableState = randi(obj.nStates);
         while desirableState == undesirableState
            desirableState = randi(obj.nStates);
         end
         obj.desirableState = desirableState;
         obj.rewardMatrix(obj.transitionMatrix == ...
            desirableState) = 100;
         
         %Get an initial state
         obj.currentState = randi(obj.nStates);
      end
      
      function [obj, nextState, reward] = interfaceWithEnvironment(obj, inputAction)
         %interface We interface with the environment via this method. For
         %a given current state and action, we end up in and output a new
         %state and issue a reward.
         
         nextState = obj.transitionMatrix(obj.currentState,inputAction);
         reward = obj.rewardMatrix(obj.currentState,inputAction);
         obj.currentState = nextState;   
      end
   end
end

