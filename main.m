clear;clc;close all;

% Create a random environment
env = Environment;

% Create a Q-LearningAgent
agent = QLearningAgent;

disp("Environment:   Rewards:"); 
disp([env.transitionMatrix,  env.rewardMatrix])

disp(['Desired State:', num2str(env.desirableState), ...
   '. Undesired State:' num2str(env.undesirableState)]);

%% Perform Q-Learning
nIterations = 800;

% Set up variables for analysis
arrayForm = zeros(length(agent.QMatrix(:)),nIterations);
stateArray = zeros(1,nIterations);


for i = 1:nIterations
   %Choose action based on the current state, Q-table, random exploration
   currentState = env.currentState();
   action = agent.chooseAction(currentState);
   
   [env, nextState, reward] = env.interfaceWithEnvironment(action);
   
   agent = agent.updateQMatrix(currentState,action,reward,nextState);
   
   % Record Statistics
   arrayForm(:,i) = agent.QMatrix(:);
   stateArray(i) = nextState;
end
disp("Final Q-Table:");
disp(agent.QMatrix);

%Figures
figure
plot(arrayForm');
figure
plot(stateArray);