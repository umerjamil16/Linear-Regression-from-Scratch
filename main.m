%% ================ Part 1: Initialization ================
% Initialization
% Clear console and close all prevously opened figures
clear; 
close all; 
clc 

fprintf('Program Started...\n');
%% ================ Part 2: LOADING DATA ================
% Load Data
fprintf('Loading data ...\n');
filename = "data3.csv";
data = csvread(filename);
fprintf('DATA LOADING DONE. Press enter to continue.\n');
pause;

%% ================ Part 3: DATA CLEANING + FEATURE ENGINEERING ================

% DATA CLEANING
fprintf('Data Cleaning/Feature Engineering...\n');
data = dataCleaning_featureEngg(data);
fprintf('Data Cleaning/Feature Engineering DONE. Press enter to continue.\n');
pause;
%% ================ Part 4: DATA SPLITING ================
fprintf('Data Spliting START...\n');

% Testing/training data spliting
rng('default'); %to ensure constant seed of random gen each time the code runs
[m,n] = size(data) ;% get the size of data matrix
P = 0.80 ; %Spliting 80-20
rnd = randperm(m)  ; %Take the row number vector and randomize the row number in it
data_train = data(rnd(1:round(P*m)),:) ; %get 80% of the data
data_test = data(rnd(round(P*m)+1:end),:) ; %get 20% of the data

X_train = data_train(:, 2:end); % get feature vectors 
Y_train = data_train(:, 1); % get label vector
Y_train = log(Y_train);

X_test = data_test(:, 2:end); % get feature vectors 
Y_test = data_test(:, 1); % get label vector
Y_test = log(Y_test);

m_train = length(Y_train); % No. of training examples for train set
m_test = length(Y_test); % No. of training examples for test set

fprintf('Data Spliting DONE. Press enter to continue.\n');
pause;
%% ================ Part 5: FEATURE NORMALIZATION ================

fprintf('Feature Normalization Start. Press enter to continue.\n');
X_train_n = featureNorm(X_train); %Normalize featuers of training set
X_test_n = featureNorm(X_test); %Normalize featuers of testing set
fprintf('Feature Normalization DONE. Press enter to continue.\n');
pause;

% Create intercept term to X_train_n and X_test_n
% createMatrix(no_of_rows, no_of_cols, elem_value)
intercept_train = createMatrix(m_train, 1, 1); 
intercept_test = createMatrix(m_test, 1, 1);

% Add intercept term to X_train_n and X_test_n
X_train_n = [intercept_train X_train_n];
X_test_n = [intercept_test X_test_n];

%% ================ Part 6: Gradient Descent ================

fprintf('Running Gradient Descent ...\n');

% Choose some alpha value 
alpha = 0.01; % alpha is the learning rate
epochs  = 3000; %Number of epochs

[~, q] = size(X_train_n); %get number of columns/features of training set 
% Initilize Thetas 
theta = createMatrix(q, 1, 0);%zeros(q, 1);
[theta, J_values] = GD(X_train_n, Y_train, theta, epochs,  alpha); %Run Gradient Descent 
fprintf('Gradient Descent DONE. Press enter to continue.\n');
pause;

%% ================ Part 7: Results Printing ================
fprintf('Ploting Covergence Graph.\n');
% To Plot the convergence graph, Gradient Descent
figure;
plot(1:numel(J_values), J_values, '-r', 'LineWidth', 1.5);
xlabel('Epochs');
ylabel('Cost Function Value');

fprintf('Printing theta values obtained using Gradient Descent.\n');
fprintf('\n');
fprintf('----------------------------------\n');
fprintf('i |    theta(i)         \n');
fprintf('----------------------------------\n');
for i = 1:length(theta)
    fprintf('%i | %f \n', (i-1), theta(i));
end
fprintf('----------------------------------\n');
fprintf('\n');

%% ================ Part 7: PREDICTION ================
% Pridiction using parameters values obtained using Gradient Descent
%Using Training Set
predictedPrice_train = prediction(theta, X_train_n);
%Using Testing Set
predictedPrice_test = prediction(theta, X_test_n);

%% ================ Part 8: MODEL EVALUATION (/METRICS) ================
% To RMSE/R-Squared using predected price obtained using parameters computed by Gradient
% Descent
fprintf('MODEL EVALUATION...\n');
[r2_train, RMSE_train] = modelEval(predictedPrice_train, Y_train);
[r2_test, RMSE_test] = modelEval(predictedPrice_test, Y_test);
fprintf('MODEL EVAL DONE. Press enter to continue.\n');
pause;  

fprintf('\n');
fprintf('Evualation Metrics for parameters computed using Gradient Descent:\n');

fprintf('--------------------------------------------------\n');
fprintf('   Dataset  |         RMSE        |    R-Squared         \n');
fprintf('--------------------------------------------------\n');
fprintf('   Training |    %f    |     %f \n', RMSE_train, r2_train)
fprintf('   Testing  |    %f    |     %f \n', RMSE_test, r2_test)
fprintf('--------------------------------------------------\n');

fprintf('\n');
fprintf('--- Program Ended ---\n');
