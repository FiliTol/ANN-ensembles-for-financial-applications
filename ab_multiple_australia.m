clc
close all
clear all

format long

% =========================================================================
% MULTIPLE CLASSIFIER
% =========================================================================

% Save the results for every country
results = zeros(8, 4);

results(2,1) = 3;
results(3,1) = 5;
results(4,1) = 7;
results(5,1) = 9;
results(6,1) = 11;
results(7,1) = 13;
results(8,1) = 15;

% Number of times to run the ensemble method
runs = 3;

% Function to extract best hyperparameters
f = @extract_top_values;

%------------
% AUSTRALIA |
% -----------

% Replicate the data preparation as for single classifier

australian = importdata('data/australian/australian.dat');
A1_encoding = double(categorical(australian(:,1)));
A4 = categorical(australian(:,4));
A4_encoding = onehotencode(A4,2);
A5 = categorical(australian(:,5));
A5_encoding = onehotencode(A5,2);
A6 = categorical(australian(:,6));
A6_encoding = onehotencode(A6,2);
A8_encoding = double(categorical(australian(:,8)));
A9_encoding = double(categorical(australian(:,9)));
A11_encoding = double(categorical(australian(:,11)));
A12 = categorical(australian(:,12));
A12_encoding = onehotencode(A12,2);
TARGET = double(categorical(australian(:,15)));

australian_df = horzcat(A1_encoding(:,1), ...
    normalize(australian(:,2)),...
    normalize(australian(:,3)),...
    A4_encoding(:,1:3),...
    A5_encoding(:,1:14),...data
    A6_encoding(:,1:8),...
    normalize(australian(:,7)),...
    A8_encoding(:,1),...
    A9_encoding(:,1),...
    normalize(australian(:,10)),...
    A11_encoding(:,1),...
    A12_encoding(:,1:3),...
    normalize(australian(:,13)),...
    normalize(australian(:,14)),...
    TARGET(:,1));

cv = cvpartition(size(australian_df,1),'HoldOut',0.3);
idx = cv.test;
dataTrain = australian_df(~idx,:);
dataTest  = australian_df(idx,:);
X_train = dataTrain(:, 1:end-1);
Y_train = dataTrain(:, end);
X_test = dataTest(:, 1:end-1);
Y_test = dataTest(:, end);

clear A11_encoding A12 A12_encoding A1_encoding A4_encoding A4 A5 A5_encoding...
    A6 A6_encoding A8_encoding A9_encoding australian cv dataTest dataTrain...
    idx TARGET

% -------------------------------------------------------------------------
% 3 classifiers
% -------------------------------------------------------------------------

% According to the saved instance of Single Classifiers these are the best
% 3 single classifiers

reserve = zeros(runs,1);

for i = 1:runs
    
    net1 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,3).Var2(1),...
        'Activations','tanh',...
        'IterationLimit',f(3,3).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,3).Var2(2),...
        'Activations','tanh',...
        'IterationLimit',f(3,3).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,3).Var2(3),...
        'Activations','tanh',...
        'IterationLimit',f(3,3).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    % Thus by comparing the prediction with and using the majority vote
    % criteria:
    prediction = [net1.predict(X_test), net2.predict(X_test), net3.predict(X_test)];
    final_decision = mode(prediction,2);
    
    accuracy_best_of_3 = sum(final_decision == Y_test)/length(final_decision);

    reserve(i,1) = accuracy_best_of_3;

end

accuracy_best_of_3 = mean(reserve);

results(2,2) = accuracy_best_of_3;

% -------------------------------------------------------------------------
% 5 classifiers
% -------------------------------------------------------------------------

reserve = zeros(runs,1);

for i = 1:runs
    
    net1 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,5).Var2(1),...
        'Activations','tanh',...
        'IterationLimit',f(3,5).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,5).Var2(2),...
        'Activations','tanh',...
        'IterationLimit',f(3,5).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,5).Var2(3),...
        'Activations','tanh',...
        'IterationLimit',f(3,5).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,5).Var2(4),...
        'Activations','tanh',...
        'IterationLimit',f(3,5).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,5).Var2(5),...
        'Activations','tanh',...
        'IterationLimit',f(3,5).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    % Thus by comparing the prediction with and using the majority vote
    % criteria:
    prediction = [net1.predict(X_test),...
        net2.predict(X_test),...
        net3.predict(X_test),...
        net4.predict(X_test),...
        net5.predict(X_test)];
    final_decision = mode(prediction,2);
    
    accuracy_best_of_5 = sum(final_decision == Y_test)/length(final_decision);

    reserve(i,1) = accuracy_best_of_5;

end

accuracy_best_of_5 = mean(reserve);

results(3,2) = accuracy_best_of_5;

% -------------------------------------------------------------------------
% 7 classifiers
% -------------------------------------------------------------------------

reserve = zeros(runs,1);

for i = 1:runs
    
    net1 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,7).Var2(1),...
        'Activations','tanh',...
        'IterationLimit',f(3,7).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,7).Var2(2),...
        'Activations','tanh',...
        'IterationLimit',f(3,7).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,7).Var2(3),...
        'Activations','tanh',...
        'IterationLimit',f(3,7).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,7).Var2(4),...
        'Activations','tanh',...
        'IterationLimit',f(3,7).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,7).Var2(5),...
        'Activations','tanh',...
        'IterationLimit',f(3,7).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    net6 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,7).Var2(6),...
        'Activations','tanh',...
        'IterationLimit',f(3,7).Var1(6),...
        'LayerBiasesInitializer','ones');
    
    net7 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,7).Var2(7),...
        'Activations','tanh',...
        'IterationLimit',f(3,7).Var1(7),...
        'LayerBiasesInitializer','ones');
    
    % Thus by comparing the prediction with and using the majority vote
    % criteria:
    prediction = [net1.predict(X_test),...
        net2.predict(X_test),...
        net3.predict(X_test),...
        net4.predict(X_test),...
        net5.predict(X_test),...
        net6.predict(X_test),...
        net7.predict(X_test)];
    final_decision = mode(prediction,2);
    
    accuracy_best_of_7 = sum(final_decision == Y_test)/length(final_decision);

    reserve(i,1) = accuracy_best_of_7;

end

accuracy_best_of_7 = mean(reserve);

results(4,2) = accuracy_best_of_7;

% -------------------------------------------------------------------------
% 9 classifiers
% -------------------------------------------------------------------------

reserve = zeros(runs,1);

for i = 1:runs
    
    net1 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,9).Var2(1),...
        'Activations','tanh',...
        'IterationLimit',f(3,9).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,9).Var2(2),...
        'Activations','tanh',...
        'IterationLimit',f(3,9).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,9).Var2(3),...
        'Activations','tanh',...
        'IterationLimit',f(3,9).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,9).Var2(4),...
        'Activations','tanh',...
        'IterationLimit',f(3,9).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,9).Var2(5),...
        'Activations','tanh',...
        'IterationLimit',f(3,9).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    net6 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,9).Var2(6),...
        'Activations','tanh',...
        'IterationLimit',f(3,9).Var1(6),...
        'LayerBiasesInitializer','ones');
    
    net7 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,9).Var2(7),...
        'Activations','tanh',...
        'IterationLimit',f(3,9).Var1(7),...
        'LayerBiasesInitializer','ones');
    
    net8 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,9).Var2(8),...
        'Activations','tanh',...
        'IterationLimit',f(3,9).Var1(8),...
        'LayerBiasesInitializer','ones');
    
    net9 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,9).Var2(9),...
        'Activations','tanh',...
        'IterationLimit',f(3,9).Var1(9),...
        'LayerBiasesInitializer','ones');
    
    
    % Thus by comparing the prediction with and using the majority vote
    % criteria:
    prediction = [net1.predict(X_test),...
        net2.predict(X_test),...
        net3.predict(X_test),...
        net4.predict(X_test),...
        net5.predict(X_test),...
        net6.predict(X_test),...
        net7.predict(X_test),...
        net8.predict(X_test),...
        net9.predict(X_test)];
    final_decision = mode(prediction,2);
    
    accuracy_best_of_9 = sum(final_decision == Y_test)/length(final_decision);

    reserve(i,1) = accuracy_best_of_9;

end

accuracy_best_of_9 = mean(reserve);

results(5,2) = accuracy_best_of_9;

% -------------------------------------------------------------------------
% 11 classifiers
% -------------------------------------------------------------------------

reserve = zeros(runs,1);

for i = 1:runs

    net1 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(1),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(2),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(3),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(4),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(5),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    net6 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(6),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(6),...
        'LayerBiasesInitializer','ones');
    
    net7 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(7),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(7),...
        'LayerBiasesInitializer','ones');
    
    net8 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(8),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(8),...
        'LayerBiasesInitializer','ones');
    
    net9 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(9),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(9),...
        'LayerBiasesInitializer','ones');
    
    net10 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(10),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(10),...
        'LayerBiasesInitializer','ones');
    
    net11 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,11).Var2(11),...
        'Activations','tanh',...
        'IterationLimit',f(3,11).Var1(11),...
        'LayerBiasesInitializer','ones');
    
    
    % Thus by comparing the prediction with and using the majority vote
    % criteria:
    prediction = [net1.predict(X_test),...
        net2.predict(X_test),...
        net3.predict(X_test),...
        net4.predict(X_test),...
        net5.predict(X_test),...
        net6.predict(X_test),...
        net7.predict(X_test),...
        net8.predict(X_test),...
        net9.predict(X_test),...
        net10.predict(X_test),...
        net11.predict(X_test)];
    final_decision = mode(prediction,2);
    
    accuracy_best_of_11 = sum(final_decision == Y_test)/length(final_decision);

    reserve(i,1) = accuracy_best_of_11;

end

accuracy_best_of_11 = mean(reserve);

results(6,2) = accuracy_best_of_11;

% -------------------------------------------------------------------------
% 13 classifiers
% -------------------------------------------------------------------------

reserve = zeros(runs,1);

for i = 1:runs

    net1 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(1),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(2),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(3),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(4),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(5),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    net6 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(6),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(6),...
        'LayerBiasesInitializer','ones');
    
    net7 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(7),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(7),...
        'LayerBiasesInitializer','ones');
    
    net8 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(8),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(8),...
        'LayerBiasesInitializer','ones');
    
    net9 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(9),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(9),...
        'LayerBiasesInitializer','ones');
    
    net10 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(10),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(10),...
        'LayerBiasesInitializer','ones');
    
    net11 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(11),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(11),...
        'LayerBiasesInitializer','ones');
    
    net12 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(12),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(12),...
        'LayerBiasesInitializer','ones');
    
    net13 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,13).Var2(13),...
        'Activations','tanh',...
        'IterationLimit',f(3,13).Var1(13),...
        'LayerBiasesInitializer','ones');
    
    
    % Thus by comparing the prediction with and using the majority vote
    % criteria:
    prediction = [net1.predict(X_test),...
        net2.predict(X_test),...
        net3.predict(X_test),...
        net4.predict(X_test),...
        net5.predict(X_test),...
        net6.predict(X_test),...
        net7.predict(X_test),...
        net8.predict(X_test),...
        net9.predict(X_test),...
        net10.predict(X_test),...
        net11.predict(X_test),...
        net12.predict(X_test),...
        net13.predict(X_test)];
    final_decision = mode(prediction,2);
    
    accuracy_best_of_13 = sum(final_decision == Y_test)/length(final_decision);

    reserve(i,1) = accuracy_best_of_13;

end

accuracy_best_of_13 = mean(reserve);

results(7,2) = accuracy_best_of_13;

% -------------------------------------------------------------------------
% 15 classifiers
% -------------------------------------------------------------------------

reserve = zeros(runs,1);

for i = 1:runs

    net1 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(1),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(2),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(3),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(4),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(5),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    net6 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(6),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(6),...
        'LayerBiasesInitializer','ones');
    
    net7 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(7),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(7),...
        'LayerBiasesInitializer','ones');
    
    net8 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(8),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(8),...
        'LayerBiasesInitializer','ones');
    
    net9 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(9),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(9),...
        'LayerBiasesInitializer','ones');
    
    net10 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(10),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(10),...
        'LayerBiasesInitializer','ones');
    
    net11 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(11),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(11),...
        'LayerBiasesInitializer','ones');
    
    net12 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(12),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(12),...
        'LayerBiasesInitializer','ones');
    
    net13 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(13),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(13),...
        'LayerBiasesInitializer','ones');
    
    net14 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(14),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(14),...
        'LayerBiasesInitializer','ones');
    
    net15 = fitcnet(X_train, Y_train,...
        'LayerSizes', f(3,15).Var2(15),...
        'Activations','tanh',...
        'IterationLimit',f(3,15).Var1(15),...
        'LayerBiasesInitializer','ones');
    
    
    % Thus by comparing the prediction with and using the majority vote
    % criteria:
    prediction = [net1.predict(X_test),...
        net2.predict(X_test),...
        net3.predict(X_test),...
        net4.predict(X_test),...
        net5.predict(X_test),...
        net6.predict(X_test),...
        net7.predict(X_test),...
        net8.predict(X_test),...
        net9.predict(X_test),...
        net10.predict(X_test),...
        net11.predict(X_test),...
        net12.predict(X_test),...
        net13.predict(X_test),...
        net14.predict(X_test),...
        net15.predict(X_test)];
    final_decision = mode(prediction,2);
    
    accuracy_best_of_15 = sum(final_decision == Y_test)/length(final_decision);

    reserve(i,1) = accuracy_best_of_15;

end

accuracy_best_of_15 = mean(reserve);

results(8,2) = accuracy_best_of_15;


% Save results into a table to be used also for other datasets
writematrix(results,'data/multiple.csv')



% The following function is used to retrieve the values of hyperparameters
% for the n-best performing models in the single classificator final table.

% For example, by calling then "f(4,5).Var1(1)" we retrieve the value of
% the epochs of the best model (1) used in the german dataset (4) column,
% given that we extracted the top 5.


function top_n_values = extract_top_values(column_num, n)
    T = readtable("data/single.csv");
    
    % Choose the column based on which you want to extract the top values
    column_of_interest = ['Var', num2str(column_num)];
    
    % Sort the table based on the chosen column
    sorted_T = sortrows(T, column_of_interest, 'descend');
    
    % Extract the top n rows
    top_n_rows = sorted_T(1:n, :);
    
    % Extract the values from the first and second columns of the top n rows
    top_n_values = top_n_rows(:, {'Var1', 'Var2'});
end







