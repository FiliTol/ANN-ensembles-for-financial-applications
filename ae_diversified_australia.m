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

clear A11_encoding A12 A12_encoding A1_encoding A4_encoding A4 A5 A5_encoding...
    A6 A6_encoding A8_encoding A9_encoding australian TARGET


% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
% From now on, for every different number of multiple classifiers used for
% the majority voting process, a new dataset must be created. The new
% dataset is based on the number of classifiers in such a manner that
% every different model used for the classifiers is fueled with a different
% not-intersecting subset of observations. Lastly, a testing subset is
% extracted; the testing dataset is proportional to the train datasets,
% such that it contains half of the observations used for the train
% datasets.

% Clearly, the more the classifiers used, the less observations will be
% included in each model's train set. The assumption given in the paper is
% that the poor performance of diversified multiple classifiers is tied to
% the insufficient size of training sets, especially for classifiers with
% more models.


% -------------------------------------------------------------------------
% 3 classifiers
% -------------------------------------------------------------------------

% The number of splits is computed as n*2+1. Then every training split is
% composed by r*2 observations and the test split is composed by r
% observations.

r = floor(size(australian_df)/7);
r = r(1);

X_d1 = australian_df(1:r*2,1:end-1);
Y_d1 = australian_df(1:r*2,end);
X_d2 = australian_df(r*2+1:r*4,1:end-1);
Y_d2 = australian_df(r*2+1:r*4,end);
X_d3 = australian_df(r*4+1:r*6,1:end-1);
Y_d3 = australian_df(r*4+1:r*6,end);
X_test = australian_df(r*6+1:end,1:end-1);
Y_test = australian_df(r*6+1:end,end);

reserve = zeros(runs,1);

for i = 1:runs
    
    net1 = fitcnet(X_d1, Y_d1,...
        'LayerSizes', f(3,3).Var2(1),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,3).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_d2, Y_d2,...
        'LayerSizes', f(3,3).Var2(2),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,3).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_d3, Y_d3,...
        'LayerSizes', f(3,3).Var2(3),...
        'Activations','sigmoid',...
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

clear accuracy_best_of_3 final_decision i net1 net2 net3 prediction...
    reserve X_test Y_test X_d1 X_d2 X_d3 Y_d1 Y_d2 Y_d3

% -------------------------------------------------------------------------
% 5 classifiers
% -------------------------------------------------------------------------

% The number of splits is computed as n*2+1. Then every training split is
% composed by r*2 observations and the test split is composed by r
% observations.

r = floor(size(australian_df)/11);
r = r(1);

X_d1 = australian_df(1:r*2,1:end-1);
Y_d1 = australian_df(1:r*2,end);
X_d2 = australian_df(r*2+1:r*4,1:end-1);
Y_d2 = australian_df(r*2+1:r*4,end);
X_d3 = australian_df(r*4+1:r*6,1:end-1);
Y_d3 = australian_df(r*4+1:r*6,end);
X_d4 = australian_df(r*6+1:r*8,1:end-1);
Y_d4 = australian_df(r*6+1:r*8,end);
X_d5 = australian_df(r*8+1:r*10,1:end-1);
Y_d5 = australian_df(r*8+1:r*10,end);

X_test = australian_df(r*10+1:end,1:end-1);
Y_test = australian_df(r*10+1:end,end);

reserve = zeros(runs,1);

for i = 1:runs
    
    net1 = fitcnet(X_d1, Y_d1,...
        'LayerSizes', f(3,5).Var2(1),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,5).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_d2, Y_d2,...
        'LayerSizes', f(3,5).Var2(2),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,5).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_d3, Y_d3,...
        'LayerSizes', f(3,5).Var2(3),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,5).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_d4, Y_d4,...
        'LayerSizes', f(3,5).Var2(4),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,5).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_d5, Y_d5,...
        'LayerSizes', f(3,5).Var2(5),...
        'Activations','sigmoid',...
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


clear accuracy_best_of_5 final_decision i net1 net2 net3 net4 net5 prediction...
    reserve X_test Y_test X_d1 X_d2 X_d3 X_d4 X_d5 Y_d1 Y_d2 Y_d3 Y_d4 Y_d5


% -------------------------------------------------------------------------
% 7 classifiers
% -------------------------------------------------------------------------

% The number of splits is computed as n*2+1. Then every training split is
% composed by r*2 observations and the test split is composed by r
% observations.

r = floor(size(australian_df)/15);
r = r(1);

X_d1 = australian_df(1:r*2,1:end-1);
Y_d1 = australian_df(1:r*2,end);
X_d2 = australian_df(r*2+1:r*4,1:end-1);
Y_d2 = australian_df(r*2+1:r*4,end);
X_d3 = australian_df(r*4+1:r*6,1:end-1);
Y_d3 = australian_df(r*4+1:r*6,end);
X_d4 = australian_df(r*6+1:r*8,1:end-1);
Y_d4 = australian_df(r*6+1:r*8,end);
X_d5 = australian_df(r*8+1:r*10,1:end-1);
Y_d5 = australian_df(r*8+1:r*10,end);
X_d6 = australian_df(r*10+1:r*12,1:end-1);
Y_d6 = australian_df(r*10+1:r*12,end);
X_d7 = australian_df(r*12+1:r*14,1:end-1);
Y_d7 = australian_df(r*12+1:r*14,end);

X_test = australian_df(r*14+1:end,1:end-1);
Y_test = australian_df(r*14+1:end,end);

reserve = zeros(runs,1);

for i = 1:runs
    
    net1 = fitcnet(X_d1, Y_d1,...
        'LayerSizes', f(3,7).Var2(1),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,7).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_d2, Y_d2,...
        'LayerSizes', f(3,7).Var2(2),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,7).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_d3, Y_d3,...
        'LayerSizes', f(3,7).Var2(3),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,7).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_d4, Y_d4,...
        'LayerSizes', f(3,7).Var2(4),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,7).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_d5, Y_d5,...
        'LayerSizes', f(3,7).Var2(5),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,7).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    net6 = fitcnet(X_d6, Y_d6,...
        'LayerSizes', f(3,7).Var2(6),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,7).Var1(6),...
        'LayerBiasesInitializer','ones');
    
    net7 = fitcnet(X_d7, Y_d7,...
        'LayerSizes', f(3,7).Var2(7),...
        'Activations','sigmoid',...
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

clear accuracy_best_of_7 final_decision i net1 net2 net3 net4 net5 net6 net7 prediction...
    reserve X_test Y_test X_d1 X_d2 X_d3 X_d4 X_d5 X_d6 X_d7 Y_d1 Y_d2 Y_d3 Y_d4 Y_d5 Y_d6 Y_d7

% -------------------------------------------------------------------------
% 9 classifiers
% -------------------------------------------------------------------------

% The number of splits is computed as n*2+1. Then every training split is
% composed by r*2 observations and the test split is composed by r
% observations.

r = floor(size(australian_df)/19);
r = r(1);

X_d1 = australian_df(1:r*2,1:end-1);
Y_d1 = australian_df(1:r*2,end);
X_d2 = australian_df(r*2+1:r*4,1:end-1);
Y_d2 = australian_df(r*2+1:r*4,end);
X_d3 = australian_df(r*4+1:r*6,1:end-1);
Y_d3 = australian_df(r*4+1:r*6,end);
X_d4 = australian_df(r*6+1:r*8,1:end-1);
Y_d4 = australian_df(r*6+1:r*8,end);
X_d5 = australian_df(r*8+1:r*10,1:end-1);
Y_d5 = australian_df(r*8+1:r*10,end);
X_d6 = australian_df(r*10+1:r*12,1:end-1);
Y_d6 = australian_df(r*10+1:r*12,end);
X_d7 = australian_df(r*12+1:r*14,1:end-1);
Y_d7 = australian_df(r*12+1:r*14,end);
X_d8 = australian_df(r*14+1:r*16,1:end-1);
Y_d8 = australian_df(r*14+1:r*16,end);
X_d9 = australian_df(r*16+1:r*18,1:end-1);
Y_d9 = australian_df(r*16+1:r*18,end);

X_test = australian_df(r*18+1:end,1:end-1);
Y_test = australian_df(r*18+1:end,end);

reserve = zeros(runs,1);

for i = 1:runs
    
    net1 = fitcnet(X_d1, Y_d1,...
        'LayerSizes', f(3,9).Var2(1),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,9).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_d2, Y_d2,...
        'LayerSizes', f(3,9).Var2(2),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,9).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_d3, Y_d3,...
        'LayerSizes', f(3,9).Var2(3),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,9).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_d4, Y_d4,...
        'LayerSizes', f(3,9).Var2(4),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,9).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_d5, Y_d5,...
        'LayerSizes', f(3,9).Var2(5),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,9).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    net6 = fitcnet(X_d6, Y_d6,...
        'LayerSizes', f(3,9).Var2(6),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,9).Var1(6),...
        'LayerBiasesInitializer','ones');
    
    net7 = fitcnet(X_d7, Y_d7,...
        'LayerSizes', f(3,9).Var2(7),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,9).Var1(7),...
        'LayerBiasesInitializer','ones');
    
    net8 = fitcnet(X_d8, Y_d8,...
        'LayerSizes', f(3,9).Var2(8),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,9).Var1(8),...
        'LayerBiasesInitializer','ones');
    
    net9 = fitcnet(X_d9, Y_d9,...
        'LayerSizes', f(3,9).Var2(9),...
        'Activations','sigmoid',...
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

clear accuracy_best_of_9 final_decision i net1 net2 net3 net4 net5 net6 net7...
    net8 net9 prediction reserve X_test Y_test X_d1 X_d2 X_d3 X_d4 X_d5 X_d6...
    X_d7 X_d8 X_d9 Y_d1 Y_d2 Y_d3 Y_d4 Y_d5 Y_d6 Y_d7 Y_d8 Y_d9

% -------------------------------------------------------------------------
% 11 classifiers
% -------------------------------------------------------------------------

% The number of splits is computed as n*2+1. Then every training split is
% composed by r*2 observations and the test split is composed by r
% observations.

r = floor(size(australian_df)/23);
r = r(1);

X_d1 = australian_df(1:r*2,1:end-1);
Y_d1 = australian_df(1:r*2,end);
X_d2 = australian_df(r*2+1:r*4,1:end-1);
Y_d2 = australian_df(r*2+1:r*4,end);
X_d3 = australian_df(r*4+1:r*6,1:end-1);
Y_d3 = australian_df(r*4+1:r*6,end);
X_d4 = australian_df(r*6+1:r*8,1:end-1);
Y_d4 = australian_df(r*6+1:r*8,end);
X_d5 = australian_df(r*8+1:r*10,1:end-1);
Y_d5 = australian_df(r*8+1:r*10,end);
X_d6 = australian_df(r*10+1:r*12,1:end-1);
Y_d6 = australian_df(r*10+1:r*12,end);
X_d7 = australian_df(r*12+1:r*14,1:end-1);
Y_d7 = australian_df(r*12+1:r*14,end);
X_d8 = australian_df(r*14+1:r*16,1:end-1);
Y_d8 = australian_df(r*14+1:r*16,end);
X_d9 = australian_df(r*16+1:r*18,1:end-1);
Y_d9 = australian_df(r*16+1:r*18,end);
X_d10 = australian_df(r*18+1:r*20,1:end-1);
Y_d10 = australian_df(r*18+1:r*20,end);
X_d11 = australian_df(r*20+1:r*22,1:end-1);
Y_d11 = australian_df(r*20+1:r*22,end);

X_test = australian_df(r*22+1:end,1:end-1);
Y_test = australian_df(r*22+1:end,end);


reserve = zeros(runs,1);

for i = 1:runs

    net1 = fitcnet(X_d1, Y_d1,...
        'LayerSizes', f(3,11).Var2(1),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,11).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_d2, Y_d2,...
        'LayerSizes', f(3,11).Var2(2),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,11).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_d3, Y_d3,...
        'LayerSizes', f(3,11).Var2(3),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,11).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_d4, Y_d4,...
        'LayerSizes', f(3,11).Var2(4),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,11).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_d5, Y_d5,...
        'LayerSizes', f(3,11).Var2(5),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,11).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    net6 = fitcnet(X_d6, Y_d6,...
        'LayerSizes', f(3,11).Var2(6),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,11).Var1(6),...
        'LayerBiasesInitializer','ones');
    
    net7 = fitcnet(X_d7, Y_d7,...
        'LayerSizes', f(3,11).Var2(7),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,11).Var1(7),...
        'LayerBiasesInitializer','ones');
    
    net8 = fitcnet(X_d8, Y_d8,...
        'LayerSizes', f(3,11).Var2(8),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,11).Var1(8),...
        'LayerBiasesInitializer','ones');
    
    net9 = fitcnet(X_d9, Y_d9,...
        'LayerSizes', f(3,11).Var2(9),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,11).Var1(9),...
        'LayerBiasesInitializer','ones');
    
    net10 = fitcnet(X_d10, Y_d10,...
        'LayerSizes', f(3,11).Var2(10),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,11).Var1(10),...
        'LayerBiasesInitializer','ones');
    
    net11 = fitcnet(X_d11, Y_d11,...
        'LayerSizes', f(3,11).Var2(11),...
        'Activations','sigmoid',...
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


clear accuracy_best_of_11 final_decision i net1 net2 net3 net4 net5 net6 net7...
    net8 net9 net10 net11 prediction reserve X_test Y_test X_d1 X_d2 X_d3 X_d4 X_d5 X_d6...
    X_d7 X_d8 X_d9 X_d10 X_d11 Y_d1 Y_d2 Y_d3 Y_d4 Y_d5 Y_d6 Y_d7 Y_d8 Y_d9 Y_d10 Y_d11


% -------------------------------------------------------------------------
% 13 classifiers
% -------------------------------------------------------------------------

% The number of splits is computed as n*2+1. Then every training split is
% composed by r*2 observations and the test split is composed by r
% observations.

r = floor(size(australian_df)/27);
r = r(1);

X_d1 = australian_df(1:r*2,1:end-1);
Y_d1 = australian_df(1:r*2,end);
X_d2 = australian_df(r*2+1:r*4,1:end-1);
Y_d2 = australian_df(r*2+1:r*4,end);
X_d3 = australian_df(r*4+1:r*6,1:end-1);
Y_d3 = australian_df(r*4+1:r*6,end);
X_d4 = australian_df(r*6+1:r*8,1:end-1);
Y_d4 = australian_df(r*6+1:r*8,end);
X_d5 = australian_df(r*8+1:r*10,1:end-1);
Y_d5 = australian_df(r*8+1:r*10,end);
X_d6 = australian_df(r*10+1:r*12,1:end-1);
Y_d6 = australian_df(r*10+1:r*12,end);
X_d7 = australian_df(r*12+1:r*14,1:end-1);
Y_d7 = australian_df(r*12+1:r*14,end);
X_d8 = australian_df(r*14+1:r*16,1:end-1);
Y_d8 = australian_df(r*14+1:r*16,end);
X_d9 = australian_df(r*16+1:r*18,1:end-1);
Y_d9 = australian_df(r*16+1:r*18,end);
X_d10 = australian_df(r*18+1:r*20,1:end-1);
Y_d10 = australian_df(r*18+1:r*20,end);
X_d11 = australian_df(r*20+1:r*22,1:end-1);
Y_d11 = australian_df(r*20+1:r*22,end);
X_d12 = australian_df(r*22+1:r*24,1:end-1);
Y_d12 = australian_df(r*22+1:r*24,end);
X_d13 = australian_df(r*24+1:r*26,1:end-1);
Y_d13 = australian_df(r*24+1:r*26,end);

X_test = australian_df(r*26+1:end,1:end-1);
Y_test = australian_df(r*26+1:end,end);


reserve = zeros(runs,1);

for i = 1:runs

    net1 = fitcnet(X_d1, Y_d1,...
        'LayerSizes', f(3,13).Var2(1),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_d2, Y_d2,...
        'LayerSizes', f(3,13).Var2(2),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_d3, Y_d3,...
        'LayerSizes', f(3,13).Var2(3),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_d4, Y_d4,...
        'LayerSizes', f(3,13).Var2(4),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_d5, Y_d5,...
        'LayerSizes', f(3,13).Var2(5),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    net6 = fitcnet(X_d6, Y_d6,...
        'LayerSizes', f(3,13).Var2(6),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(6),...
        'LayerBiasesInitializer','ones');
    
    net7 = fitcnet(X_d7, Y_d7,...
        'LayerSizes', f(3,13).Var2(7),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(7),...
        'LayerBiasesInitializer','ones');
    
    net8 = fitcnet(X_d8, Y_d8,...
        'LayerSizes', f(3,13).Var2(8),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(8),...
        'LayerBiasesInitializer','ones');
    
    net9 = fitcnet(X_d9, Y_d9,...
        'LayerSizes', f(3,13).Var2(9),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(9),...
        'LayerBiasesInitializer','ones');
    
    net10 = fitcnet(X_d10, Y_d10,...
        'LayerSizes', f(3,13).Var2(10),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(10),...
        'LayerBiasesInitializer','ones');
    
    net11 = fitcnet(X_d11, Y_d11,...
        'LayerSizes', f(3,13).Var2(11),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(11),...
        'LayerBiasesInitializer','ones');
    
    net12 = fitcnet(X_d12, Y_d12,...
        'LayerSizes', f(3,13).Var2(12),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,13).Var1(12),...
        'LayerBiasesInitializer','ones');
    
    net13 = fitcnet(X_d13, Y_d13,...
        'LayerSizes', f(3,13).Var2(13),...
        'Activations','sigmoid',...
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


clear accuracy_best_of_13 final_decision i net1 net2 net3 net4 net5 net6 net7...
    net8 net9 net10 net11 net12 net13 prediction reserve X_test Y_test X_d1 X_d2 X_d3 X_d4 X_d5 X_d6...
    X_d7 X_d8 X_d9 X_d10 X_d11 X_d12 X_d13 Y_d1 Y_d2 Y_d3 Y_d4 Y_d5 Y_d6 Y_d7 Y_d8 Y_d9 Y_d10 Y_d11 Y_d12 Y_d13

% -------------------------------------------------------------------------
% 15 classifiers
% -------------------------------------------------------------------------

% The number of splits is computed as n*2+1. Then every training split is
% composed by r*2 observations and the test split is composed by r
% observations.

r = floor(size(australian_df)/31);
r = r(1);

X_d1 = australian_df(1:r*2,1:end-1);
Y_d1 = australian_df(1:r*2,end);
X_d2 = australian_df(r*2+1:r*4,1:end-1);
Y_d2 = australian_df(r*2+1:r*4,end);
X_d3 = australian_df(r*4+1:r*6,1:end-1);
Y_d3 = australian_df(r*4+1:r*6,end);
X_d4 = australian_df(r*6+1:r*8,1:end-1);
Y_d4 = australian_df(r*6+1:r*8,end);
X_d5 = australian_df(r*8+1:r*10,1:end-1);
Y_d5 = australian_df(r*8+1:r*10,end);
X_d6 = australian_df(r*10+1:r*12,1:end-1);
Y_d6 = australian_df(r*10+1:r*12,end);
X_d7 = australian_df(r*12+1:r*14,1:end-1);
Y_d7 = australian_df(r*12+1:r*14,end);
X_d8 = australian_df(r*14+1:r*16,1:end-1);
Y_d8 = australian_df(r*14+1:r*16,end);
X_d9 = australian_df(r*16+1:r*18,1:end-1);
Y_d9 = australian_df(r*16+1:r*18,end);
X_d10 = australian_df(r*18+1:r*20,1:end-1);
Y_d10 = australian_df(r*18+1:r*20,end);
X_d11 = australian_df(r*20+1:r*22,1:end-1);
Y_d11 = australian_df(r*20+1:r*22,end);
X_d12 = australian_df(r*22+1:r*24,1:end-1);
Y_d12 = australian_df(r*22+1:r*24,end);
X_d13 = australian_df(r*24+1:r*26,1:end-1);
Y_d13 = australian_df(r*24+1:r*26,end);
X_d14 = australian_df(r*26+1:r*28,1:end-1);
Y_d14 = australian_df(r*26+1:r*28,end);
X_d15 = australian_df(r*28+1:r*30,1:end-1);
Y_d15 = australian_df(r*28+1:r*30,end);

X_test = australian_df(r*30+1:end,1:end-1);
Y_test = australian_df(r*30+1:end,end);

reserve = zeros(runs,1);

for i = 1:runs

    net1 = fitcnet(X_d1, Y_d1,...
        'LayerSizes', f(3,15).Var2(1),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(1),...
        'LayerBiasesInitializer','ones');
    
    net2 = fitcnet(X_d2, Y_d2,...
        'LayerSizes', f(3,15).Var2(2),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(2),...
        'LayerBiasesInitializer','ones');
    
    net3 = fitcnet(X_d3, Y_d3,...
        'LayerSizes', f(3,15).Var2(3),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(3),...
        'LayerBiasesInitializer','ones');
    
    net4 = fitcnet(X_d4, Y_d4,...
        'LayerSizes', f(3,15).Var2(4),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(4),...
        'LayerBiasesInitializer','ones');
    
    net5 = fitcnet(X_d5, Y_d5,...
        'LayerSizes', f(3,15).Var2(5),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(5),...
        'LayerBiasesInitializer','ones');
    
    net6 = fitcnet(X_d6, Y_d6,...
        'LayerSizes', f(3,15).Var2(6),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(6),...
        'LayerBiasesInitializer','ones');
    
    net7 = fitcnet(X_d7, Y_d7,...
        'LayerSizes', f(3,15).Var2(7),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(7),...
        'LayerBiasesInitializer','ones');
    
    net8 = fitcnet(X_d8, Y_d8,...
        'LayerSizes', f(3,15).Var2(8),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(8),...
        'LayerBiasesInitializer','ones');
    
    net9 = fitcnet(X_d9, Y_d9,...
        'LayerSizes', f(3,15).Var2(9),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(9),...
        'LayerBiasesInitializer','ones');
    
    net10 = fitcnet(X_d10, Y_d10,...
        'LayerSizes', f(3,15).Var2(10),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(10),...
        'LayerBiasesInitializer','ones');
    
    net11 = fitcnet(X_d11, Y_d11,...
        'LayerSizes', f(3,15).Var2(11),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(11),...
        'LayerBiasesInitializer','ones');
    
    net12 = fitcnet(X_d12, Y_d12,...
        'LayerSizes', f(3,15).Var2(12),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(12),...
        'LayerBiasesInitializer','ones');
    
    net13 = fitcnet(X_d13, Y_d13,...
        'LayerSizes', f(3,15).Var2(13),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(13),...
        'LayerBiasesInitializer','ones');
    
    net14 = fitcnet(X_d14, Y_d14,...
        'LayerSizes', f(3,15).Var2(14),...
        'Activations','sigmoid',...
        'IterationLimit',f(3,15).Var1(14),...
        'LayerBiasesInitializer','ones');
    
    net15 = fitcnet(X_d15, Y_d15,...
        'LayerSizes', f(3,15).Var2(15),...
        'Activations','sigmoid',...
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

clear accuracy_best_of_15 final_decision i net1 net2 net3 net4 net5 net6 net7...
    net8 net9 net10 net11 net12 net13 net14 net15 prediction reserve X_test...
    Y_test X_d1 X_d2 X_d3 X_d4 X_d5 X_d6 X_d7 X_d8 X_d9 X_d10 X_d11 X_d12...
    X_d13 X_d14 X_d15 Y_d1 Y_d2 Y_d3 Y_d4 Y_d5 Y_d6 Y_d7 Y_d8 Y_d9 Y_d10...
    Y_d11 Y_d12 Y_d13 Y_d14 Y_d15

% Save results into a table to be used also for other datasets
writematrix(results,'data/diversified.csv')


















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

