clc
close all
clear all

format long

final_errors = zeros(2,4);
final_errors(1,1) = 1;
final_errors(2,1) = 2;

runs = 5;
epochs = [50 100 200 300];
hidden_nodes = [8 12 16 24 32];

% -------------------------------------------------------------------------
% Australian
% -------------------------------------------------------------------------

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

results = zeros(4*5, 2);

for e = 1:length(epochs)
    epoch = epochs(e);

    for h = 1:length(hidden_nodes)
        node = hidden_nodes(h);

        partial_results = zeros(runs,2);

        for j = 1:runs

            net = fitcnet(X_train, Y_train,...
                'LayerSizes', node,...
                'Activations','tanh',...
                'IterationLimit',epoch,...
                'LayerBiasesInitializer','ones');
            
            predicted_labels = net.predict(X_test);
            
            conf_matrix = confusionmat(Y_test, predicted_labels);
            
            true_positive = conf_matrix(2, 2); % Actual positive and predicted positive
            false_positive = conf_matrix(1, 2); % Actual negative but predicted positive
            true_negative = conf_matrix(1, 1); % Actual negative and predicted negative
            false_negative = conf_matrix(2, 1); % Actual positive but predicted negative
            
            % Compute Type I error (False Positive Rate)
            type1_error = false_positive / (false_positive + true_negative);
            
            % Compute Type II error (False Negative Rate)
            type2_error = false_negative / (false_negative + true_positive);

            partial_results(j, 1) = type1_error;
            partial_results(j, 2) = type2_error;
            
            
        end
        base = (e - 1) * 5;
        results(base + h, 1) = mean(partial_results(:,1));
        results(base + h, 2) = mean(partial_results(:,2));
    end
end

% Add mean error to final table for Australian
final_errors(1,2) = mean(results(:,1));
final_errors(2,2) = mean(results(:,2));

clear A11_encoding A12 A12_encoding A1_encoding A4 A4_encoding A5 A5_encoding...
    A6 A6_encoding A8_encoding A9_encoding australian base conf_matrix...
    cv dataTest dataTrain e epoch false_positive false_negative...
    h idx j net node partial_results predicted_labels results...
    TARGET true_negative true_positive type1_error type2_error X_test X_train...
    Y_test Y_train

% -------------------------------------------------------------------------
% German
% -------------------------------------------------------------------------

german = importdata('data/german/german.dat');
A1 = categorical(german.textdata(:,1));
A1_encoding = onehotencode(A1,2);
A3 = categorical(german.textdata(:,3));
A3_encoding = onehotencode(A3,2);
A4 = categorical(german.textdata(:,4));
A4_encoding = onehotencode(A4,2);
A6 = categorical(german.textdata(:,6));
A6_encoding = onehotencode(A6,2);
A7 = categorical(german.textdata(:,7));
A7_encoding = onehotencode(A7,2);
A9 = categorical(german.textdata(:,9));
A9_encoding = onehotencode(A9,2);
A10 = categorical(german.textdata(:,10));
A10_encoding = onehotencode(A10,2);
A12 = categorical(german.textdata(:,12));
A12_encoding = onehotencode(A12,2);
A14 = categorical(german.textdata(:,14));
A14_encoding = onehotencode(A14,2);
A15 = categorical(german.textdata(:,15));
A15_encoding = onehotencode(A15,2);
A17 = categorical(german.textdata(:,17));
A17_encoding = onehotencode(A17,2);
A19_encoding = double(categorical(german.textdata(:,19)));
A20_encoding = double(categorical(german.textdata(:,20)));
TARGET = double(categorical(german.data(:,1)));

german_df = horzcat(A1_encoding(:,1:4),...
    normalize(str2double(german.textdata(:,2))),...
    A3_encoding(:,1:5),...
    A4_encoding(:,1:10),...
    normalize(str2double(german.textdata(:,5))),...
    A6_encoding(:,1:5),...
    A7_encoding(:,1:5),...
    normalize(str2double(german.textdata(:,8))),...
    A9_encoding(:,1:4),...
    A10_encoding(:,1:3),...
    normalize(str2double(german.textdata(:,11))),...
    A12_encoding(:,1:4),...
    normalize(str2double(german.textdata(:,13))),...
    A14_encoding(:,1:3),...
    A15_encoding(:,1:3),...
    normalize(str2double(german.textdata(:,16))),...
    A17_encoding(:,1:4),...
    normalize(str2double(german.textdata(:,18))),...
    A19_encoding(:,1),...
    A20_encoding(:,1),...
    TARGET(:,1));

cv = cvpartition(size(german_df,1),'HoldOut',0.3);
idx = cv.test;
dataTrain = german_df(~idx,:);
dataTest  = german_df(idx,:);
X_train = dataTrain(:, 1:end-1);
Y_train = dataTrain(:, end);
X_test = dataTest(:, 1:end-1);
Y_test = dataTest(:, end);

results = zeros(4*5, 2);

for e = 1:length(epochs)
    epoch = epochs(e);

    for h = 1:length(hidden_nodes)
        node = hidden_nodes(h);

        partial_results = zeros(runs,2);

        for j = 1:runs

            net = fitcnet(X_train, Y_train,...
                'LayerSizes', node,...
                'Activations','tanh',...
                'IterationLimit',epoch,...
                'LayerBiasesInitializer','ones');
            
            predicted_labels = net.predict(X_test);
            
            conf_matrix = confusionmat(Y_test, predicted_labels);
            
            true_positive = conf_matrix(2, 2); % Actual positive and predicted positive
            false_positive = conf_matrix(1, 2); % Actual negative but predicted positive
            true_negative = conf_matrix(1, 1); % Actual negative and predicted negative
            false_negative = conf_matrix(2, 1); % Actual positive but predicted negative
            
            % Compute Type I error (False Positive Rate)
            type1_error = false_positive / (false_positive + true_negative);
            
            % Compute Type II error (False Negative Rate)
            type2_error = false_negative / (false_negative + true_positive);

            partial_results(j, 1) = type1_error;
            partial_results(j, 2) = type2_error;       
            
        end
        base = (e - 1) * 5;
        results(base + h, 1) = mean(partial_results(:,1));
        results(base + h, 2) = mean(partial_results(:,2));
    end
end

% Add mean error to final table for Australian
final_errors(1,3) = mean(results(:,1));
final_errors(2,3) = mean(results(:,2));


clear A1 A10 A10_encoding A12 A12_encoding A14 A14_encoding A15 A15_encoding...
    A17 A17_encoding A19_encoding A1_encoding A20_encoding A3 A3_encoding...
    A4 A4_encoding A6 A6_encoding A7 A7_encoding A9 A9_encoding base cv dataTest...
    dataTrain e epoch false_negative false_positive german h idx j net node...
    partial_results predicted_labels results TARGET true_negative true_positive...
    type1_error type2_error X_test X_train Y_test Y_train conf_matrix


% -------------------------------------------------------------------------
% Japanese
% -------------------------------------------------------------------------

opts = delimitedTextImportOptions("NumVariables", 16);
opts.DataLines = [1, Inf];
opts.Delimiter = " ";
opts.VariableNames = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "TARGET"];
opts.VariableTypes = ["categorical", "double", "double", "categorical", "categorical", "categorical", "categorical", "double", "categorical", "categorical", "double", "categorical", "categorical", "double", "double", "categorical"];
opts.ImportErrorRule = "omitrow";
opts.MissingRule = "omitrow";
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";
opts = setvaropts(opts, ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13", "TARGET"], "EmptyFieldRule", "auto");
japanese = readtable("data/japanese/original/data.dat", opts);
clear opts
A1 = categorical(japanese.A1);
A1_encoding = onehotencode(A1,2);
A4 = categorical(japanese.A4);
A4_encoding = onehotencode(A4,2);
A5 = categorical(japanese.A5);
A5_encoding = onehotencode(A5,2);
A6 = categorical(japanese.A6);
A6_encoding = onehotencode(A6,2);
A7 = categorical(japanese.A7);
A7_encoding = onehotencode(A7,2);
A9_encoding = double(categorical(japanese.A9));
A10_encoding = double(categorical(japanese.A10));
A12_encoding = double(categorical(japanese.A12));
A13 = categorical(japanese.A13);
A13_encoding = onehotencode(A13,2);
TARGET = double(categorical(japanese.TARGET));

japanese_df = horzcat(A1_encoding(:,1:3),...
    normalize(japanese.A2),...
    normalize(japanese.A3),...
    A4_encoding(:,1:4),...
    A5_encoding(:,1:4),...
    A6_encoding(:,1:15),...
    A7_encoding(:,1:10),...
    normalize(japanese.A8),...
    A9_encoding(:,1),...
    A10_encoding(:,1),...
    normalize(japanese.A11),...
    A12_encoding(:,1),...
    A13_encoding(:,1:3),...
    normalize(japanese.A14),...
    normalize(japanese.A15),...
    TARGET(:,1));

cv = cvpartition(size(japanese_df,1),'HoldOut',0.3);
idx = cv.test;
dataTrain = japanese_df(~idx,:);
dataTest  = japanese_df(idx,:);
X_train = dataTrain(:, 1:end-1);
Y_train = dataTrain(:, end);
X_test = dataTest(:, 1:end-1);
Y_test = dataTest(:, end);

results = zeros(4*5, 2);

for e = 1:length(epochs)
    epoch = epochs(e);

    for h = 1:length(hidden_nodes)
        node = hidden_nodes(h);

        partial_results = zeros(runs,2);

        for j = 1:runs

            net = fitcnet(X_train, Y_train,...
                'LayerSizes', node,...
                'Activations','tanh',...
                'IterationLimit',epoch,...
                'LayerBiasesInitializer','ones');
            
            predicted_labels = net.predict(X_test);
            
            conf_matrix = confusionmat(Y_test, predicted_labels);
            
            true_positive = conf_matrix(2, 2); % Actual positive and predicted positive
            false_positive = conf_matrix(1, 2); % Actual negative but predicted positive
            true_negative = conf_matrix(1, 1); % Actual negative and predicted negative
            false_negative = conf_matrix(2, 1); % Actual positive but predicted negative
            
            % Compute Type I error (False Positive Rate)
            type1_error = false_positive / (false_positive + true_negative);
            
            % Compute Type II error (False Negative Rate)
            type2_error = false_negative / (false_negative + true_positive);

            partial_results(j, 1) = type1_error;
            partial_results(j, 2) = type2_error;
            
            
        end
        base = (e - 1) * 5;
        results(base + h, 1) = mean(partial_results(:,1));
        results(base + h, 2) = mean(partial_results(:,2));
    end
end

% Add mean error to final table for Australian
final_errors(1,4) = mean(results(:,1));
final_errors(2,4) = mean(results(:,2));


clear A1 A1_encoding A10_encoding A12_encoding A13 A13_encoding A1 A4 A4_encoding A5...
    A5_encoding A6 A6_encoding A7 A7_encoding A9_encoding base conf_matrix...
    cv dataTest dataTrain e epoch false_negative false_positive h idx...
    j japanese net node partial_results predicted_labels results TARGET...
    true_negative true_positive type2_error type1_error X_test X_train...
    Y_test Y_train


writematrix(final_errors,"data/single_errors.csv")



























