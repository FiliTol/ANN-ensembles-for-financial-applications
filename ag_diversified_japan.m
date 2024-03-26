clc
close all
clear all

format long

% =========================================================================
% MULTIPLE CLASSIFIER
% =========================================================================

results = importdata("data/diversified.csv");

% Number of times to run the ensemble method
runs = 3;

% Function to extract best hyperparameters
f = @extract_top_values;

%-----------
% JAPANESE |
% ----------

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

clear A1 A10_encoding A12_encoding A13 A13_encoding A1_encoding A4 A4_encoding...
    A5 A5_encoding A6 A6_encoding A7 A7_encoding A9_encoding japanese TARGET



















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