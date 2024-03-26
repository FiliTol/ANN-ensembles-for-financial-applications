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

%---------
% GERMAN |
% --------

% Replicate the data preparation as for single classifier
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


clear A1 A10 A10_encoding A12 A12_encoding A14 A14_encoding A15 A15_encoding...
    A17 A17_encoding A19_encoding A1_encoding A20_encoding A3 A3_encoding...
    A4 A4_encoding A6 A6_encoding A7 A7_encoding A9 A9_encoding german TARGET



























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