clc
close all
clear all

format long

single = importdata("data/single.csv");
multiple = importdata("data/multiple.csv");
diversified = importdata("data/diversified.csv")

% Australian
subplot(3,1,1);
plot([repelem(max(single(:,3)),7)' multiple(2:end,2) diversified(2:end,2)], 'LineWidth', 2)
custom_xticks = 1:7;
custom_xtick_labels = {'3', '5', '7', '9', '11', '13', '15'};
xticks(custom_xticks);
xticklabels(custom_xtick_labels);
xlabel('Multiple classifier');
ylabel('Accuracy');
title('Australian');
legend('Single classifier', 'Multiple classifier', 'Diversified multiple classifier');
grid on;

% German
subplot(3,1,2);
plot([repelem(max(single(:,4)),7)' multiple(2:end,3) diversified(2:end,3)], 'LineWidth', 2)
custom_xticks = 1:7;
custom_xtick_labels = {'3', '5', '7', '9', '11', '13', '15'};
xticks(custom_xticks);
xticklabels(custom_xtick_labels);
xlabel('Multiple classifier');
ylabel('Accuracy');
title('German');
legend('Single classifier', 'Multiple classifier', 'Diversified multiple classifier');
grid on;

% Japanese
subplot(3,1,3);
plot([repelem(max(single(:,5)),7)' multiple(2:end,4)], 'LineWidth', 2)
custom_xticks = 1:7;
custom_xtick_labels = {'3', '5', '7', '9', '11', '13', '15'};
xticks(custom_xticks);
xticklabels(custom_xtick_labels);
xlabel('Multiple classifier');
ylabel('Accuracy');
title('Japanese');
legend('Single classifier', 'Multiple classifier');
grid on;