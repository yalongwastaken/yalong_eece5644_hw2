% Author: Anthony Yalong
% NUID: 002156860
% EECE5644 - Question 1, Part 1

clear; close all; clc;

%% GENERATE DATA
[x_validate, y_validate] = generate_data(10000);

%% THEORETICAL OPTIMAL CLASSIFIER

% apply theoretical optimal classifier
[decisions_opt, scores_opt] = theoretical_optimal_classifier(x_validate);

% calculate minimum probability of error
errors = (decisions_opt ~= y_validate);
min_p_error = sum(errors) / length(y_validate);
fprintf('Minimum P(error) = %.4f\n', min_p_error);

% compute ROC
[fpr, tpr, thresholds] = compute_roc(y_validate, scores_opt);


% create plot
figure();
plot(fpr, tpr, 'b-', 'LineWidth', 2);
hold on;

% confusion matrix
tp = sum((decisions_opt == 1) & (y_validate == 1));
fp = sum((decisions_opt == 1) & (y_validate == 0));
fn = sum((decisions_opt == 0) & (y_validate == 1));
tn = sum((decisions_opt == 0) & (y_validate == 0));
tpr_opt = tp / (tp + fn);
fpr_opt = fp / (fp + tn);

% plot
plot(fpr_opt, tpr_opt, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
plot([0, 1], [0, 1], 'g--', 'LineWidth', 1);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - Theoretical Optimal Classifier');
legend('ROC Curve', 'Min-P(error) Point', 'Random Classifier', 'Location', 'southeast');
grid on;
axis square;

%% HELPER FUNCTIONS

function [x, y] = generate_data(n)
    %   Creates N samples from a 2-class GMM
    %   Each class is a mixture of 2 Gaussian components
    %   Class 0: 60% prior probability
    %   Class 1: 40% prior probability

    % priors
    p_l0 = 0.6;
    p_l1 = 0.4;
    
    % class 0 parameters
    w01 = 0.5; w02 = 0.5;
    m01 = [-0.9; -1.1];
    m02 = [0.8; 0.75];
    c01 = [0.75, 0; 0, 1.25];
    c02 = [0.75, 0; 0, 1.25];
    
    % class 1 parameters
    w11 = 0.5; w12 = 0.5;
    m11 = [-1.1; 0.9];
    m12 = [0.9; -0.75];
    c11 = [0.75, 0; 0, 1.25];
    c12 = [0.75, 0; 0, 1.25];
    
    % randomly decide classes
    labels = rand(n, 1) < p_l0;

    % initialize data matrices
    x = zeros(n, 2);
    y = zeros(n, 1);
    
    % generate class 0 samples
    idx_l0 = find(labels);
    for i = 1:length(idx_l0)
        if rand() < w01
            x(idx_l0(i), :) = mvnrnd(m01, c01);
        else
            x(idx_l0(i), :) = mvnrnd(m02, c02);
        end
        y(idx_l0(i)) = 0;
    end
    
    % generate class 1 samples
    idx_l1 = find(~labels);
    for i = 1:length(idx_l1)
        if rand() < w11
            x(idx_l1(i), :) = mvnrnd(m11, c11);
        else
            x(idx_l1(i), :) = mvnrnd(m12, c12);
        end
        y(idx_l1(i)) = 1;
    end
end

function pdf_val = class_conditional_pdf(x, label)
    %   Computes p(x|L) for a given sample and label
    %   This is the likelihood of observing feature vector x given class label
    %   Returns weighted sum of Gaussian PDFs (mixture model)

    % class 0 parameters
    w01 = 0.5; w02 = 0.5;
    m01 = [-0.9; -1.1];
    m02 = [0.8; 0.75];
    
    % class 1 parameters
    w11 = 0.5; w12 = 0.5;
    m11 = [-1.1; 0.9];
    m12 = [0.9; -0.75];

    % shared covariance
    c = [0.75, 0; 0, 1.25];
    
    if label == 0
        % compute p(x|L=0): weighted sum of 2 Gaussians for class 0
        pdf1 = mvnpdf(x, m01', c);
        pdf2 = mvnpdf(x, m02', c);
        pdf_val = w01 * pdf1 + w02 * pdf2;
    else
        % compute p(x|L=1): weighted sum of 2 Gaussians for class 1
        pdf1 = mvnpdf(x, m11', c);
        pdf2 = mvnpdf(x, m12', c);
        pdf_val = w11 * pdf1 + w12 * pdf2;
    end
end

function [decisions, scores] = theoretical_optimal_classifier(x)
    %   Implements MAP classifier
    %   Uses Bayes' rule: P(L|x) = p(x|L) * P(L) / p(x)
    %   Decision rule: Choose class with highest posterior probability

    % prior probabilities
    p_l0 = 0.6;
    p_l1 = 0.4;
    
    % setup
    n = size(x, 1);
    decisions = zeros(n, 1);
    scores = zeros(n, 1);
    
    % classify each sample
    for i = 1:n
        % compute class conditional likelihoods
        p_x_l0 = class_conditional_pdf(x(i, :), 0);
        p_x_l1 = class_conditional_pdf(x(i, :), 1);
        
        % compute posterior probabilities
        posterior_l0 = p_l0 * p_x_l0;
        posterior_l1 = p_l1 * p_x_l1;
        
        % compute discriminant score
        decisions(i) = double(posterior_l1 > posterior_l0);
        scores(i) = log(posterior_l1 + 1e-10) - log(posterior_l0 + 1e-10);
    end
end

function [fpr, tpr, thresholds] = compute_roc(y_true, scores)
    %   Computes ROC curve by varying decision threshold
    %   For each threshold t, classify as class 1 if score >= t
    %   Calculate TPR and FPR for each threshold

    % threshold values
    thresholds = unique(scores);
    thresholds = [-inf; thresholds; inf];
    
    % setup
    fpr = zeros(length(thresholds), 1);
    tpr = zeros(length(thresholds), 1);
    
    % calculate TPR & FPR for each threshold
    for i = 1:length(thresholds)
        % predictions against threshold
        predictions = double(scores >= thresholds(i));
        
        % confusion matrix
        tp = sum((predictions == 1) & (y_true == 1));
        fp = sum((predictions == 1) & (y_true == 0));
        tn = sum((predictions == 0) & (y_true == 0));
        fn = sum((predictions == 0) & (y_true == 1));
        
        % calculate rates
        if (tp + fn) > 0
            tpr(i) = tp / (tp + fn);
        end
        if (fp + tn) > 0
            fpr(i) = fp / (fp + tn);
        end
    end
end