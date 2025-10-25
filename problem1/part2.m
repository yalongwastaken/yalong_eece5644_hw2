% Author: Anthony Yalong
% NUID: 002156860
% EECE5644 - Question 1, Part 2

clear; close all; clc;

%% GENERATE DATASETS

% create data
[x_train_50, y_train_50] = generate_data(50);
[x_train_500, y_train_500] = generate_data(500);
[x_train_5000, y_train_5000] = generate_data(5000);
[x_validate, y_validate] = generate_data(10000);

% theoretical optimal performance for comparison
[decisions_opt, ~] = theoretical_optimal_classifier(x_validate);
errors = (decisions_opt ~= y_validate);
min_p_error = sum(errors) / length(y_validate);

%% LOGISTIC LINEAR MODELS

% setup
datasets = {x_train_50, x_train_500, x_train_5000};
labels = {y_train_50, y_train_500, y_train_5000};
dataset_names = {'50 samples', '500 samples', '5000 samples'};
p_errors_linear = zeros(3, 1);

% train on each dataset size
for i = 1:3
    % train: minimize negative log-likelihood (MLE)
    w_linear = train_logistic_linear(datasets{i}, labels{i});

    % evaluate: classify validation set and compute error rate
    decisions_linear = classify_logistic_linear(x_validate, w_linear);
    errors = (decisions_linear ~= y_validate);
    p_errors_linear(i) = sum(errors) / length(y_validate);
end

%% LOGISTIC QUADRATIC MODELS

% setup
p_errors_quadratic = zeros(3, 1);

% train on each dataset size
for i = 1:3
    % train: minimize negative log-likelihood (MLE)
    w_quadratic = train_logistic_quadratic(datasets{i}, labels{i});

    % evaluate: classify validation set and compute error rate
    decisions_quadratic = classify_logistic_quadratic(x_validate, w_quadratic);
    errors = (decisions_quadratic ~= y_validate);
    p_errors_quadratic(i) = sum(errors) / length(y_validate);
end

%% SUMMARY RESULTS
fprintf('\nTheoretical Optimal P(error): %.4f\n', min_p_error);
fprintf('\nLogistic Linear P(error):\n');
for i = 1:3
    fprintf('  %s: %.4f\n', dataset_names{i}, p_errors_linear(i));
end
fprintf('\nLogistic Quadratic P(error):\n');
for i = 1:3
    fprintf('  %s: %.4f\n', dataset_names{i}, p_errors_quadratic(i));
end

% plot
figure();

subplot(1, 2, 1);
bar([p_errors_linear, p_errors_quadratic]);
set(gca, 'XTickLabel', {'50', '500', '5000'});
xlabel('Training Set Size');
ylabel('P(error) on Validation Set');
title('Model Performance Comparison');
legend('Linear', 'Quadratic', 'Location', 'northeast');
grid on;

subplot(1, 2, 2);
semilogy([50, 500, 5000], p_errors_linear, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
semilogy([50, 500, 5000], p_errors_quadratic, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
yline(min_p_error, 'g--', 'LineWidth', 2);
xlabel('Training Set Size');
ylabel('P(error) on Validation Set (log scale)');
title('Learning Curves');
legend('Linear', 'Quadratic', 'Theoretical Optimal', 'Location', 'northeast');
grid on;

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

function h = logistic_linear(x, w)
    %   Logistic-linear function: h(x,w) = 1/(1+exp(-w^T*z(x)))
    %   Feature vector z(x) = [1, x1, x2]^T (3 parameters)

    z = [ones(size(x, 1), 1), x];
    h = 1 ./ (1 + exp(-z * w));
end

function h = logistic_quadratic(x, w)
    %   Logistic-quadratic function: h(x,w) = 1/(1+exp(-w^T*z(x)))
    %   Feature vector z(x) = [1, x1, x2, x1^2, x1*x2, x2^2]^T (6 parameters)

    z = [ones(size(x, 1), 1), x(:,1), x(:,2), x(:,1).^2, x(:,1).*x(:,2), x(:,2).^2];
    h = 1 ./ (1 + exp(-z * w));
end

function nll = negative_log_likelihood_linear(w, x, y)
    %   Computes negative log-likelihood for logistic-linear model
    %   NLL = -sum[y*log(h) + (1-y)*log(1-h)]

    h = logistic_linear(x, w);
    h = max(min(h, 1-1e-10), 1e-10);
    nll = -sum(y .* log(h) + (1 - y) .* log(1 - h));
end

function nll = negative_log_likelihood_quadratic(w, x, y)
    %   Computes negative log-likelihood for logistic-quadratic model
    %   NLL = -sum[y*log(h) + (1-y)*log(1-h)]

    h = logistic_quadratic(x, w);
    h = max(min(h, 1-1e-10), 1e-10);
    nll = -sum(y .* log(h) + (1 - y) .* log(1 - h));
end

function w = train_logistic_linear(x, y)
    %   Trains logistic-linear model using Maximum Likelihood Estimation
    %   Uses fminsearch to minimize negative log-likelihood

    w0 = randn(3, 1) * 0.01;
    options = optimset('Display', 'off', 'MaxIter', 1000);
    w = fminsearch(@(w) negative_log_likelihood_linear(w, x, y), w0, options);
end

function w = train_logistic_quadratic(x, y)
    %   Trains logistic-quadratic model using Maximum Likelihood Estimation
    %   Uses fminsearch to minimize negative log-likelihood

    w0 = randn(6, 1) * 0.01;
    options = optimset('Display', 'off', 'MaxIter', 1000);
    w = fminsearch(@(w) negative_log_likelihood_quadratic(w, x, y), w0, options);
end

function decisions = classify_logistic_linear(x, w)
    %   Classifies samples using trained logistic-linear model
    %   Decision rule: classify as 1 if h(x,w) >= 0.5, else 0

    h = logistic_linear(x, w);
    decisions = double(h >= 0.5);
end

function decisions = classify_logistic_quadratic(x, w)
    %   Classifies samples using trained logistic-quadratic model
    %   Decision rule: classify as 1 if h(x,w) >= 0.5, else 0
    
    h = logistic_quadratic(x, w);
    decisions = double(h >= 0.5);
end