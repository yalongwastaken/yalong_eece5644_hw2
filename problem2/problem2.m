% Author: Anthony Yalong
% NUID: 002156860
% EECE5644 - Question 2

clear; close all; clc;

%% GENERATE DATASETS
n_train = 100;
n_validate = 1000;
[x_train, y_train, x_validate, y_validate] = hw2q2(n_train, n_validate);

%% MAXIMUM LIKELIHOOD (ML) ESTIMATOR

% train model
w_ml = train_ml(x_train, y_train);
fprintf('ML Estimator:\n');
fprintf('w = [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]^T\n', w_ml);

% evaluate on validation set
y_pred_ml = predict_cubic(x_validate, w_ml);
mse_ml = mean((y_validate - y_pred_ml).^2);
fprintf('ML Validation MSE: %.4f\n\n', mse_ml);

%% MAXIMUM A POSTERIORI (MAP) ESTIMATOR

% range of gamma values
gamma_values = logspace(-4, 4, 50);
mse_map_values = zeros(length(gamma_values), 1);
w_map_all = zeros(10, length(gamma_values));

% train MAP models for different gamma values
for i = 1:length(gamma_values)
    gamma = gamma_values(i);
    w_map = train_map(x_train, y_train, gamma);
    w_map_all(:, i) = w_map;
    
    % evaluate on validation set
    y_pred_map = predict_cubic(x_validate, w_map);
    mse_map_values(i) = mean((y_validate - y_pred_map).^2);
end

% find optimal gamma
[mse_map_best, idx_best] = min(mse_map_values);
gamma_best = gamma_values(idx_best);
w_map_best = w_map_all(:, idx_best);

fprintf('Best MAP Estimator:\n');
fprintf('Optimal gamma: %.4e\n', gamma_best);
fprintf('Best MAP Validation MSE: %.4f\n', mse_map_best);
fprintf('ML Validation MSE: %.4f\n\n', mse_ml);

%% VISUALIZATIONS

% plot MSE vs gamma
figure();
semilogx(gamma_values, mse_map_values, 'b-', 'LineWidth', 2);
hold on;
yline(mse_ml, 'r--', 'LineWidth', 2);
plot(gamma_best, mse_map_best, 'go', 'MarkerSize', 12, 'LineWidth', 3);
xlabel('Hyperparameter \gamma');
ylabel('Validation MSE');
title('MAP Performance vs Regularization Strength');
legend('MAP MSE', 'ML MSE', 'Optimal \gamma', 'Location', 'best');
grid on;

% plot parameter magnitudes vs gamma
figure();
for j = 1:10
    semilogx(gamma_values, abs(w_map_all(j,:)), 'LineWidth', 1.5);
    hold on;
end
xlabel('Hyperparameter \gamma');
ylabel('|w_i|');
title('Parameter Magnitudes vs \gamma');
grid on;
legend('w_0', 'w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6', 'w_7', 'w_8', 'w_9', ...
    'Location', 'best');

% visualize predictions
figure();

subplot(1, 2, 1);
scatter3(x_validate(1,:), x_validate(2,:), y_validate, 20, 'b', 'filled');
hold on;
scatter3(x_validate(1,:), x_validate(2,:), y_pred_ml, 20, 'r', 'filled');
xlabel('x_1'); ylabel('x_2'); zlabel('y');
title(sprintf('ML Predictions (MSE=%.2f)', mse_ml));
legend('True', 'Predicted', 'Location', 'best');
grid on;
view(45, 30);

subplot(1, 2, 2);
scatter3(x_validate(1,:), x_validate(2,:), y_validate, 20, 'b', 'filled');
hold on;
scatter3(x_validate(1,:), x_validate(2,:), predict_cubic(x_validate, w_map_best), ...
    20, 'r', 'filled');
xlabel('x_1'); ylabel('x_2'); zlabel('y');
title(sprintf('MAP Predictions (MSE=%.2f, \\gamma=%.2e)', mse_map_best, gamma_best));
legend('True', 'Predicted', 'Location', 'best');
grid on;
view(45, 30);

%% HELPER FUNCTIONS

function Z = create_feature_matrix(x)
    % Creates feature matrix for cubic polynomial
    % z(x) = [1, x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3]^T
    
    x1 = x(1, :)';
    x2 = x(2, :)';
    
    Z = [ones(size(x1)), ...
         x1, x2, ...
         x1.^2, x1.*x2, x2.^2, ...
         x1.^3, x1.^2.*x2, x1.*x2.^2, x2.^3];
end

function w_ml = train_ml(x, y)
    % Maximum Likelihood estimator for cubic polynomial
    % Minimizes: sum((y - w^T*z(x))^2)
    % Solution: w_ml = (Z^T * Z)^-1 * Z^T * y
    
    Z = create_feature_matrix(x);
    w_ml = (Z' * Z) \ (Z' * y');
end

function w_map = train_map(x, y, gamma)
    % Maximum A Posteriori estimator with Gaussian prior
    % Prior: w ~ N(0, gamma*I)
    % Minimizes: sum((y - w^T*z(x))^2) + (1/gamma)*||w||^2
    % Solution: w_map = (Z^T*Z + (1/gamma)*I)^-1 * Z^T * y
    
    Z = create_feature_matrix(x);
    d = size(Z, 2);
    w_map = (Z' * Z + (1/gamma) * eye(d)) \ (Z' * y');
end

function y_pred = predict_cubic(x, w)
    % Predicts y values using cubic polynomial model
    % y = w^T * z(x)
    
    Z = create_feature_matrix(x);
    y_pred = (Z * w)';
end

function [x_train, y_train, x_validate, y_validate] = hw2q2(n_train, n_validate)
    %   Generates training and validation datasets
    %   Returns feature vectors and corresponding target values

    % generate training data
    data = generate_data(n_train);
    figure();
    plot3(data(1,:), data(2,:), data(3,:), '.', 'MarkerSize', 10);
    axis equal;
    xlabel('x_1'); 
    ylabel('x_2'); 
    zlabel('y');
    title('Training Dataset');
    grid on;
    x_train = data(1:2, :);
    y_train = data(3, :);
    
    % generate validation data
    data = generate_data(n_validate);
    figure();
    plot3(data(1,:), data(2,:), data(3,:), '.', 'MarkerSize', 10);
    axis equal;
    xlabel('x_1'); 
    ylabel('x_2'); 
    zlabel('y');
    title('Validation Dataset');
    grid on;
    x_validate = data(1:2, :);
    y_validate = data(3, :);
end

function x = generate_data(N)
    %   Creates N samples from a 3-component GMM
    %   Returns [x1; x2; y] where y is cubic function of x with noise

    % GMM parameters
    gmm_parameters.priors = [.3, .4, .3];
    gmm_parameters.mean_vectors = [-10 0 10; 0 0 0; 10 0 -10];
    gmm_parameters.cov_matrices(:,:,1) = [1 0 -3; 0 1 0; -3 0 15];
    gmm_parameters.cov_matrices(:,:,2) = [8 0 0; 0 .5 0; 0 0 .5];
    gmm_parameters.cov_matrices(:,:,3) = [1 0 -3; 0 1 0; -3 0 15];
    
    % generate samples
    [x, labels] = generate_data_from_gmm(N, gmm_parameters);
end

function [x, labels] = generate_data_from_gmm(N, gmm_parameters)
    %   Generates N samples from Gaussian Mixture Model
    %   Returns samples and their component labels

    % extract parameters
    priors = gmm_parameters.priors;
    mean_vectors = gmm_parameters.mean_vectors;
    cov_matrices = gmm_parameters.cov_matrices;
    n = size(gmm_parameters.mean_vectors, 1);
    C = length(priors);
    
    % initialize
    x = zeros(n, N);
    labels = zeros(1, N);
    
    % randomly assign samples to components
    u = rand(1, N);
    thresholds = [cumsum(priors), 1];
    
    % generate samples for each component
    for l = 1:C
        indl = find(u <= thresholds(l));
        Nl = length(indl);
        labels(1, indl) = l * ones(1, Nl);
        u(1, indl) = 1.1 * ones(1, Nl);
        x(:, indl) = mvnrnd(mean_vectors(:,l), cov_matrices(:,:,l), Nl)';
    end
end