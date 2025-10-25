% Author: Anthony Yalong
% NUID: 002156860
% EECE5644 - Question 3

clear; close all; clc;

%% SETUP PARAMETERS

% true vehicle position (inside unit circle)
x_true = 0.3;
y_true = 0.4;

% prior parameters
sigma_x = 0.25;
sigma_y = 0.25;

% measurement noise
sigma_r = 0.3;

% number of landmarks to test
k_values = [1, 2, 3, 4];

%% MAIN LOOP

for k = k_values
    fprintf('\n=== K = %d landmarks ===\n', k);
    
    % place landmarks evenly on unit circle
    angles = linspace(0, 2*pi, k+1);
    angles = angles(1:k);
    landmarks = [cos(angles); sin(angles)];
    
    % generate range measurements
    measurements = generate_measurements(x_true, y_true, landmarks, sigma_r);
    
    % plot MAP objective contours
    plot_map_objective(x_true, y_true, landmarks, measurements, sigma_r, sigma_x, sigma_y, k);
end

%% HELPER FUNCTIONS

function measurements = generate_measurements(x_true, y_true, landmarks, sigma_r)
    %   Generates range measurements from true position to landmarks
    %   Rejects negative measurements and resamples
    
    k = size(landmarks, 2);
    measurements = zeros(k, 1);
    
    for i = 1:k
        % true distance
        d_true = norm([x_true; y_true] - landmarks(:, i));
        
        % add Gaussian noise, resample if negative
        valid = false;
        while ~valid
            noise = sigma_r * randn();
            r = d_true + noise;
            if r >= 0
                measurements(i) = r;
                valid = true;
            end
        end
    end
end

function obj = map_objective(x, y, landmarks, measurements, sigma_r, sigma_x, sigma_y)
    %   Computes MAP objective function (negative log posterior)
    %   Simplified by removing constant terms
    
    k = size(landmarks, 2);
    
    % likelihood term: sum over all measurements
    likelihood_term = 0;
    for i = 1:k
        d_pred = norm([x; y] - landmarks(:, i));
        likelihood_term = f + ((measurements(i) - d_pred)^2) / (2 * sigma_r^2);
    end
    
    % prior term
    prior_term = (x^2) / (2 * sigma_x^2) + (y^2) / (2 * sigma_y^2);
    
    % total objective
    obj = likelihood_term + prior_term;
end

function plot_map_objective(x_true, y_true, landmarks, measurements, sigma_r, sigma_x, sigma_y, k)
    %   Plots contours of MAP objective function
    %   Shows true position and landmark locations
    
    % create grid
    x_range = linspace(-2, 2, 200);
    y_range = linspace(-2, 2, 200);
    [x_grid, y_grid] = meshgrid(x_range, y_range);
    
    % evaluate objective at each grid point
    obj_grid = zeros(size(x_grid));
    for i = 1:size(x_grid, 1)
        for j = 1:size(x_grid, 2)
            obj_grid(i, j) = map_objective(x_grid(i, j), y_grid(i, j), ...
                landmarks, measurements, sigma_r, sigma_x, sigma_y);
        end
    end
    
    % find MAP estimate (minimum of objective)
    [min_val, min_idx] = min(obj_grid(:));
    [min_i, min_j] = ind2sub(size(obj_grid), min_idx);
    x_map = x_grid(min_i, min_j);
    y_map = y_grid(min_i, min_j);
    
    fprintf('True position: (%.2f, %.2f)\n', x_true, y_true);
    fprintf('MAP estimate: (%.2f, %.2f)\n', x_map, y_map);
    fprintf('Estimation error: %.4f\n', norm([x_map - x_true; y_map - y_true]));
    
    % create figure
    figure();
    
    % define contour levels (same across all K for comparison)
    contour_levels = [1, 2, 5, 10, 20, 50, 100];
    
    % plot contours
    contour(x_grid, y_grid, obj_grid, contour_levels, 'LineWidth', 1.5);
    hold on;
    
    % plot true position
    plot(x_true, y_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
    
    % plot MAP estimate
    plot(x_map, y_map, 'g*', 'MarkerSize', 15, 'LineWidth', 2);
    
    % plot landmarks
    plot(landmarks(1, :), landmarks(2, :), 'mo', 'MarkerSize', 10, ...
        'LineWidth', 2, 'MarkerFaceColor', 'm');
    
    % formatting
    xlabel('x');
    ylabel('y');
    title(sprintf('MAP Objective Contours (K=%d landmarks)', k));
    legend('Contours', 'True Position', 'MAP Estimate', 'Landmarks', ...
        'Location', 'best');
    grid on;
    axis equal;
    xlim([-2, 2]);
    ylim([-2, 2]);
    
    % add colorbar
    colorbar;
    colormap('jet');
end