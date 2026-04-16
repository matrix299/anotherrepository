clc; clear; close all;

%% ============================================================
% PRI_MIMO_TIR_AltOpt_TBD_DOA - MULTI-TARGET VERSION
% Extended target with multiple scatterers, each tracked via TBD
%% ============================================================

%% ---------------- Parameters ----------------
%rng(20260409);

Nt = 8;
Nr = 8;
N  = 16;
L  = 20;  % 增加延迟分辨率，避免散射点耦合
num_frames = 300;
K  = num_frames; % 关键修改：令 K 等于帧数，确保 1 PRI = 1 Frame
P  = 6;

T = 1e-6;
fs = N / T;
t_fast = (0:N-1).' / fs;

f0 = 0;
B = 5e6;
chirp_rate = B / T;

lambda_c = 1e-3;
d = lambda_c / 2;

sigma_v2 = 1;
sigma_c2 = 0;
rho = 0.05;

delta = 0.5;
zeta = 1e-3;
max_iter = 30;

gm_channel_rho = 0.95;
est_cfg.q_alpha = max(1 - gm_channel_rho^2, 1e-4);
est_cfg.p0_alpha = 1;

%% ---------------- TBD settings ----------------
theta_grid_deg = linspace(5, 85, 241);
theta_grid = theta_grid_deg * pi / 180;
theta_proc_sigma_deg = 0.20;
omega_proc_sigma_deg = 0.03;

dptbd_cfg.sigma_theta_pred = 1.0 * pi / 180;
dptbd_cfg.sigma_omega = 0.3 * pi / 180;
dptbd_cfg.init_omega_sigma = 0.5 * pi / 180;
dptbd_cfg.gate_theta_cells = 5;
dptbd_cfg.gate_omega_cells = 5;
dptbd_cfg.local_score_floor = 0;

beta_amp_sigma = 0.04;
beta_phase_sigma = 0.25;

max_jump_deg = 10.0;
sigma_jump_deg = 4.0;

%% ---------------- CVX check ----------------
if exist('cvx_begin', 'file') == 0
    error('CVX was not detected. Please install/setup CVX first.');
end

%% ---------------- Initial LFM waveform ----------------
s_lfm = exp(1j * 2 * pi * (f0 * t_fast + 0.5 * chirp_rate * t_fast.^2));

S0_mat = zeros(Nt, N);
for nt = 1:Nt
    S0_mat(nt, :) = s_lfm.' * exp(1j * 2 * pi * (nt - 1) / Nt);
end

s0 = reshape(S0_mat, N * Nt, 1);
s0 = s0 / norm(s0);

%% ============================================================
% 1. Front-end design
%% ============================================================

[H_all, ~, pri_param_all] = build_pri_channel_sequence_gm( ...
    Nt, Nr, N, L, P, K, lambda_c, d, gm_channel_rho);

[alpha_true_seq_design, theta_true_seq_design, delay_true_design] = ...
    extract_scatter_truth(pri_param_all);

H_scatter_basis = build_scatter_basis_from_truth( ...
    pri_param_all{1}.theta, pri_param_all{1}.delay, Nt, Nr, N, L, lambda_c, d);

% DEBUG: Print scatterer information
fprintf('\n=== Scatterer Information ===\n');
for p = 1:P
    fprintf('Scatterer %d: theta=%.2f deg, delay=%d, |alpha|=%.4f\n', ...
        p, theta_true_seq_design(p,1)*180/pi, delay_true_design(p), ...
        abs(alpha_true_seq_design(p,1)));
end

R_spatial = rho .^ abs((0:Nr-1).' - (0:Nr-1));
R_total = sigma_c2 * R_spatial + sigma_v2 * eye(Nr);
R_full = kron(eye(N), R_total);

if rcond(R_total) < 1e-10
    R_total = R_total + 1e-6 * eye(Nr);
    R_full = kron(eye(N), R_total);
end

[w_step0, ~] = update_receive_filters_paper(H_all, s0, R_full, Nr, N);
[~, sinr_step0_raw_worst] = evaluate_sinr_full(H_all, s0, w_step0, R_full);
[~, sinr_step0_kf_vals, meas_step0, kf_step0, kf_cov_step0, ~, ~, ~, ~] = ...
    evaluate_sinr_scatter_kf(H_scatter_basis, alpha_true_seq_design, ...
    s0, w_step0, R_full, gm_channel_rho, est_cfg);
sinr_step0_kf_worst = min(sinr_step0_kf_vals);

fprintf('\nFront-end optimization...\n');
fprintf('Step 0 raw worst-case SINR: %.4f dB\n', 10 * log10(real(sinr_step0_raw_worst)));
fprintf('Step 0 KF-receiver worst-case SINR: %.4f dB\n', 10 * log10(real(sinr_step0_kf_worst)));

s_opt = s0;
w_opt = w_step0;
history_raw = sinr_step0_raw_worst;
history_kf = sinr_step0_kf_worst;
meas_seq_design = meas_step0;
kf_out_seq_design = kf_step0;
kf_cov_seq_design = kf_cov_step0;

for iter = 1:max_iter
    s_prev = s_opt;
    w_prev = w_opt;
    prev_metric = history_kf(end);

    [~, ~, ~, ~, ~, ~, ~, H_post_curr, ~] = ...
        evaluate_sinr_scatter_kf(H_scatter_basis, alpha_true_seq_design, ...
        s_prev, w_prev, R_full, gm_channel_rho, est_cfg);

    s_candidate = optimize_waveform_no_kf(H_post_curr, w_prev, R_full, s_prev, s0, delta);

    [~, ~, ~, ~, ~, ~, ~, H_post_for_w, H_pred_for_w] = ...
        evaluate_sinr_scatter_kf(H_scatter_basis, alpha_true_seq_design, ...
        s_candidate, w_prev, R_full, gm_channel_rho, est_cfg);

    H_pred_for_w{1} = H_post_for_w{1};

    [w_candidate, ~] = update_receive_filters_paper(H_pred_for_w, s_candidate, R_full, Nr, N);

    [~, sinr_candidate_raw_worst] = ...
        evaluate_sinr_full(H_all, s_candidate, w_candidate, R_full);
    [~, sinr_candidate_kf_vals, meas_candidate, kf_candidate, kf_cov_candidate, ...
        ~, ~, ~, ~] = ...
        evaluate_sinr_scatter_kf(H_scatter_basis, alpha_true_seq_design, ...
        s_candidate, w_candidate, R_full, gm_channel_rho, est_cfg);
    sinr_candidate_kf_worst = min(sinr_candidate_kf_vals);

    if real(sinr_candidate_kf_worst) + 1e-10 < real(prev_metric)
        warning('Iteration %d violates monotonicity safeguard. Reject update.', iter);
        break;
    end

    s_opt = s_candidate;
    w_opt = w_candidate;
    history_raw(end+1, 1) = sinr_candidate_raw_worst;
    history_kf(end+1, 1) = sinr_candidate_kf_worst;
    meas_seq_design = meas_candidate;
    kf_out_seq_design = kf_candidate;
    kf_cov_seq_design = kf_cov_candidate;

    fprintf('Iter %2d: raw SINR = %8.4f dB, KF-receiver SINR = %8.4f dB\n', ...
        iter, 10 * log10(real(history_raw(end))), 10 * log10(real(history_kf(end))));

    if numel(history_kf) >= 2 && ...
            real(history_kf(end)) - real(history_kf(end-1)) <= zeta
        fprintf('Frontend alternating optimization converged at iteration %d.\n', iter);
        break;
    end
end

final_frontend_raw_db = 10 * log10(real(history_raw(end)));
final_frontend_kf_db = 10 * log10(real(history_kf(end)));

fprintf('\nFinal frontend performance:\n');
fprintf('Final raw worst-case SINR: %.4f dB\n', final_frontend_raw_db);
fprintf('Final KF-receiver worst-case SINR: %.4f dB\n', final_frontend_kf_db);

%% ============================================================
% 2. Generate true trajectory for ALL scatterers
%% ============================================================

% All scatterers share the SAME angular motion (synchronized)
% because they belong to the same extended target.
% Each scatterer has a fixed offset from a reference DOA.

theta_min = theta_grid(1);
theta_max = theta_grid(end);

% Generate initial DOA for each scatterer with sufficient separation
% to avoid coupling (at least 10 degrees apart)
theta_offset = zeros(P, 1);
theta_start_ref = 45;  % Reference starting DOA in degrees

for p = 1:P
    % Spread scatterers across different angles with sufficient separation
    theta_offset(p) = (p - (P+1)/2) * 12;  % 12 degrees separation
end

% Shared angular velocity trajectory for all scatterers
omega_shared = zeros(num_frames, 1);
omega_shared(1) = 1 * pi / 180;  % Initial angular velocity

for frame_idx = 2:num_frames
    omega_shared(frame_idx) = omega_shared(frame_idx - 1) + ...
        omega_proc_sigma_deg * pi / 180 * randn;
end

% Reflect at boundaries to keep within valid range
theta_pred_shared = zeros(num_frames, 1);
theta_pred_shared(1) = theta_start_ref * pi / 180;

for frame_idx = 2:num_frames
    theta_pred = theta_pred_shared(frame_idx - 1) + omega_shared(frame_idx - 1) + ...
        theta_proc_sigma_deg * pi / 180 * randn;
    
    if theta_pred < theta_min || theta_pred > theta_max
        omega_shared(frame_idx) = -omega_shared(frame_idx);
        theta_pred = theta_pred_shared(frame_idx - 1) + omega_shared(frame_idx) + ...
            theta_proc_sigma_deg * pi / 180 * randn;
    end
    
    theta_pred_shared(frame_idx) = min(max(theta_pred, theta_min), theta_max);
end

% Now generate individual scatterer trajectories with shared motion + fixed offset
theta_true_all = zeros(num_frames, P);
amp_true_all = zeros(num_frames, P);
phase_true_all = zeros(num_frames, P);

for p = 1:P
    for frame_idx = 1:num_frames
        % Each scatterer follows the same angular motion but with its own offset
        theta_true_all(frame_idx, p) = theta_pred_shared(frame_idx) + theta_offset(p) * pi / 180;
        
        % Ensure within bounds
        theta_true_all(frame_idx, p) = min(max(theta_true_all(frame_idx, p), theta_min), theta_max);
    end
    
    % Amplitude and phase are independent for each scatterer
    amp_true_all(1, p) = 1.0;
    phase_true_all(1, p) = 0;
    
    fprintf('Scatterer %d initial DOA: %.2f deg\n', p, theta_true_all(1,p)*180/pi);
    
    for frame_idx = 2:num_frames
        amp_true_all(frame_idx, p) = max(0.6, amp_true_all(frame_idx - 1, p) + beta_amp_sigma * randn);
        phase_true_all(frame_idx, p) = phase_true_all(frame_idx - 1, p) + beta_phase_sigma * randn;
    end
end

beta_true_all = amp_true_all .* exp(1j * phase_true_all);

fprintf('\n=== True Trajectories (Synchronized Motion) ===\n');
for p = 1:P
    fprintf('Scatterer %d: Start=%.2f deg, End=%.2f deg, Offset=%.1f deg\n', ...
        p, theta_true_all(1,p)*180/pi, theta_true_all(end,p)*180/pi, theta_offset(p));
end

%% ============================================================
% 3. Build score maps for EACH scatterer (严格 1 PRI = 1 Frame)
%% ============================================================


num_grid = numel(theta_grid);

% Precompute steering vectors for each scatterer (each has different delay)
a_grid_all = zeros(N * Nr, num_grid, P);
for p = 1:P
    delay_eq = delay_true_design(p);
    for g = 1:num_grid
        H_unit = build_point_target_channel(1, delay_eq, theta_grid(g), ...
            Nt, Nr, N, L, lambda_c, d);
        a_grid_all(:, g, p) = H_unit * s_opt;
    end
end

% Noise covariance processing
R_full_herm = (R_full + R_full') / 2;
if rcond(R_full_herm) < 1e-10
    R_full_herm = R_full_herm + 1e-6 * eye(size(R_full_herm, 1));
end
L_noise = chol(R_full_herm, 'lower');

% Score map for each scatterer: [num_frames x num_grid x P]
score_map_all = zeros(num_frames, num_grid, P);
frame_ml_idx_all = zeros(num_frames, P);

for p = 1:P
    delay_eq = delay_true_design(p);
    
    for frame_idx = 1:num_frames
        % Generate received vector for current frame with all scatterers
        % But we compute matched filter output for scatterer p specifically
        H_true_total = zeros(N * Nr, N * Nt);
        for pp = 1:P
            H_pp = build_point_target_channel(beta_true_all(frame_idx, pp), ...
                delay_true_design(pp), theta_true_all(frame_idx, pp), ...
                Nt, Nr, N, L, lambda_c, d);
            H_true_total = H_true_total + H_pp;
        end
        
        noise_white = (randn(N * Nr, 1) + 1j * randn(N * Nr, 1)) / sqrt(2);
        noise_vec = L_noise * noise_white;
        y_fk = H_true_total * s_opt + noise_vec;
        
        % GLRT score for scatterer p's grid points
        Rinv_y = R_full_herm \ y_fk;           % M×1
        Rinv_a = R_full_herm \ a_grid_all(:, :, p);  % M×num_grid

        num_vec = abs((a_grid_all(:, :, p)' * Rinv_y).').^2;   % 1×num_grid
        den_vec = real(sum(conj(a_grid_all(:, :, p)) .* Rinv_a, 1)); % 1×num_grid

        den_vec = max(den_vec, 1e-12);

        score_map_all(frame_idx, :, p) = num_vec ./ den_vec;

        [~, frame_ml_idx_all(frame_idx, p)] = max(score_map_all(frame_idx, :, p));
    end
end

theta_frame_ml_all = zeros(num_frames, P);
for p = 1:P
    theta_frame_ml_all(:, p) = theta_grid(frame_ml_idx_all(:, p));
end

%% ============================================================
% 4. Multi-target TBD: Run TBD for EACH scatterer independently
%% ============================================================

tbd_track_idx_all = zeros(num_frames, P);
cumulative_score_all = zeros(num_frames, num_grid, P);

for p = 1:P
    % Use the true initial DOA of scatterer p for initialization
    theta_init = theta_true_all(1, p);
    
    [tbd_track_idx_all(:, p), cumulative_score_all(:, :, p)] = run_tbd_viterbi( ...
        score_map_all(:, :, p), theta_grid, max_jump_deg * pi / 180, ...
        sigma_jump_deg * pi / 180, theta_init);
end

theta_tbd_all = zeros(num_frames, P);
for p = 1:P
    theta_tbd_all(:, p) = theta_grid(tbd_track_idx_all(:, p));
end

% Compute RMSE for each scatterer and average
rmse_ml_deg_per_scatterer = zeros(P, 1);
rmse_tbd_deg_per_scatterer = zeros(P, 1);

for p = 1:P
    rmse_ml_deg_per_scatterer(p) = sqrt(mean((theta_frame_ml_all(:, p) - theta_true_all(:, p)).^2)) * 180 / pi;
    rmse_tbd_deg_per_scatterer(p) = sqrt(mean((theta_tbd_all(:, p) - theta_true_all(:, p)).^2)) * 180 / pi;
end

rmse_ml_deg = mean(rmse_ml_deg_per_scatterer);
rmse_tbd_deg = mean(rmse_tbd_deg_per_scatterer);

fprintf('\n=== Results ===\n');
fprintf('Per-scatterer ML DOA RMSE (deg):\n');
for p = 1:P
    fprintf('  Scatterer %d: %.4f deg\n', p, rmse_ml_deg_per_scatterer(p));
end
fprintf('  Average: %.4f deg\n', rmse_ml_deg);

fprintf('\nPer-scatterer TBD DOA RMSE (deg):\n');
for p = 1:P
    fprintf('  Scatterer %d: %.4f deg\n', p, rmse_tbd_deg_per_scatterer(p));
end
fprintf('  Average: %.4f deg\n', rmse_tbd_deg);

%% ============================================================
% 5. Visualization (same format as original)
%% ============================================================

% Plot for first scatterer as representative example
p_plot = 1;

figure('Name', 'Score Map Debug', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 400]);

subplot(1, 2, 1);
plot(theta_grid_deg, score_map_all(1, :, p_plot), 'b-', 'LineWidth', 1.5);
hold on;
xline(theta_true_all(1, p_plot)*180/pi, 'r-', 'True DOA', 'LineWidth', 2);
xline(theta_frame_ml_all(1, p_plot)*180/pi, 'g--', 'ML Estimate', 'LineWidth', 2);
grid on;
xlabel('DOA (deg)');
ylabel('Score');
title(sprintf('Frame 1 Score (Scatterer %d): True=%.1f deg, ML=%.1f deg', ...
    p_plot, theta_true_all(1, p_plot)*180/pi, theta_frame_ml_all(1, p_plot)*180/pi));
legend('Score', 'True', 'ML');

subplot(1, 2, 2);
score_slice = score_map_all(:, :, p_plot);
max_val = max(score_slice(:));
if max_val > 1e-12
    score_map_db = 10 * log10(score_slice / max_val + 1e-12);
else
    score_map_db = 10 * log10(score_slice + 1e-12);
end
imagesc(1:num_frames, theta_grid_deg, score_map_db);
axis xy;
colorbar;
hold on;
plot(1:num_frames, theta_true_all(:, p_plot) * 180 / pi, 'w-', 'LineWidth', 2.5);
plot(1:num_frames, theta_frame_ml_all(:, p_plot) * 180 / pi, 'yo', 'MarkerSize', 4);
xlabel('Frame index');
ylabel('DOA (deg)');
title(sprintf('Score Map Heatmap (Scatterer %d)', p_plot));
legend('True Trajectory', 'ML Estimates');

figure('Name', 'DOA Tracking Comparison', 'NumberTitle', 'off');
plot(1:num_frames, theta_true_all(:, p_plot) * 180 / pi, 'k-', 'LineWidth', 2.0); hold on;
plot(1:num_frames, theta_frame_ml_all(:, p_plot) * 180 / pi, 'bo--', ...
    'LineWidth', 1.4, 'MarkerSize', 5);
plot(1:num_frames, theta_tbd_all(:, p_plot) * 180 / pi, 'rs-', ...
    'LineWidth', 1.8, 'MarkerSize', 5);
grid on;
xlabel('Frame index');
ylabel('DOA (deg)');
title(sprintf('DOA Tracking (Scatterer %d): ML RMSE = %.3f deg, TBD RMSE = %.3f deg', ...
    p_plot, rmse_ml_deg_per_scatterer(p_plot), rmse_tbd_deg_per_scatterer(p_plot)));
legend('True DOA', 'Frame-by-frame ML', 'TBD track', 'Location', 'best');

%% Additional figure: All scatterers tracking comparison
figure('Name', 'All Scatterers Tracking', 'NumberTitle', 'off', 'Position', [100, 100, 1400, 800]);

colors = lines(P);
for p = 1:P
    subplot(ceil(P/2), 2, p);
    plot(1:num_frames, theta_true_all(:, p) * 180 / pi, 'k-', 'LineWidth', 2.0); hold on;
    plot(1:num_frames, theta_frame_ml_all(:, p) * 180 / pi, 'bo--', ...
        'LineWidth', 1.0, 'MarkerSize', 3, 'Color', colors(p,:));
    plot(1:num_frames, theta_tbd_all(:, p) * 180 / pi, 'rs-', ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'Color', colors(p,:));
    grid on;
    xlabel('Frame index');
    ylabel('DOA (deg)');
    title(sprintf('Scatterer %d: ML=%.3f°, TBD=%.3f°', ...
        p, rmse_ml_deg_per_scatterer(p), rmse_tbd_deg_per_scatterer(p)));
    legend('True', 'ML', 'TBD', 'Location', 'best');
end

%% ============================================================
% Local functions
%% ============================================================

function [H_all, H_l_all, pri_param_all] = build_pri_channel_sequence_gm( ...
    Nt, Nr, N, L, P, K, lambda_c, d, gm_rho)

    n_t = (0:Nt-1).';
    n_r = (0:Nr-1).';

    delay = randi([0, L-1], P, 1);
    theta = rand(P, 1) * pi / 2;

    at = zeros(Nt, P);
    ar = zeros(Nr, P);
    for p_idx = 1:P
        at(:, p_idx) = exp(1j * 2 * pi / lambda_c * d * n_t * sin(theta(p_idx)));
        ar(:, p_idx) = exp(1j * 2 * pi / lambda_c * d * n_r * sin(theta(p_idx)));
    end

    alpha_seq = zeros(P, K);
    alpha_seq(:, 1) = (randn(P, 1) + 1j * randn(P, 1)) / sqrt(2);
    for pri_idx = 2:K
        innovation = (randn(P, 1) + 1j * randn(P, 1)) / sqrt(2);
        alpha_seq(:, pri_idx) = gm_rho * alpha_seq(:, pri_idx - 1) + ...
            sqrt(max(1 - gm_rho^2, 0)) * innovation;
    end

    H_all = cell(K, 1);
    H_l_all = cell(K, 1);
    pri_param_all = cell(K, 1);

    for pri_idx = 1:K
        H_l = cell(L, 1);
        for ell = 1:L
            H_l{ell} = zeros(Nr, Nt);
        end

        for p_idx = 1:P
            ell = delay(p_idx) + 1;
            H_l{ell} = H_l{ell} + alpha_seq(p_idx, pri_idx) * (ar(:, p_idx) * at(:, p_idx)');
        end

        H = zeros(N * Nr, N * Nt);
        for row = 1:N
            for col = 1:N
                ell = row - col + 1;
                if ell >= 1 && ell <= L
                    r_idx = (row - 1) * Nr + (1:Nr);
                    c_idx = (col - 1) * Nt + (1:Nt);
                    H(r_idx, c_idx) = H_l{ell};
                end
            end
        end

        H_all{pri_idx} = H;
        H_l_all{pri_idx} = H_l;

        pri_param.alpha = alpha_seq(:, pri_idx);
        pri_param.delay = delay;
        pri_param.theta = theta;
        pri_param.gm_rho = gm_rho;
        pri_param_all{pri_idx} = pri_param;
    end
end

function [alpha_true_seq, theta_true_seq, delay_true] = extract_scatter_truth(pri_param_all)

    K = numel(pri_param_all);
    P = numel(pri_param_all{1}.alpha);

    alpha_true_seq = zeros(P, K);
    theta_true_seq = zeros(P, K);

    for pri_idx = 1:K
        alpha_true_seq(:, pri_idx) = pri_param_all{pri_idx}.alpha(:);
        theta_true_seq(:, pri_idx) = pri_param_all{pri_idx}.theta(:);
    end

    delay_true = pri_param_all{1}.delay(:);
end

function H_scatter_basis = build_scatter_basis_from_truth(theta_vec, delay_vec, Nt, Nr, N, L, lambda_c, d)

    P = numel(theta_vec);
    n_t = (0:Nt-1).';
    n_r = (0:Nr-1).';
    H_scatter_basis = cell(P, 1);

    for p_idx = 1:P
        at = exp(1j * 2 * pi / lambda_c * d * n_t * sin(theta_vec(p_idx)));
        ar = exp(1j * 2 * pi / lambda_c * d * n_r * sin(theta_vec(p_idx)));

        H_l = cell(L, 1);
        for ell = 1:L
            H_l{ell} = zeros(Nr, Nt);
        end
        ell = delay_vec(p_idx) + 1;
        H_l{ell} = ar * at';

        H_basis = zeros(N * Nr, N * Nt);
        for row = 1:N
            for col = 1:N
                lag_idx = row - col + 1;
                if lag_idx >= 1 && lag_idx <= L
                    r_idx = (row - 1) * Nr + (1:Nr);
                    c_idx = (col - 1) * Nt + (1:Nt);
                    H_basis(r_idx, c_idx) = H_l{lag_idx};
                end
            end
        end

        H_scatter_basis{p_idx} = H_basis;
    end
end

function [w_list, peak_bins] = update_receive_filters_paper(H_all, s, R_full, Nr, N)

    K = numel(H_all);
    w_list = cell(K, 1);
    peak_bins = compute_reference_peak_bins(H_all, s, Nr, N);

    R_herm = (R_full + R_full') / 2;
    if rcond(R_herm) < 1e-10
        R_herm = R_herm + 1e-6 * eye(size(R_herm, 1));
    end

    for pri_idx = 1:K
        h_eff = H_all{pri_idx} * s;
        w_num = R_herm \ h_eff;
        w_den = sqrt(max(real(w_num' * R_herm * w_num), 1e-12));
        w_i = w_num / w_den;

        resp_i = w_i' * h_eff;
        if abs(resp_i) > 1e-12
            w_i = w_i * exp(-1j * angle(resp_i));
        end

        w_list{pri_idx} = w_i;
    end
end

function peak_bins = compute_reference_peak_bins(H_all, s, Nr, N)

    K = numel(H_all);
    peak_bins = zeros(K, 1);

    for pri_idx = 1:K
        r_vec = H_all{pri_idx} * s;
        r_mat = reshape(r_vec, Nr, N);
        col_energy = sum(abs(r_mat).^2, 1);
        [~, peak_bins(pri_idx)] = max(col_energy);
    end
end

function s_opt = optimize_waveform_no_kf(H_all, w_list, R_full, s_prev, s0, delta)

    signal_dim = length(s_prev);
    K = numel(H_all);
    c_list = cell(K, 1);
    beta_list = zeros(K, 1);

    for pri_idx = 1:K
        c_i = H_all{pri_idx}' * w_list{pri_idx};
        resp_i = c_i' * s_prev;
        if abs(resp_i) > 1e-12
            c_i = c_i * exp(-1j * angle(resp_i));
        end
        c_list{pri_idx} = c_i;
        beta_list(pri_idx) = max(real(w_list{pri_idx}' * R_full * w_list{pri_idx}), 1e-12);
    end

    cvx_begin quiet
        variable s_var(signal_dim) complex
        variable t

        maximize(t)
        subject to
            norm(s_var, 2) <= 1;
            norm(s_var - s0, 2) <= delta;

            for pri_idx = 1:K
                real(c_list{pri_idx}' * s_var) >= t * sqrt(beta_list(pri_idx));
                real(c_list{pri_idx}' * s_var) >= 0;
            end
    cvx_end

    if contains(cvx_status, 'Solved')
        s_opt = s_var;
    else
        warning('CVX status is %s. Keep previous waveform.', cvx_status);
        s_opt = s_prev;
    end
end

function [sinr_vals, sinr_worst] = evaluate_sinr_full(H_all, s, w_list, R_full)

    K = numel(H_all);
    sinr_vals = zeros(K, 1);

    for pri_idx = 1:K
        h_eff = H_all{pri_idx} * s;
        w_i = w_list{pri_idx};

        num_val = abs(w_i' * h_eff)^2;
        den_val = max(real(w_i' * R_full * w_i), 1e-12);
        sinr_vals(pri_idx) = real(num_val / den_val);
    end

    sinr_worst = min(sinr_vals);
end

function [sinr_raw_vals, sinr_kf_vals, meas_seq, kf_out_seq, post_var_seq, ...
    alpha_post_seq, alpha_pred_seq, H_post_all, H_pred_all] = ...
    evaluate_sinr_scatter_kf(H_scatter_basis, alpha_true_seq, s, w_list, R_full, gm_rho, est_cfg)

    P = numel(H_scatter_basis);
    K = numel(w_list);

    sinr_raw_vals = zeros(K, 1);
    sinr_kf_vals = zeros(K, 1);
    meas_seq = zeros(K, 1);
    kf_out_seq = zeros(K, 1);
    post_var_seq = zeros(K, 1);
    alpha_post_seq = zeros(P, K);
    alpha_pred_seq = zeros(P, K);
    H_post_all = cell(K, 1);
    H_pred_all = cell(K, 1);

    alpha_post_prev = zeros(P, 1);
    P_post_prev = est_cfg.p0_alpha * eye(P);
    Q_alpha = est_cfg.q_alpha * eye(P);

    for pri_idx = 1:K
        c_k = zeros(P, 1);
        for p_idx = 1:P
            c_k(p_idx) = w_list{pri_idx}' * H_scatter_basis{p_idx} * s;
        end

        z_signal = c_k' * alpha_true_seq(:, pri_idx);
        noise_var = max(real(w_list{pri_idx}' * R_full * w_list{pri_idx}), 1e-12);

        sinr_raw_vals(pri_idx) = real(abs(z_signal)^2 / noise_var);
        meas_seq(pri_idx) = z_signal;

        alpha_pred = gm_rho * alpha_post_prev;
        P_pred = gm_rho^2 * P_post_prev + Q_alpha;
        alpha_pred_seq(:, pri_idx) = alpha_pred;
        H_pred_all{pri_idx} = assemble_channel_from_basis(H_scatter_basis, alpha_pred);

        S_k = c_k' * P_pred * c_k + noise_var;
        K_gain = (P_pred * c_k) / max(real(S_k), 1e-12);

        innov_k = z_signal - c_k' * alpha_pred;
        alpha_post = alpha_pred + K_gain * innov_k;

        P_post = (eye(P) - K_gain * c_k') * P_pred * (eye(P) - K_gain * c_k')' + ...
            K_gain * noise_var * K_gain';
        P_post = (P_post + P_post') / 2;

        kf_out_seq(pri_idx) = c_k' * alpha_post;
        post_var_seq(pri_idx) = max(real(c_k' * P_post * c_k), 1e-12);
        sinr_kf_vals(pri_idx) = real(abs(kf_out_seq(pri_idx))^2 / post_var_seq(pri_idx));

        alpha_post_seq(:, pri_idx) = alpha_post;
        H_post_all{pri_idx} = assemble_channel_from_basis(H_scatter_basis, alpha_post);

        alpha_post_prev = alpha_post;
        P_post_prev = P_post;
    end
end

function H_mat = assemble_channel_from_basis(H_scatter_basis, alpha_vec)

    H_mat = zeros(size(H_scatter_basis{1}));
    for p_idx = 1:numel(H_scatter_basis)
        H_mat = H_mat + alpha_vec(p_idx) * H_scatter_basis{p_idx};
    end
end

function H = build_point_target_channel(alpha, delay_idx, theta, Nt, Nr, N, L, lambda_c, d)

    n_t = (0:Nt-1).';
    n_r = (0:Nr-1).';

    at = exp(1j * 2 * pi / lambda_c * d * n_t * sin(theta));
    ar = exp(1j * 2 * pi / lambda_c * d * n_r * sin(theta));

    H_l = cell(L, 1);
    for ell = 1:L
        H_l{ell} = zeros(Nr, Nt);
    end

    ell = delay_idx + 1;
    H_l{ell} = alpha * (ar * at');

    H = zeros(N * Nr, N * Nt);
    for row = 1:N
        for col = 1:N
            lag_idx = row - col + 1;
            if lag_idx >= 1 && lag_idx <= L
                r_idx = (row - 1) * Nr + (1:Nr);
                c_idx = (col - 1) * Nt + (1:Nt);
                H(r_idx, c_idx) = H_l{lag_idx};
            end
        end
    end
end

function [track_idx, score_acc] = run_tbd_viterbi(score_map, theta_grid, max_jump, sigma_jump, theta_true_first)
    
    num_frames = size(score_map, 1);
    num_grid = size(score_map, 2);
    
    score_acc = zeros(num_frames, num_grid);
    back_ptr = ones(num_frames, num_grid);
    
    score_map_norm = score_map;
    for k = 1:num_frames
        max_val = max(score_map(k, :));
        if max_val > 1e-10
            score_map_norm(k, :) = score_map(k, :) / max_val;
        end
    end
    
    % Initialize first frame
    if nargin > 4 && ~isempty(theta_true_first)
        [~, first_idx] = min(abs(theta_grid - theta_true_first));
        search_range = max(1, first_idx-30):min(num_grid, first_idx+30);
        score_acc(1, :) = -inf;
        score_acc(1, search_range) = score_map_norm(1, search_range);
    else
        score_acc(1, :) = score_map_norm(1, :);
    end
    
    theta_curr = theta_grid(:);
    theta_prev = theta_grid(:).';
    diff_mat = abs(theta_curr - theta_prev);
    
    trans_prob = zeros(num_grid, num_grid);
    for i = 1:num_grid
        for j = 1:num_grid
            diff = diff_mat(i, j);
            if diff <= max_jump
                trans_prob(i, j) = exp(-0.5 * (diff / sigma_jump)^2);
            else
                trans_prob(i, j) = 1e-6;
            end
        end
    end
    
    trans_prob = trans_prob / max(trans_prob(:));
    
    log_trans = log(trans_prob + 1e-12);

    for frame_idx = 2:num_frames
        for grid_idx = 1:num_grid
            
            transition_scores = score_acc(frame_idx - 1, :) + log_trans(grid_idx, :);
            
            [best_prev_score, best_idx] = max(transition_scores);
            
            score_acc(frame_idx, grid_idx) = best_prev_score + score_map(frame_idx, grid_idx);
            
            back_ptr(frame_idx, grid_idx) = best_idx;
        end
    end
    
    track_idx = zeros(num_frames, 1);
    [~, track_idx(end)] = max(score_acc(end, :));
    
    for frame_idx = num_frames-1:-1:1
        track_idx(frame_idx) = back_ptr(frame_idx + 1, track_idx(frame_idx + 1));
    end
end
