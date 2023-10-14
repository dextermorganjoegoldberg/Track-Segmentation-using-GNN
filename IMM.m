clc; clear all; close all;
%% load data from csv file.
Target_Num = 5; % number of targets.
Before_Tra = cell(Target_Num, 1); % trajectory before discontinuity.
Between_Tra = cell(Target_Num, 1); % trajectory between discontinuity.
After_Tra = cell(Target_Num, 1); % trajectory after discontinuity.
for it=1:Target_Num
    path_name = sprintf('./data/%d/', it);
    file_name = 'h.csv';
    Before_Tra{it} = importdata([path_name, file_name]);
    Before_Tra{it} = Before_Tra{it}.data';
    Before_Len = size(Before_Tra{it}, 2);
    file_name = 'b.csv';
    Between_Tra{it} = importdata([path_name, file_name]);
    Between_Tra{it} = Between_Tra{it}.data';
    Between_Len = size(Between_Tra{it}, 2);
    file_name = 't.csv';
    After_Tra{it} = importdata([path_name, file_name]);
    After_Tra{it} = After_Tra{it}.data';
    After_Len = size(After_Tra{it}, 2);
end

%% set up parameters.
T = 1; % time interval.
F = [1,T,0,0;
      0,1,0,0;
      0,0,1,T;
      0,0,0,1];
G =[T^2/2,    0;
    T,      0;
    0,      T^2/2;
    0,      T] ;
Q = [0.1^2 0;
    0 0.1^2];

%% IMM using Before_Tra.
Predict_using_Before = cell(Target_Num, 1);
for it=1:Target_Num 
    % use IMM to initialize model probability.
    for k=1:Before_Len
        if k==1 % initialization.
            model_num = 3;
            miu_k_k = ones(1, model_num) / model_num;
            X_k_ks = repmat(Before_Tra{it}(:,1), 1, model_num);
            P_k_ks = cell(1, model_num);
            for im=1:model_num
                P_k_ks{im} = eye(4)*0.1^2;
            end
        else
            % prediction.
            [X_k_k_1, P_k_k_1, miu_k_k_1, X_k_k_1s, P_k_k_1s] = IMM_Prediction(miu_k_k, X_k_ks, P_k_ks, F, G, Q, T);
            % update.
            z_k = Before_Tra{it}(:,k);
            H = eye(4);
            R = eye(4)*0.05^2;
            [X_k_k, P_k_k, miu_k_k, X_k_ks, P_k_ks] = IMM_Update(X_k_k_1s, P_k_k_1s, miu_k_k_1, z_k, H, R);
        end
    end
    % use IMM to predict trajectory.
    Predict_using_Before{it} = zeros(4, Between_Len);
    for k=1:Between_Len
        [X_k_k, P_k_k, miu_k_k, X_k_ks, P_k_ks] = IMM_Prediction(miu_k_k, X_k_ks, P_k_ks, F, G, Q, T);
        Predict_using_Before{it}(:, k) = X_k_k;
    end
end

%% IMM using After_Tra.
Predict_using_After = cell(Target_Num, 1);
for it=1:Target_Num 
    % use IMM to initialize model probability.
    for k=1:After_Len
        if k==1 % initialization.
            model_num = 3;
            miu_k_k = ones(1, model_num) / model_num;
            x_ = After_Tra{it}(:,end);
            x_(2) = -x_(2);
            x_(4) = -x_(4);
            X_k_ks = repmat(x_, 1, model_num);
            P_k_ks = cell(1, model_num);
            for im=1:model_num
                P_k_ks{im} = eye(4)*0.1^2;
            end
        else
            % prediction.
            [X_k_k_1, P_k_k_1, miu_k_k_1, X_k_k_1s, P_k_k_1s] = IMM_Prediction(miu_k_k, X_k_ks, P_k_ks, F, G, Q, T);
            % update.
            z_k = After_Tra{it}(:,After_Len+1-k);
            z_k(2) = -z_k(2);
            z_k(4) = -z_k(4);
            H = eye(4);
            R = eye(4)*0.05^2;
            [X_k_k, P_k_k, miu_k_k, X_k_ks, P_k_ks] = IMM_Update(X_k_k_1s, P_k_k_1s, miu_k_k_1, z_k, H, R);
        end
    end
    % use IMM to predict trajectory.
    Predict_using_After{it} = zeros(4, Between_Len);
    for k=1:Between_Len
        [X_k_k, P_k_k, miu_k_k, X_k_ks, P_k_ks] = IMM_Prediction(miu_k_k, X_k_ks, P_k_ks, F, G, Q, T);
        Predict_using_After{it}(:, Between_Len+1-k) = X_k_k;
    end
end

%% calculate prediction error.
Before_error = zeros(Target_Num, 1);
After_error = zeros(Target_Num, 1);
for it=1:Target_Num
    Before_error(it) = mean(sqrt(sum((Predict_using_Before{it}([1,3],:)-Between_Tra{it}([1,3],:)).^2)));
    After_error(it) = mean(sqrt(sum((Predict_using_After{it}([1,3],:)-Between_Tra{it}([1,3],:)).^2)));
end

%% plot.
figure(1);
plot(Before_error, 'r-*'); hold on
plot(After_error, 'b-o'); hold on
legend('前向', '后向');
xlabel('Trajectory Index');
ylabel('error/m');
title('Prediction Error');

figure(2);
for it=1:Target_Num
    plot(Before_Tra{it}(1,:), Before_Tra{it}(3,:), 'k', 'LineWidth', 1); hold on
    plot(After_Tra{it}(1,:), After_Tra{it}(3,:), 'k', 'LineWidth', 1); hold on
   % p1 = plot([Before_Tra{it}(1,end), Between_Tra{it}(1,:), After_Tra{it}(1,1)], ...
      %  [Before_Tra{it}(3,end), Between_Tra{it}(3,:), After_Tra{it}(3,1)], 'g:', 'LineWidth', 1); hold on
    p2 = plot([Before_Tra{it}(1,end), Predict_using_Before{it}(1,:), After_Tra{it}(1,1)], ...
        [Before_Tra{it}(3,end), Predict_using_Before{it}(3,:), After_Tra{it}(3,1)], 'r--', 'LineWidth', 1); hold on
    p3 = plot([Before_Tra{it}(1,end), Predict_using_After{it}(1,:), After_Tra{it}(1,1)], ...
        [Before_Tra{it}(3,end), Predict_using_After{it}(3,:), After_Tra{it}(3,1)], 'b-.', 'LineWidth', 1); hold on
end
xlabel('X / m');
ylabel('Y / m');
legend([ p2, p3], { '前向预测轨迹', '后向预测轨迹'});
title('预测轨迹');