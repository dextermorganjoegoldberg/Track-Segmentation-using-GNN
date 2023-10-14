clc; clear all; close all;
%% set up parameters.
Target_Num=10;
px=100*rand(1, Target_Num);
py=100*rand(1, Target_Num);
vx=5*(rand(1, Target_Num)+1)*2;
vy=5*(rand(1, Target_Num)+0.5)*2;
X0 = [px;vx;py;vy];
t_window = int32(30+10*rand()); % lasting time corresponding to each mode.
mode_change_time = 3;  % the number of mode changed.
flags = zeros(Target_Num, mode_change_time);
flags(rand(Target_Num, mode_change_time)>0.5) = 1; % flag of mode. CV-0, CT-1.

%% params.
CV_sigma = 0.9;
CT_sigma = 1.1;
CT_w_max = 10.0/180.0*pi/double(t_window*1.0); % maximum turning rate.
Discont_Len = 15;


%% initialize discontinues trahectory.
Trajectory_Discont = cell(3, Target_Num);
for i=1:Target_Num
    Trajectory_Discont{1, i} = zeros(4, 10); % before 10s.*
    Trajectory_Discont{2, i} = zeros(4, 10); % after 10s.
    Trajectory_Discont{3, i} = zeros(4, Discont_Len); % between 5s.
end
Log_Len = 10;
Discont_Begin_Times = zeros(1, Target_Num);
Discont_Range = 600;

%% produce tragectory.
T = 1;
Trajectory = cell(1, Target_Num);
for i=1:Target_Num
    Trajectory{i} = zeros(4, t_window*mode_change_time);
    Trajectory{i}(:,1) = X0(:, i);
end
for i=1:Target_Num
	for j=1:mode_change_time
        flag = flags(i, j);
        if flag == 0 % CV
            A = [1,T,0,0;
                  0,1,0,0;
                  0,0,1,T;
                  0,0,0,1];
            G=[T^2/2,    0;
                T,      0;
                0,      T^2/2;
                0,      T] ;
            Q=[CV_sigma^2 0;
                0 CV_sigma^2];
        else % CT
            ct_w = (rand()-0.5)*2.0*CT_w_max;
%             ct_w = -rand()*CT_w_max;
            A=CreatCTF(ct_w, T);
            G=CreatCTT(T);
            Q=[CT_sigma^2 0;
                0, CT_sigma^2];
        end
        for t=1+(j-1)*t_window : j*t_window
            if t == t_window*mode_change_time
                continue
            end
            Trajectory{i}(:, t+1) = A*Trajectory{i}(:, t) + G*sqrt(Q)*[randn,randn]';
            % get discontinues timestamp.
            if Trajectory{i}(1, t+1) > Discont_Range && Trajectory{i}(1, t) < Discont_Range
                Discont_Begin_Times(1, i) = t+1 + round((rand()-0.5)*2*2);
            end
        end
	end
end
Discont_Over_Times = Discont_Begin_Times + Discont_Len - 1;

%% get discontinuous trajectory.
Trajectory_show = cell(2, Target_Num);
for i=1:Target_Num
    Trajectory_show{i} = [];
end
for i=1:Target_Num
    for t=1:j*t_window
        % get discontinues trajectory.
        discont_begin_time = Discont_Begin_Times(1, i);
        discont_over_time = Discont_Over_Times(1, i);
        delta_begin_t = discont_begin_time - t;
        delta_over_t = discont_over_time - t;
        if delta_begin_t <= Log_Len && delta_begin_t>0 % before discontinues.
            dt = 10-delta_begin_t+1;
            Trajectory_Discont{1, i}(:, dt) = Trajectory{i}(:, t);
        elseif delta_over_t >= -Log_Len && delta_over_t<0 % after discontinues.
            dt = -delta_over_t;
            Trajectory_Discont{2, i}(:, dt) = Trajectory{i}(:, t);
        elseif (-delta_begin_t) <= Discont_Len-1 && (-delta_begin_t)>=0
            dt = -delta_begin_t+1;
            Trajectory_Discont{3, i}(:, dt) = Trajectory{i}(:, t);
        end
        
        if t < discont_begin_time
            Trajectory_show{1, i} = [Trajectory_show{1, i}, Trajectory{i}(:, t)];
        elseif t > discont_over_time
            Trajectory_show{2, i} = [Trajectory_show{2, i}, Trajectory{i}(:, t)];
        end
    end
end

%% plot Trajectory.
figure(1);
for i=1:Target_Num
    plot(Trajectory{i}(1,:), Trajectory{i}(3,:), 'k', 'LineWidth', 1); hold on
end
xlabel('X / m');
ylabel('Y / m');
title(sprintf('未中断轨迹, CV-sigma: %.4f, CT-sigma: %.4f', CV_sigma, CT_sigma));

%% plot Trajectory_Discont.
figure(2);
for i=1:Target_Num
    plot(Trajectory_show{1, i}(1,:), Trajectory_show{1, i}(3,:), 'r', 'LineWidth', 1); hold on
    plot(Trajectory_show{2, i}(1,:), Trajectory_show{2, i}(3,:), 'b', 'LineWidth', 1); hold on
end
xlabel('X / m');
ylabel('Y / m');
title(sprintf('中断轨迹, CV-sigma: %.4f, CT-sigma: %.4f', CV_sigma, CT_sigma));
%% save to csv.
for it=1:Target_Num
    path_name = sprintf('./data/%d/', it);
    mkdir(path_name);
    for j=1:3
        Data_Len = size(Trajectory_Discont{j, it},2);
        Csv_Data = cell(Data_Len+1, 4*Target_Num);
        Csv_Data{1, 1} = sprintf('x');
        Csv_Data{1, 2} = sprintf('vx');
        Csv_Data{1, 3} = sprintf('y');
        Csv_Data{1, 4} = sprintf('vy');
        for k=1:Data_Len
             Csv_Data{k+1, 1} = Trajectory_Discont{j, it}(1,k);
             Csv_Data{k+1, 2} = Trajectory_Discont{j, it}(2,k);
             Csv_Data{k+1, 3} = Trajectory_Discont{j, it}(3,k);
             Csv_Data{k+1, 4} = Trajectory_Discont{j, it}(4,k);
        end
        if j==1
            file_name = 'h.csv';
        elseif j==2
            file_name = 't.csv';
        else
            file_name = 'b.csv';
        end
        writecell(Csv_Data, [path_name, file_name]);
    end
end
