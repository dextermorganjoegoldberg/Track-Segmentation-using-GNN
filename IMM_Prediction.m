function [X_k_k_1, P_k_k_1, miu_k_k_1, X_k_k_1s, P_k_k_1s] = IMM_Prediction(miu_k_k, X_k_ks, P_k_ks, F, G, Q, T)
    % miu_k_k: probability of each model at k-1. [1*m].
    % X_k_ks: state of each model at k-1. [n*m], n is dim of state, m is
    % the number of models.
    % P_k_ks: state of each model at k-1. cell(1, m).
    Models = [0, 0;
              0.1, -0.1;
              -0.1, 0.1]'; % acceleration of each model. [2*m].
    Model_Num = size(Models, 2); % model num.
    if abs(T) > 1e-5
        PI = [0.9, 0.05, 0.05;
              0.05, 0.9, 0.05;
              0.05, 0.05, 0.9]; % markov matrix. [Pij], i->j.
    else
        PI = eye(3);
    end
      
    n = size(X_k_ks, 1); % state dim.
    m = size(X_k_ks, 2); % model num.
    if m ~= Model_Num % error.
        return;
    end
    
    % reinitialization.
    miu_k_k_1 = miu_k_k*PI; % 未归一化，c_k_1.
    temp = repmat(miu_k_k', 1, m).*PI;
    temp = temp ./ repmat(miu_k_k_1, m,1); % i-j的条件概率。
    X_k_ks_re = zeros(size(X_k_ks));
    P_k_ks_re = cell(size(P_k_ks));
    for im=1:m
        weight = temp(:,im);
        X_k_ks_re(:,im) = sum(repmat(weight',n,1).*X_k_ks, 2);
        P_k_ks_re{im} = zeros(n, n);
        for imm=1:m
            P_k_ks_re{im} = P_k_ks_re{im} + weight(imm)* ...
                (P_k_ks{imm} + (X_k_ks(:,imm)-X_k_ks_re(:,im))*(X_k_ks(:,imm)-X_k_ks_re(:,im))');
        end
    end
    
    % prediction.
    miu_k_k_1 = miu_k_k_1 / sum(miu_k_k_1);
    X_k_k_1s = zeros(n, m); X_k_k_1 = zeros(n,1);
    P_k_k_1s = cell(1, m); P_k_k_1 = zeros(n, n);
    for im=1:Model_Num
        X_k_k_1s(:,im) = F*X_k_ks_re(:,im) + G*Models(:,im);
        P_k_k_1s{im} = F*P_k_ks_re{im}*F' + G*Q*G';
        
        X_k_k_1 = X_k_k_1 + miu_k_k_1(im)*X_k_k_1s(:,im);
        P_k_k_1 = P_k_k_1 + miu_k_k_1(im)*P_k_k_1s{im};
    end
    for im=1:Model_Num
        P_k_k_1 = P_k_k_1 + miu_k_k_1(im)*(X_k_k_1s(:,im)-X_k_k_1)*(X_k_k_1s(:,im)-X_k_k_1)';
    end
end