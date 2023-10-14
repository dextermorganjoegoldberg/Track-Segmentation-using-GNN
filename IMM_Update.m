function [X_k_k, P_k_k, miu_k_k, X_k_ks, P_k_ks] = IMM_Update(X_k_k_1s, P_k_k_1s, miu_k_k_1, z_k, H, R)
    n = size(X_k_k_1s, 1);
    m = size(X_k_k_1s, 2);
    likelihoods = zeros(1, m);
    X_k_ks = zeros(n, m); 
    P_k_ks = cell(1, m); 
    for im=1:m
        X_k_k_1 = X_k_k_1s(:, im);
        P_k_k_1 = P_k_k_1s{im};
        S = H*P_k_k_1*H' + R;
        innovation = z_k - H*X_k_k_1;
        likelihoods(im) = exp(-0.5*innovation'*inv(S)*innovation) / sqrt( det(2*pi*S) );
        K = P_k_k_1*H'*inv(S);
        X_k_ks(:, im) = X_k_k_1 + K*innovation;
        P_k_ks{im} = (eye(n)-K*H)*P_k_k_1;
    end
    
    miu_k_k = miu_k_k_1.*likelihoods;
    if sum(miu_k_k) == 0
        miu_k_k = miu_k_k_1;
        fprintf('likelihoods is zeros.');
    else
        miu_k_k = miu_k_k / sum(miu_k_k);
    end
    
    X_k_k = zeros(n,1);
    P_k_k = zeros(n, n);
    for im=1:m
        X_k_k = X_k_k + miu_k_k(im)*X_k_ks(:, im);
        P_k_k = P_k_k + miu_k_k(im)*P_k_ks{im};
    end
    for im=1:m
        P_k_k = P_k_k + miu_k_k(im)*(X_k_ks(:, im)-X_k_k)*(X_k_ks(:, im)-X_k_k)';
    end
end