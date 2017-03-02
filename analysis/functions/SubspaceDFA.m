function [ K, O, U, S, V ] = SubspaceDFA( Xf, Xp, k )
%SubspaceDFA subspace method of dynamic factor analysis
%   Xf: Future values of data vectors
%   Xp: Past value of data vectors
%   k: The number of factors that will be extracted

    prms = 1:k;
    %F = Xf*(Xp'*(Xp*Xp')^-1);
    F = Xf*pinv(Xp);
    Qp_ = cov(Xp')';
    Qf_ = cov(Xf')';
    Qp_ = eye(size(Qp_));
    Qf_ = eye(size(Qf_));
    % Take real part of scale factor
    F_sc = real(Qf_^(-.5))*F*real(Qp_^(.5));
    [U,S,V] = svd(F_sc);
    Sk = S(prms,prms);
    Vk = V(:,prms);
    Uk = U(:,prms);
    K = Sk^(1/2)*Vk'*real(Qp_^(-.5));
    O = real(Qf_^(.5))*Uk*Sk^(1/2);

end

