function [ W, H] = NMF( X , R, epsilon, max_iter)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if nargin < 4
    max_iter = 500;
end

if nargin < 3
    epsilon = 0.1;
end

[M, N] = size(X);

% X -> M, N
% W -> M, R
% H -> R, N

W_old = zeros(M, R);
H_old = zeros(R, N);
W = rand(M, R);
H = rand(R, N);


iter = 0;
while sum(sum(abs(X-W*H)))/(N*M)> epsilon
    
    % update W
    W_old = W;
    H_old = H;
    
    %X_ = X.*((W*H+10^(-10)).^(-1));

    W = W.*((X./(W*H+10^(-10)))*H')./(ones(M, N)*H'+10^(-10));
    H = H.*(W'*(X./(W*H+10^(-10))))./(W'*ones(M, N)+10^(-10));
    
    
    iter = iter+1;
    if iter > max_iter
        %max_iter
        %sum(sum(abs(X-W*H)))/(N*M)
        break;
    end

end


