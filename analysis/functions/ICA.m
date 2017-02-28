function [ output, W ] = ICA( data, mu, epsilon, max_iter )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if nargin < 4
    max_iter = 500;
end

if nargin < 3
    epsilon = 0.001;
end

%epsilon = 0.5;
[M, N] = size(data)
y = data;

delta = eye(M)-(y.^3)*y'/N;

% update rule
% W = W + mu*delta*W
W = eye(M);
iter = 0;
while sum(sum(abs(delta)))>epsilon

    W = W+mu*delta*W;
    y = W*data;
    delta = eye(M)-(y.^3)*y'/N;

    iter=iter+1;
    if iter > max_iter
        break
    end
end
%iter
output = W*data;
