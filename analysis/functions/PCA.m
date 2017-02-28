function [ output, W, vec, val ] = PCA( data, dim )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

[N, ~] = size(data);

% Eigen Decomp
[vec, val] = eigs(data*data', dim);
val = (val/N)^(1/2);

% PCA the data
W = val^(-1)*vec';
output = W*data;


end

