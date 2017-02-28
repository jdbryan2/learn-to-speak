function [ distance_matrix ] = PairwiseNorm( data_matrix, p )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
    p = 2;
end

Nf = size(data_matrix, 2);


distance_matrix = zeros(Nf,Nf);
% Calculate distances between each set of points
for i = 1:Nf
    for j = i:Nf
        distance_matrix(i,j) = norm( data_matrix(:,i) - data_matrix(:,j), p);
        distance_matrix(j,i) = distance_matrix(i,j);
    end
    fprintf('i = %d out of %d\n',i,Nf);
end


end

