function [ distance_matrix ] = PairwiseLPC( data_matrix, p )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

Nf = size(data_matrix, 2);


distance_matrix = zeros(Nf,Nf);
% Calculate distances between each set of points
for i = 1:Nf
    for j = i:Nf
        x = data_matrix(:,i);
        y = data_matrix(:,j);
        distance_matrix(i,j) = symmetricLPC( x, y, p);
        distance_matrix(j,i) = distance_matrix(i,j);
    end
    fprintf('i = %d out of %d\n',i,Nf);
end

end

