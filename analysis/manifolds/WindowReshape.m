function [ data_matrix ] = WindowReshape( data_vector, window, step_size )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

len = length(window);
overlap = len-step_size;

data_matrix = buffer(data_vector, len, overlap);

segments = size(data_matrix, 2);
if segments < 500
    
    window_mat = repmat(window, 1, segments);
    data_matrix = data_matrix.*window_mat;
    
else
    
    for k =1:segments
        data_matrix(:, k) = window.*data_matrix(:, k);
    end
    
end

end

