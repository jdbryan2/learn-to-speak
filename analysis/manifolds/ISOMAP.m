function X = ISOMAP(dist_mat,K)
%% Generate Isomap from video data
% Code based off of:
% CS 598PS
% Problem Set 2
% Lecture 7
% Slide 68
% Jacob Bryan

% Variables:
% dist_mat: matrix of distances between each data point
% K: number of nearest neighbors to begin with
Nf = size(dist_mat, 2);

% construct distance matrix
D = inf*ones(Nf, Nf);

% connect only K nearest neighbors
for n = 1:Nf
    %d = spec_tens - repmat(spec_tens(:, n), 1, Nf);
    %d = d.^2;
    %d = sum(d, 1);
    d = dist_mat(:,n);
    [dist, ind] = sort(d);
    D(n, ind(1:K)) = dist(1:K);
    D(ind(1:K), n) = dist(1:K);
end

% find geodesics
for k = 1:Nf
    d1 = repmat(D(:,k), 1, Nf);
    d2 = repmat(D(k,:), Nf, 1);
    
    D = min(D, d1+d2);
end


S= -0.5*(D - D*ones(Nf)/Nf - ones(Nf)*D/Nf+ones(Nf)*D*ones(Nf)/(Nf*Nf));
[vec, val] = eigs(S, 3);
X = val^(0.5)*vec';

end