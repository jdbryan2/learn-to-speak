%% Concatenate Local Manifold Audio
% After running one of the ISOMAPS, this will select points that are in a
% certain region near each other, and it will concatenate the audio, so we
% can hear what the general sound is like in that region

% Choose samples in same region
x_data = X(1,:);
xrng = 0.01*[-10 35];
% xrng = [-630 -550];
y_data = X(2,:);
yrng = 0.01*[-35 -8];
% yrng = [-20 100];

% Indices to keep
x_idx = (x_data > xrng(1)).*(x_data < xrng(2));
y_idx = (y_data > yrng(1)).*(y_data < yrng(2));
idx = x_idx.*y_idx;

% Concatenate sound
y_local = reshape(y_mtrx(:,logical(idx)),length(y_mtrx(:,1))*sum(idx),1);
sound(y_local,Sr);

% audiowrite('DarkGreen_silence.mp4',y_local,44100);