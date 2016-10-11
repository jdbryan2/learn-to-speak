%% Generate an ISOMAP from the audio data with LPC
% Break audio into fragments corresponding to each frame
% This also removes the last frame, and the last fragement of audio
% [y_mtrx,v,Nf] = func_CreateAudioMatrix(y,v,Nf,Sr);

% script level variables
window_size = 20; % ms
step_size = 5; % ms 
p = 25;


% load TIMIT file(s)
fname = 'sa1';
[y,Fs] = audioread(['timit/', fname, '.wav']);

% convert window size and overlap to samples
window_size = window_size*Fs/1000;
step_size = step_size*Fs/1000;

data = WindowReshape(y, hamming(window_size), step_size);
distance_mat = PairwiseLPC(data, p);

K = 20; % K nearest neighbors
if true %generate_ISOMAP_LPC
    X = ISOMAP(distance_mat,K);
    if true %save_ISOMAP_LPC
        save(['LPCISOMAP_',num2str(K),'_', fname, '.mat'],'X','K')
    end
elseif false %load_ISOMAP_LPC
    load('X_IsomapLPC_3trials.mat')
end

% Scaling factor
sc = 0.01*2;
scatter(X(1, :), X(2,:), 'LineWidth', 2)
%for i = 1:40:Nf-1
 %   % Store xy location of the point that we will plot the image onto
 %   x = X(1,i);
 %   y = X(2,i);
 %   hold on
 %   colormap gray
    % Plot the image, scaling it appropriately
    %imagesc([x-sc x+sc],[y+sc y-sc],v(:,:,i))
    %imagesc([x-sc x+sc],[y+sc y-sc],spec_frames(:,:,:,i))
    %axis equal
%end
xlabel('Component 1','FontSize',16)
ylabel('Component 2','FontSize',16)
title('ISOMAP of the LPC of audio data','Fontsize',18)

% Plot several graphs

