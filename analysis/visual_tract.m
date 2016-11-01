% Plot area function and/or pressure function as video

[data, labels, Fs, data_len] = import_datalog('../data/datalog1.log');
sub_sample = 100;
loops = floor(data_len/sub_sample);
pressure = data(:, 90:178);
pressure = pressure-mean(mean(pressure));

area= data(:, 1:89);

% figure;
% for k = 1:20
%     plot(pressure(k, :));
%     hold on
% end
% 
% figure;
% for k = 1:20
%     plot(area(k, :));
%     hold on
% end

%loops = 40;
hold off

F(loops) = struct('cdata',[],'colormap',[]);
v = VideoWriter('vocal_tract.avi');
open(v);

t = 1:64;
t = t-20;
lungs = 6:22;
trachea = 22:35;
glottis = 35:37;
tract = 37:64;
nasal = 64:77;

nasal_vert = max(max(area(:, tract)))+max(max(area(:, nasal)));
nasal_hor = -13;

for j = 1:loops
    plot(t(lungs), 0.025*area(j*sub_sample, lungs), '-r')
    hold on
    plot(t(lungs),-0.025*area(j*sub_sample, lungs), '-r')
    
    plot(t(trachea), 0.5*area(j*sub_sample, trachea), '-b')
    plot(t(trachea), -0.5*area(j*sub_sample, trachea), '-b')
    
    plot(t(glottis), 0.5*area(j*sub_sample, glottis), '-g')
    plot(t(glottis), -0.5*area(j*sub_sample, glottis), '-g')
    
    plot(t(tract), 0.5*area(j*sub_sample, tract), '-b')
    plot(t(tract), -0.5*area(j*sub_sample, tract), '-b')    
    
    plot(t(nasal+nasal_hor), 0.5*area(j*sub_sample, nasal)+nasal_vert, '-b')
    plot(t(nasal+nasal_hor), -0.5*area(j*sub_sample, nasal)+nasal_vert, '-b')
    
    hold off
    drawnow
    F(j) = getframe(gcf);
    writeVideo(v, F(j));
end
close(v);
fig = figure;
movie(fig,F,2)