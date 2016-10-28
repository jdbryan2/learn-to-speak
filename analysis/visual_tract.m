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

for j = 1:loops
    plot(-21:0, 0.025*area(j*sub_sample, 1:22), '-r')
    hold on
    plot(-21:0,-0.025*area(j*sub_sample, 1:22), '-r')
    plot(0.5*area(j*sub_sample, 23:89), '-b')
    plot(-0.5*area(j*sub_sample, 23:89), '-b')
    hold off
    drawnow
    F(j) = getframe(gcf);
end

fig = figure;
movie(fig,F,2)