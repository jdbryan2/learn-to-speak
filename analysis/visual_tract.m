% Plot area function and/or pressure function as video
clear
%[data1, labels, Fs, data_len_ref] = import_datalog('testThesis4/artword_logs/apa1.log');
%[data2, labels, Fs, data_len] = import_datalog('testBatch3/prim_logs/primlog1.log');
[data2, labels, Fs, data_len] = import_datalog('testNew1/artword_logs/ohh1.log');
[Snd,fs,duration] = import_sound('testNew1/artword_logs/ohh_sound1.log',true);

sub_sample = 1;
loops = floor(data_len/sub_sample);
%loops_ref = floor(data_len_ref/sub_sample);
%pressure = data(:, 90:178);
%pressure = pressure-mean(mean(pressure));

%area1= data1(:, 1:89);
area2= data2(:, 1:89);

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

figure(12);
for j = 1:loops
    plot(0:21,0.025*area2(j*sub_sample, 1:22), '.r')
    hold on
    plot(0:21,-0.025*area2(j*sub_sample, 1:22), '.r')
    plot(22:88,0.5*area2(j*sub_sample, 23:89), '.b')
    plot(22:88,-0.5*area2(j*sub_sample, 23:89), '.b')
    ind = j;
    
    % Plot initial sample as a reference
    plot(0:21,0.025*area2(sub_sample, 1:22), '-r')
    plot(0:21,-0.025*area2(sub_sample, 1:22), '-r')
    plot(22:88,0.5*area2(sub_sample, 23:89), '-b')
    plot(22:88,-0.5*area2(sub_sample, 23:89), '-b')
%     if j> loops_ref
%         ind = loops_ref;
%     end
%     plot(0:21, 0.025*area1(ind*sub_sample, 1:22), '-r')
%     plot(0:21,-0.025*area1(ind*sub_sample, 1:22), '-r')
%     plot(22:88,0.5*area1(ind*sub_sample, 23:89), '-b')
%     plot(22:88,-0.5*area1(ind*sub_sample, 23:89), '-b')
    hold off
    drawnow
    F(j) = getframe(gcf);
end

fig = gcf;
movie(fig,F,2,10)