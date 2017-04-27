% Plot area function and/or pressure function as video
clear
[data1, labels, Fs, data_len_ref] = import_datalog('testBatch1000/tubart-default8/prim_logs/primlog8.log');
[data2, labels, Fs, data_len] = import_datalog('testBatch1000/tubart-default8/prim_logs/primlog8.log');
%[data1, labels, Fs, data_len_ref] = import_datalog('testBatch1000/artword_logs/apa1.log');
%[data2, labels, Fs, data_len] = import_datalog('testBatch1000/artword_logs/apa1.log');
%[data2, labels, Fs, data_len] = import_datalog('testNew1/artword_logs/ohh1.log');
%[Snd,fs,duration] = import_sound('testNew1/artword_logs/ohh_sound1.log',true);
%[data1, labels, Fs, data_len_ref] = import_datalog('testRandArt1/prim_logs/primlog1.log');
%[Snd,fs,duration] = import_sound('testRandArt1/prim_logs/sound8.log',true);
%[data2, labels, Fs, data_len] = import_datalog('testRandArt1/prim_logs/primlog8.log');
%[Snd,fs,duration] = import_sound('testRandArt1/prim_logs/sound8.log',true);

sub_sample = 1;
loops = floor(data_len/sub_sample);
loops_ref = floor(data_len_ref/sub_sample);
%pressure = data(:, 90:178);
%pressure = pressure-mean(mean(pressure));

area1= data1(:, 1:89);
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

max1 = max(max(data1(:,1:22)));
max2 = max(max(data2(:,1:22)));
max12 = max([max1,max2]);

f12=figure(12);
ylim([-0.025*max12*1.2,0.025*max12*1.2])
for j = 1:loops
    plot(0:21,0.025*area2(j*sub_sample, 1:22), '.r')
    hold on
    plot(0:21,-0.025*area2(j*sub_sample, 1:22), '.r')
    plot(22:88,0.5*area2(j*sub_sample, 23:89), '.b')
    plot(22:88,-0.5*area2(j*sub_sample, 23:89), '.b')
    ind = j;
    
    % Plot initial sample as a reference
%     plot(0:21,0.025*area2(sub_sample, 1:22), '-r')
%     plot(0:21,-0.025*area2(sub_sample, 1:22), '-r')
%     plot(22:88,0.5*area2(sub_sample, 23:89), '-b')
%     plot(22:88,-0.5*area2(sub_sample, 23:89), '-b')
    if j> loops_ref
        ind = loops_ref;
    end
    plot(0:21, 0.025*area1(ind*sub_sample, 1:22), '-r')
    plot(0:21,-0.025*area1(ind*sub_sample, 1:22), '-r')
    plot(22:88,0.5*area1(ind*sub_sample, 23:89), '-b')
    plot(22:88,-0.5*area1(ind*sub_sample, 23:89), '-b')
    hold off
    ylim([-0.025*max12*1.2,0.025*max12*1.2])
    drawnow
    F(j) = getframe(f12);
end

movie(f12,F,2,5)