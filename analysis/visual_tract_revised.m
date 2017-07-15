clear;
close all;

% For plotting IPA figures
trial_ids = [132,134,140,142,301,304,305,316];
testdir = 'testStim3Batch300/artword_logs/';
filename_fmt = 'ipa%i_ex1.log';
logtype = 'IPA %i';
snd_fmt = 'ipa%i_ex_sound1.log';
take_snapshot = true;

% For plotting Primlogs
% trial_ids = [0,1,2,3,4,5,6,7,8];
% %testname = 'testStim3Batch300';
% testname = 'testStim1Batch50';
% data_type = 'tubart';
% config = 'original_50noisemaker';
% k = 8;
% testdir = [testname,'/',data_type,'-',config,num2str(k),'/prim_logs/'];
% filename_fmt = 'primlog%i.log';
% snd_fmt = 'sound%i.log';
% logtype = 'Primitive %i Controller';
%take_snapshot = false;

psize = [8,5];
fsize = psize./[8.5,11];
xsz = fsize(1)*0.4221;
ysz = fsize(2)*0.8923;
%xsz=1;ysz=1;

for trial = 1:length(trial_ids)
    % Plot area function and/or pressure function as video
    fname = sprintf([testdir,filename_fmt], trial_ids(trial));
    [data, labels, Fs, data_len] = import_datalog(fname);
    sndname = sprintf([testdir,snd_fmt], trial_ids(trial));
    [Snd,fs,duration] = import_sound(sndname,true);
    %data(1, :) = []; % remove first row to line up with sound
    %data_len = data_len-1;
    %sub_sample = fs/Fs;
    sub_sample = 1;
    loops = floor(data_len*sub_sample);
    %%%pressure = data(:, 90:178);

    % take only the tubes that were used and rescale to atm
    %pressure = pressure(:, 6:77); %pressure-mean(mean(pressure));
    %pressure = pressure/142054.26;
    %%%pressure= pressure-mean(mean(pressure(:, 7:16),1));
    %%%pressure = pressure/max(max(abs(pressure(:, 7:77))));
    %%%pressure = pressure*1e-3;

    area= data(:, 1:89)*1e4; %to convert from meters^2 to cm^2
    
    audiowrite([fname(1:end-4),'.wav'], Snd, fs, 'BitsPerSample', 64)

    samp_rate = Fs*sub_sample;
    v = vision.VideoFileWriter([fname(1:end-4),'.avi'],'FrameRate',samp_rate);%,'AudioInputPort',true);
    %v = VideoWriter([fname,'.avi'],'Uncompressed AVI'); %#ok<TNMLP>
    %v.FrameRate = Fs/sub_sample;
    %open(v);

    t = 0:63+1;
    %t = t-1;
    lungs = (6:22)+1;
    bronchi = (23:28)+1;
    lower_resp = [lungs,bronchi];
    l_scale = 0.05;
    trachea = (29:34)+1;
    glottis = (35:36)+1;
    g_scale = 5;
    upper_tract = (37:63)+1;
    tract = [trachea,glottis,upper_tract];
    t_scale = 1;
    nasal = (64:77)+1;
    n_scale = 1;
    
    lmax = max(max(area(:,lower_resp)));

    glott_vert = -5.7;%(max(max(area(:,upper_tract)))+30*max(max(area(:,glottis))))/1;
    
    nasal_vert = (max(max(area(:, upper_tract)))+max(max(area(:, nasal))))/1.9;
    nasal_hor = -13-1; % makes first nasal tube match up with tube 50 which is where they diverge
    
    % full screen figure
    f1 = figure(1);
    f1.Units = 'normalized';
    f1.Position = [0, 0,xsz,ysz];
    
    for j = 1:loops
        vt_ind = floor((j/samp_rate)*(Fs));
        tb = annotation('textbox',[0.75-.01,0.2,0.1,0.1],'String',['Scaling:' char(10),'Blue - 1:1' char(10),'Red - 1:20' char(10),'Green - 5:1'],'FitBoxToText','on');
        %tb.Color = 'magenta';
        tb.FontSize = 12;
        
        plot(t(6:end),zeros(1,length(t)-5),'k--','LineWidth',1.5);
        title(sprintf(['Vocal Tract Area Function: ',logtype], trial_ids(trial)),'FontSize',12);
        ylabel('Area cm^2')
        xlabel('Tube Section # (Does not correspond exactly to tube length)');
        %ylim([-1.2*lmax*l_scale/2,1.2*lmax*l_scale/2]);
        set(gca,'fontsize',12)
        %set(gca,'XTickLabel','')
        set(gca,'YLim',[-11.5,11.5]);
        hold on
        l_area = [area(vt_ind, lower_resp),area(vt_ind, lower_resp(end))];
        l_t = [t(lower_resp),t(lower_resp(end)+1)];
        stairs(l_t, 0.5*l_scale*l_area, '-r','LineWidth',1.5)
        stairs(l_t,-0.5*l_scale*l_area, '-r','LineWidth',1.5)
        la = annotation('textbox',[0.23-.01,0.42,0.1,0.05],'String','Lungs','FitBoxToText','on');
        la.Color = 'red';
        la.FontSize = 12;
        la.LineStyle = 'none';
        
        ba = annotation('textbox',[0.37-.01,0.43,0.1,0.05],'String','Bronchi','FitBoxToText','on');
        ba.Color = 'red';
        ba.FontSize = 12;
        ba.LineStyle = 'none';
        %%%plot(t(lungs)+1, pressure(vt_ind, lungs+1), '.c')

        t_a = [area(vt_ind, tract),area(vt_ind, tract(end))];
        t_t = [t(tract),t(tract(end)+1)];
        stairs(t_t, 0.5*t_scale*t_a, '-b','LineWidth',1.5)
        stairs(t_t, -0.5*t_scale*t_a, '-b','LineWidth',1.5)
        
        ta = annotation('textbox',[0.45-.01,0.58,0.1,0.05],'String','Trachea','FitBoxToText','on');
        ta.Color = 'Blue';
        ta.FontSize = 12;
        ta.LineStyle = 'none';
        
        pa = annotation('textbox',[.56-.01,0.39,0.1,0.05],'String','Pharynx','FitBoxToText','on');
        pa.Color = 'Blue';
        pa.FontSize = 12;
        pa.LineStyle = 'none';
        
        oca = annotation('textbox',[.67-.01,0.39,0.1,0.05],'String','Oral Cavity','FitBoxToText','on');
        oca.Color = 'Blue';
        oca.FontSize = 12;
        oca.LineStyle = 'none';
        %%%plot(t(trachea), pressure(vt_ind, trachea), '.c')

        g_area = [area(vt_ind, glottis),area(vt_ind, glottis(end))];
        g_t = [t(glottis),t(glottis(end)+1)];
        %stairs(g_t, 0.5*t_scale*g_area, '--g','LineWidth',1.5)
        %stairs(g_t, -0.5*t_scale*g_area, '--g','LineWidth',1.5)
        plot(g_t,zeros(size(g_t))+glott_vert,'k:');
        stairs(g_t, 0.5*g_scale*g_area+glott_vert, '-g','LineWidth',1.5)
        stairs(g_t, -0.5*g_scale*g_area+glott_vert, '-g','LineWidth',1.5)
        
        ga = annotation('textbox',[0.50,0.16,0.1,0.05],'String','Glottis','FitBoxToText','on');
        ga.Color = 'green';
        ga.FontSize = 12;
        ga.LineStyle = 'none';
        
        annotation('ellipse',[.527-.01, .495 .025 .0425])
        annotation('ellipse',[.49-.012, .23 .1 .17])
        
        annotation('textarrow',[0.54-.01,0.54-.01],[0.485,0.41]);
        
        
        %%%plot(t(glottis), pressure(vt_ind, glottis), '.c')

%         ut_area = [area(vt_ind, upper_tract),area(vt_ind, upper_tract(end))];
%         ut_t = [t(upper_tract),t(upper_tract(end)+1)];
%         stairs(ut_t, 0.5*ut_area, '-b','LineWidth',1.5)
%         stairs(ut_t, -0.5*ut_area, '-b','LineWidth',1.5)
%         %%%plot(t(tract), pressure(vt_ind, tract), '.c')

        n_area = [area(vt_ind, nasal),area(vt_ind, nasal(end))];
        n_t = [t(nasal+nasal_hor),t(nasal(end)+nasal_hor+1)];
        plot(n_t,zeros(size(n_t))+nasal_vert,'k--');
        stairs(n_t, 0.5*n_scale*n_area+nasal_vert, '-b','LineWidth',1.5)
        stairs(n_t, -0.5*n_scale*n_area+nasal_vert, '-b','LineWidth',1.5)
        
        na = annotation('textbox',[.72-.01,0.8,0.1,0.05],'String','Nasal Cavity','FitBoxToText','on');
        na.Color = 'Blue';
        na.FontSize = 12;
        na.LineStyle = 'none';

        %%%plot(t(nasal+nasal_hor), pressure(vt_ind, nasal)+nasal_vert, '.c')

        hold off
        drawnow
        %writeVideo(v, getframe(gcf));
        I = getframe(gcf);
        step(v,I.cdata);%,Snd(j));
        if j==loops
            break;
        elseif (take_snapshot== true) && j==round(loops/2)
            set(f1,'PaperPosition',[.25,1.5,psize])
            print('-f1',[fname(1:end-4),'_snapshot'],'-depsc','-r150');
            saveas(f1,[fname(1:end-4),'_snapshot'],'fig');
        end
        clf;
    end
    release(v);
    clf;
end

% Combine audio and video files using ffmpeg like this
% ffmpeg -i primlog0.avi -i primlog0.wav primlog0_combined.mp4