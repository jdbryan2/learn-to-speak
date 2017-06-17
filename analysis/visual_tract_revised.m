clear; %#ok<CLALL>
close all;

for trial = 1:1
    % Plot area function and/or pressure function as video

    [data, labels, Fs, data_len] = import_datalog(sprintf('testStim3Batch300/artword_logs/ipa132_ex%i.log', trial));
    %data(1, :) = []; % remove first row because its shitty. JB Why is it bad? JW
    %data_len = data_len-1;
    sub_sample = 1;
    loops = floor(data_len/sub_sample);
    %%%pressure = data(:, 90:178);

    % take only the tubes that were used and rescale to atm
    %pressure = pressure(:, 6:77); %pressure-mean(mean(pressure));
    %pressure = pressure/142054.26;
    %%%pressure= pressure-mean(mean(pressure(:, 7:16),1));
    %%%pressure = pressure/max(max(abs(pressure(:, 7:77))));
    %%%pressure = pressure*1e-3;

    area= data(:, 1:89);
    
    audio = data(:, end);
    audio = audio/max(abs(audio));

    audiowrite(sprintf('testStim3Batch300/artword_logs/ipa101_ex%i.wav', trial), audio, Fs, 'BitsPerSample', 64)

    v = VideoWriter(sprintf('testStim3Batch300/artword_logs/ipa101_ex%i.avi', trial),'Uncompressed AVI'); %#ok<TNMLP>
    v.FrameRate = Fs/sub_sample;
    open(v);

    t = 1:64+1;
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

    glott_vert = -5.1526e-04;%(max(max(area(:,upper_tract)))+30*max(max(area(:,glottis))))/1;
    
    nasal_vert = (max(max(area(:, upper_tract)))+max(max(area(:, nasal))))/1.9;
    nasal_hor = -13-1;
    
    % full screen figure
    f1 = figure(1);
    f1.Units = 'normalized';
    f1.Position = [0 0 1 1];
    
    for j = 1:loops
        tb = annotation('textbox',[0.8,0.2,0.1,0.1],'String',['Scaling:' char(10),'Blue - 1:1' char(10),'Red - 1:20' char(10),'Green - 5:1']);
        %tb.Color = 'magenta';
        tb.FontSize = 16;
        
        plot(t(7:end),zeros(1,65-6),'k--','LineWidth',1.5);
        title(sprintf('Vocal Tract Area Function: Trial %i', trial),'FontSize',16);
        ylabel('Area in cm^3')
        ylim([-1.2*lmax*l_scale/2,1.2*lmax*l_scale/2]);
        set(gca,'fontsize',18)
        set(gca,'XTickLabel','')
        hold on
        l_area = [area(j*sub_sample, lower_resp),area(j*sub_sample, lower_resp(end))];
        l_t = [t(lower_resp),t(lower_resp(end)+1)];
        stairs(l_t, 0.5*l_scale*l_area, '-r','LineWidth',1.5)
        stairs(l_t,-0.5*l_scale*l_area, '-r','LineWidth',1.5)
        la = annotation('textbox',[0.25,0.1,0.1,0.05],'String','Lungs','FitBoxToText','on');
        la.Color = 'red';
        la.FontSize = 16;
        la.LineStyle = 'none';
        
        ba = annotation('textbox',[0.39,0.43,0.1,0.05],'String','Bronchi','FitBoxToText','on');
        ba.Color = 'red';
        ba.FontSize = 16;
        ba.LineStyle = 'none';
        %%%plot(t(lungs)+1, pressure(j*sub_sample, lungs+1), '.c')

        stairs(t(tract), 0.5*t_scale*area(j*sub_sample, tract), '-b','LineWidth',1.5)
        stairs(t(tract), -0.5*t_scale*area(j*sub_sample, tract), '-b','LineWidth',1.5)
        
        ta = annotation('textbox',[0.47,0.58,0.1,0.05],'String','Trachea','FitBoxToText','on');
        ta.Color = 'Blue';
        ta.FontSize = 16;
        ta.LineStyle = 'none';
        
        pa = annotation('textbox',[.58,0.39,0.1,0.05],'String','Pharynx','FitBoxToText','on');
        pa.Color = 'Blue';
        pa.FontSize = 16;
        pa.LineStyle = 'none';
        
        oca = annotation('textbox',[.69,0.39,0.1,0.05],'String','Oral Cavity','FitBoxToText','on');
        oca.Color = 'Blue';
        oca.FontSize = 16;
        oca.LineStyle = 'none';
        %%%plot(t(trachea), pressure(j*sub_sample, trachea), '.c')

        g_area = [area(j*sub_sample, glottis),area(j*sub_sample, glottis(end))];
        g_t = [t(glottis),t(glottis(end)+1)];
        %stairs(g_t, 0.5*t_scale*g_area, '--g','LineWidth',1.5)
        %stairs(g_t, -0.5*t_scale*g_area, '--g','LineWidth',1.5)
        plot(g_t,zeros(size(g_t))+glott_vert,'k--');
        stairs(g_t, 0.5*g_scale*g_area+glott_vert, '-g','LineWidth',1.5)
        stairs(g_t, -0.5*g_scale*g_area+glott_vert, '-g','LineWidth',1.5)
        
        ga = annotation('textbox',[0.52,0.16,0.1,0.05],'String','Glottis','FitBoxToText','on');
        ga.Color = 'green';
        ga.FontSize = 16;
        ga.LineStyle = 'none';
        
        annotation('ellipse',[.527 .49 .025 .05])
        annotation('ellipse',[.49 .23 .1 .2])
        
        annotation('textarrow',[0.54,0.54],[0.485,0.44]);
        
        
        %%%plot(t(glottis), pressure(j*sub_sample, glottis), '.c')

%         ut_area = [area(j*sub_sample, upper_tract),area(j*sub_sample, upper_tract(end))];
%         ut_t = [t(upper_tract),t(upper_tract(end)+1)];
%         stairs(ut_t, 0.5*ut_area, '-b','LineWidth',1.5)
%         stairs(ut_t, -0.5*ut_area, '-b','LineWidth',1.5)
%         %%%plot(t(tract), pressure(j*sub_sample, tract), '.c')

        n_area = [area(j*sub_sample, nasal),area(j*sub_sample, nasal(end))];
        n_t = [t(nasal+nasal_hor),t(nasal(end)+nasal_hor+1)];
        plot(n_t,zeros(size(n_t))+nasal_vert,'k--');
        stairs(n_t, 0.5*n_scale*n_area+nasal_vert, '-b','LineWidth',1.5)
        stairs(n_t, -0.5*n_scale*n_area+nasal_vert, '-b','LineWidth',1.5)
        
        na = annotation('textbox',[.74,0.80,0.1,0.05],'String','Nasal Cavity','FitBoxToText','on');
        na.Color = 'Blue';
        na.FontSize = 16;
        na.LineStyle = 'none';

        %%%plot(t(nasal+nasal_hor), pressure(j*sub_sample, nasal)+nasal_vert, '.c')

        hold off
        drawnow
        writeVideo(v, getframe(gcf));
        if j==loops
            break;
        end
        clf;
    end
    close(v);
end