clear all; %#ok<CLALL>
close all;

for trial = 1:50
    % Plot area function and/or pressure function as video

    [data, labels, Fs, data_len] = import_datalog(sprintf('../data/datalog%i.log', trial));
    data(1, :) = []; % remove first row because its shitty
    data_len = data_len-1;
    sub_sample = 100;
    loops = floor(data_len/sub_sample);
    pressure = data(:, 90:178);

    % take only the tubes that were used and rescale to atm
    %pressure = pressure(:, 6:77); %pressure-mean(mean(pressure));
    %pressure = pressure/142054.26;
    pressure= pressure-mean(mean(pressure(:, 7:16),1));
    pressure = pressure/max(max(abs(pressure(:, 7:77))));
    pressure = pressure*1e-3;

    area= data(:, 1:89);
    
    audio = data(:, end);
    audio = audio/max(abs(audio));

    audiowrite(sprintf('audio/Nov17_trial_%i.wav', trial), audio, Fs, 'BitsPerSample', 64)

    v = VideoWriter(sprintf('video/Nov17_trial_%i.avi', trial),'Uncompressed AVI'); %#ok<TNMLP>
    v.FrameRate = Fs/sub_sample;
    open(v);

    t = 1:64;
    t = t-20;
    lungs = 6:18;
    trachea = 21:35;
    glottis = 35:37;
    tract = 37:64;
    nasal = 64:77;

    nasal_vert = max(max(area(:, tract)))+max(max(area(:, nasal)));
    nasal_hor = -13;
    
    % full screen figure
    figure('units','normalized','position',[0 0 1 1])
    title(sprintf('Trial %i', trial));

    for j = 1:loops
        plot(t(lungs), 0.025*area(j*sub_sample, lungs), '-r')
        hold on
        plot([t(18), t(21)], [0.025*area(j*sub_sample, 18), 0.5*area(j*sub_sample, 21)], '-r') % connect lungs
        plot(t(lungs),-0.025*area(j*sub_sample, lungs), '-r')
        plot([t(18), t(21)], -[0.025*area(j*sub_sample, 18), 0.5*area(j*sub_sample, 21)], '-r') % connect lungs

        plot(t(lungs)+1, pressure(j*sub_sample, lungs+1), '.c')

        plot(t(trachea), 0.5*area(j*sub_sample, trachea), '-b')
        plot(t(trachea), -0.5*area(j*sub_sample, trachea), '-b')
        plot(t(trachea), pressure(j*sub_sample, trachea), '.c')

        plot(t(glottis), 0.5*area(j*sub_sample, glottis), '-g')
        plot(t(glottis), -0.5*area(j*sub_sample, glottis), '-g')
        plot(t(glottis), pressure(j*sub_sample, glottis), '.c')

        plot(t(tract), 0.5*area(j*sub_sample, tract), '-b')
        plot(t(tract), -0.5*area(j*sub_sample, tract), '-b')
        plot(t(tract), pressure(j*sub_sample, tract), '.c')

        plot(t(nasal+nasal_hor), 0.5*area(j*sub_sample, nasal)+nasal_vert, '-b')
        plot(t(nasal+nasal_hor), -0.5*area(j*sub_sample, nasal)+nasal_vert, '-b')

        plot(t(nasal+nasal_hor), pressure(j*sub_sample, nasal)+nasal_vert, '.c')

        hold off
        drawnow
        writeVideo(v, getframe(gcf));
    end
    close(v);
    clf;
end
