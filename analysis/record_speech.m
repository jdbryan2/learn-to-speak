% Record Speech
clear

T = 1; % Number of seconds to record for
fs = 9000;
recObj = audiorecorder(fs,16,1,0);

ipa_num = num2str(133);
num_examples = 5;
ex = 1;
while ex <= num_examples
    disp('Start speaking.')
    pause(0.5)
    recordblocking(recObj,T);
    disp('End of Recording.')

    pause(2)

    disp('Replaying recording.')
    play(recObj);

    y = getaudiodata(recObj);

    answer = input('Keep Recording\n','s');
    if strcmp(answer,'y')
        % Save recording
        display(['Saving Example ',num2str(ex)]);
        filename = ['TestMySpeech1/logs/ipa',ipa_num,'_ex',num2str(ex)];
        save(filename,'y','fs','T');
        ex = ex + 1;
    elseif strcmp(answer,'q')
        break;
    end     
end