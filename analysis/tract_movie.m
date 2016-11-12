function tract_movie(area1,Fs,area2)

if nargin < 3
   compare = false;
else
    compare = true;
end
% Plot area function and/or pressure function as video
sub_sample = 1;
lung_scale = .5;
[~,ts1] = size(area1);
loops = floor(ts1/sub_sample);
if compare
    [~,ts2] = size(area2);
    loops_ref = floor(ts2/sub_sample);
end

F(loops) = struct('cdata',[],'colormap',[]);
v = VideoWriter('vocal_tract.avi');

min_area = min(min(area1'));
if compare
    min_area = min(min_area,min(min(area2')));
    area2 = area2-min_area;
end
area1 = area1-min_area;

max_vt = max(max(area1(1:22,:)'));
max_lung = max(max(area1(23:89,:)'));
max_area = max(max_lung*lung_scale,max_vt*.5);
if compare
    max_vt = max(max(area2(1:22,:)'));
    max_lung = max(max(area2(23:89,:)'));
    max_area = max(max_area,max_lung*lung_scale,max_vt*.5);
end
max_area = 1.5*max_area;

figure(12);
for j = 1:loops
    plot(-21:0, lung_scale*area1(1:22,j*sub_sample), '-r')
    hold on
    plot(-21:0,-lung_scale*area1(1:22,j*sub_sample), '-r')
    plot(0.5*area1(23:89,j*sub_sample), '-b')
    plot(-0.5*area1(23:89,j*sub_sample), '-b')
    if compare
        ind = j;
        if j> loops_ref
            ind = loops_ref;
        end
        plot(-21:0, lung_scale*area2(1:22,ind*sub_sample), '.r')
        plot(-21:0,-lung_scale*area2(1:22,ind*sub_sample), '.r')
        plot(0.5*area2(23:89,ind*sub_sample), '.b')
        plot(-0.5*area2(23:89,ind*sub_sample), '.b')
    end
    hold off
    axis([-22,69,-max_area,max_area]);
    drawnow
    F(j) = getframe(gcf);
end

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

fig = gcf;
movie(fig,F,3,5)