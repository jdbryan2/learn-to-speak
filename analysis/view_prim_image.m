function [] = view_prim_image(K,O,p,f,k,num_vars,lab,dt,num_logs,mk,nk,save_figs,naming);
    % Graph Input and Output Primitives in subplot form
    % K = Input Prims
    % O = Output Prims
    % p = # past times used in prims
    % f = # of future times used in prims
    % k = # of Prims
    % num_vars = number of variables in observation vector sample
    % lab = String labels for observation variables
    % dt = Time between samples
    % num_logs = number of logs used to create Prims
    % mk = number of rows of subplot
    % nk = number of columns in subplot
    % save_figs = Boolean to save figures to .fig and .eps files
    % naming = {testname,data_type,config} cell array of strings for
    %   filenameing
    
    % Note: This uses a hacky way to make surf not delete 1 row and 1 col at end of data
    % could use imagesc instead, but I think label editing is harder
    % http://www.mathworks.com/examples/matlab/community/6386-offsets-and-discarded-data-via-pcolor-and-surf
    % Or use interpolation which doesn't get rid of values
    %surf((1:p)*dt,1:num_vars,Ps{i},'EdgeColor','none'); %shading interp
    

    leg = [];
    figure(2); clf; f2 = gcf;
    [ha2,~] =tight_subplot(mk,nk,[.01 .03],[.1 .05],[.1 .2]);
    figure(3); clf; f3 = gcf;
    [ha3,~] = tight_subplot(mk,nk,[.01 .03],[.1 .05],[.1 .2]);
    figure(4); clf; f4 = gcf;
    ylab_ind = 1:floor(num_vars/5):num_vars;
    Kmin = min(min(K)); Kmax = max(max(K));
    Omin = min(min(O)); Omax = max(max(O));
    for i=1:k
        Ps{i} = reshape(K(i,:),[num_vars,p]);
        Fs{i} = reshape(O(:,i),[num_vars,f]);
        figure(2) % Graph Input Prims
        axes(ha2(i));
        surf(((0:p))*dt,(1:num_vars+1)-.5,[[Ps{i}; zeros(1,p)],zeros(num_vars+1,1)],'EdgeColor','none');
        axis xy; axis tight; colormap(hot); view(0,90);

        figure(3) % Graph Output Prims
        axes(ha3(i));
        surf(((0:f))*dt,(1:num_vars+1)-.5,[[Fs{i}; zeros(1,f)],zeros(num_vars+1,1)],'EdgeColor','none');
        axis xy; axis tight; colormap(hot); view(0,90);
        if mod(i-1,nk)
            set(ha2(i),'Ytick',[]);
            set(ha3(i),'Ytick',[]);
        else
            set(ha2(i),'clim',[Kmin,Kmax])
            set(ha2(i),'YTick',ylab_ind,'YTickLabel',lab(ylab_ind),...
                'YTickLabelRotation',45)
            set(ha3(i),'clim',[Omin,Omax])
            set(ha3(i),'YTick',ylab_ind,'YTickLabel',lab(ylab_ind),...
                'YTickLabelRotation',45)
        end
        if i/((mk-1)*nk)<1 && k<mk*nk
            set(ha2(i),'Xtick',[]);
            set(ha3(i),'Xtick',[]);
        else
            set(ha2(i),'XTickLabelRotation',45);
            set(ha3(i),'XTickLabelRotation',45);
        end
    end
    for i=k+1:mk*nk
        axes(ha2(i));
        set(gca,'Visible','off')
        axes(ha3(i));
        set(gca,'Visible','off')
    end

    h2 = mtit(f2,'Input Primitives','yoff',0.02);
    xl = xlabel(h2.ah,'Time (sec)','Visible','on');
    set(xl,'position',xl.Position+[0,-0.02,0]);
    ylabel(h2.ah,'Frequency (Hz)','Visible','on')
    set(h2.ah,'FontSize',14)
    set(h2.ah,'clim',[Kmin,Kmax])
    colorbar(h2.ah)

    h3 = mtit(f3,'Output Primitives','yoff',0.02);
    xl = xlabel(h3.ah,'Time (sec)','Visible','on');
    set(xl,'position',xl.Position+[0,-0.02,0]);
    ylabel(h3.ah,'Frequency (Hz)','Visible','on')
    set(h3.ah,'FontSize',14)
    set(h3.ah,'clim',[Omin,Omax])
    colorbar(h3.ah)

    if save_figs == true
        set(f2,'PaperPosition',[.25,1.5,8,5])
        %saveas(f2,[testname,'/',filename,'_in-map_',config],'epsc');
        print('-f2',[naming{1},'/',naming{2},'_in-map_',naming{3}],'-depsc','-r150');
        saveas(f2,[naming{1},'/',naming{2},'_in-map_',naming{3}],'fig');
        set(f3,'PaperPosition',[.25,1.5,8,5])
        %saveas(f3,[testname,'/',filename,'_out-map_',config],'epsc');
        print('-f3',[naming{1},'/',naming{2},'_out-map_',naming{3}],'-depsc','-r150');
        saveas(f3,[naming{1},'/',naming{2},'_out-map_',naming{3}],'fig');
    end
end