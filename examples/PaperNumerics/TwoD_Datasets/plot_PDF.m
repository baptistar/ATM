sd = 1; rng(sd);

%% --- Define test case ---

% define problems
d = 2;
problems = {@Banana, @MoG, @Funnel, @Cosine, @Ring};

% define sample size and run
N_vect = 10000;
MCruns = 1;

if ~exist('./figures', 'dir')
    mkdir('./figures')
end

% define domain for plotting PDF
Nplot = 200;
[X,Y] = meshgrid(linspace(-6,6,Nplot));
Z = [X(:), Y(:)];

for l=1:length(problems) 

    % define problems and sample sizes
    P = problems{l}(d);
    
    for j=1:length(N_vect)
        for run=MCruns

            N = N_vect(j);
            fprintf('Problem: %s, N = %d\n', P.name, N);
    
            % load data
            load(['rotation_data/' P.name '_N10000'], 'Q')

            % load data for map
            file_name = ['data/' P.name '_N' num2str(N) '_run' num2str(run) '.mat'];
            load(file_name, 'CM','output')

            % evaluate the log-density
            log_pi = CM.log_pdf(Z);

            figure()
            log_pdf = reshape(P.log_pdf(Z*Q), Nplot, Nplot);
            contourf(X, Y, exp(log_pdf));
            colormap(brewermap([],'Blues'))
            lim = caxis;
            axis equal
            set(gca,'LooseInset',get(gca,'TightInset'));
            set(gca,'box','off');
            set(gca,'visible','off');
            print('-depsc',['figures/' P.name '_truePDF_N' num2str(N)])
            close all

            figure()
            log_pdf_approx = reshape(sum(log_pi,2), Nplot, Nplot);
            contourf(X, Y, exp(log_pdf_approx))
            colormap(brewermap([],'Blues'))
            caxis(lim)
            axis equal
            set(gca,'LooseInset',get(gca,'TightInset'));
            set(gca,'box','off');
            set(gca,'visible','off');
            print('-depsc',['figures/' P.name '_approxPDF_N' num2str(N)])
            close all

        end
    end
    
end 

% -- END OF FILE --
