function flex_paper
%% this file was copied and adjusted from final results
%% This file can/will be revised any time


outdat='/datc/flex_8state/data_extend_Power264AdjACC/';
outfig='/datc/flex_8state/fig_paper/';
p_dyn='/datc/dynNet/code/';
p_flex='/datc/flex/';
addpath(genpath(p_dyn));
addpath(genpath([p_flex 'code']));
addpath('./export_fig/');
%addpath(genpath('/datc/software/brat/ccm')); %% compute clustering coeficient

nROI = 264;
listfile = load([p_dyn 'hcp_S453_sex_age_avg.txt']);

tname = {'rfMRI_REST1_LR','tfMRI_GAMBLING_LR','tfMRI_MOTOR_LR','tfMRI_SOCIAL_LR',  ...
         'tfMRI_EMOTION_LR',  'tfMRI_LANGUAGE_LR',  'tfMRI_RELATIONAL_LR', 'tfMRI_WM_LR'};
%tname = {'rfMRI_REST1_RL','tfMRI_GAMBLING_RL','tfMRI_MOTOR_RL','tfMRI_SOCIAL_RL',  ...
%         'tfMRI_EMOTION_RL',  'tfMRI_LANGUAGE_RL',  'tfMRI_RELATIONAL_RL', 'tfMRI_WM_RL'};

% 
%%
%% %% 
% disp('Reading original connectivity matrix ...');
% load([p_flex 'data_CompCor/revise_avg_corrR_all_remMActivation.mat']);  % corrR_all = nan(8,nSubj,nROI,nROI);

nStat=8;
%nStat=size(corrR_all,1);

nNet=10;
%% predefine the subNet names and 13 net index
netName={'Sensory', 'CON', 'Auditory','DMN', 'VAN', 'Visual', ...
         'FPN', 'Salience', 'DAN', 'Subcort.', 'Memory','Cerebellar', 'Uncertain'};     
Sensory=[13:46,255]; % 35
Cingulo=[47:60]; % 14
Auditory=[61:73]; % 13
DMN=[74:83,86:131,137,139]; % 58
DMNm=[75,76,88,89, 91,92, 94, 105:113];
Memory=[133:136,221]; % 5
Vatt=[138,235:242]; % 9
Visual=[143:173]; % 31
FPC=[174:181,186:202]; % 25
Salience=[203:220]; % 18
Subcort=[222:234]; % 13
Cerebellar=[243:246]; % 4
Datt=[251:252,256:264]; % 11
Uncertain=[ 1:12, 84:85,132, 140:142, 182:185, 247:250, 253:254]; % 28
powerPart=zeros(nROI,1); %% partition assignment of Power 264
powerPart(Sensory)=1;
powerPart(Cingulo)=2;
powerPart(Auditory)=3;
powerPart(DMNm)=4;
powerPart(Vatt)=5;
powerPart(Visual)=6;
powerPart(FPC)=7;
powerPart(Salience)=8;
powerPart(Datt)=9;

powerPart(Subcort)=10;
powerPart(Memory)=11;
powerPart(Cerebellar)=12;
powerPart(Uncertain)=13;

Colormap=[0.8 0.8 0.8];
cpool=[1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; 0.5 0.5 0; 0.5 0 0.5; ...
           0 0.5 0.5; 0.5 0.5 0.5; 1 0.5 0; 1 0 0.5; 0 0.5 1];
c264pool=nan(nROI,3);
for i=1:nROI
    c264pool(i,:)=cpool(powerPart(i),:);
end

addpath(genpath('/datc/software/BCT/BCT_20150125'));
addpath(genpath('/datc/dynNet/code'));  % network_thr_bin()

roiCoord=load('/datc/dynNet/code/roi264_Power.txt'); % [264,3]
bnCoord=load('/datc/flex_8state/code/BN_Atlas_246_coord.txt');

if  0
    %% Illustration: Power atlas
    %Ind10net=find(powerPart<11); % only the 10 Network nodes
    c264pool(powerPart>11,:)=repmat([1 0.5 0], [length(find(powerPart>11)),1]);
    %figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(powerPart(Ind10net),   ...
    %    roiCoord(Ind10net,:), c264pool(Ind10net,:), 5*ones(length(Ind10net),1));
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(powerPart,   ...
        roiCoord, c264pool, 5*ones(nROI,1));
    set(gcf,'Color', 'w');export_fig([outfig 'Illustration_Power_10Partition.tif'], '-dtiff', '-r300');

    %% show_bar_legend
    cpool=[1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; 0.5 0.5 0; 0.5 0 0.5; ...
           0 0.5 0.5; 0.5 0.5 0.5; 1 0.5 0; 1 0 0.5; 0 0.5 1];
    netName={'Sensory', 'CON', 'Auditory','DMN', 'VAN', 'Visual', ...
         'FPN', 'Salience', 'DAN', 'Subcort.',  'Others'};   
    nNet = 11;
    figure('Position',[100,100, 800,800]);
    bh=barh(0.09*ones(1,nNet)); set(bh, 'ShowBaseLine', 'off'); % 'LineStyle', 'none',% remove black fram
    %h=get(b, 'children');
    bh.FaceColor='flat';  bh.CData=cpool(nNet:-1:1,:);
    ylim([0 nNet+1]); xlim([0 1.2]); axis off;
    ax=xlim; ay=ylim;
    x1=ax(1); x2=ax(2); y1=ay(1); y2=ay(2);
    xdist=x2-x1; ydist=y2-y1;
    for i=1:nNet
        x=x1+0.11*xdist;
        y=y1+0.91*i/nNet*ydist;
        text(x,y, netName{nNet-i+1}, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',26);
    end
    set(gcf,'Color', 'w');export_fig([outfig 'Illustration_10Partition_legend.tif'], '-dtiff', '-r300');
    
    %% 11 balls illustration
    figure('Position',[100,100, 1500,900]);
    [x,y,z]=sphere(100);   roiWeit=15;
    roiCoord=[zeros(nNet, 1), [1:40:440]', zeros(nNet, 1)];
    for i=1:size(roiCoord,1)
        hh=surf(x*roiWeit+roiCoord(i,1),y*roiWeit+roiCoord(i,2),z*roiWeit+roiCoord(i,3)); 
        set(hh, 'FaceColor', cpool(i,:),'EdgeColor', 'none');
        hold on;
    end
    view(-90,0); 
    daspect([1 1 1]); 
    set(gcf,'Color',[1 1 1],'InvertHardcopy','off');
    lighting phong; material([0.5 0.5 0.5]); % shading interp; 
    axis tight; axis vis3d off;
    camlight('right'); 
    set(gcf,'Color', 'w');export_fig([outfig 'Illustration_10Partition_11balls.tif'], '-dtiff', '-r300');
end
    
if 0
    %% ACC data, correlation and hist
    load([outdat 'corrR_Rest_WM_bk0_bk2_ACC.mat']); % 'Rest', 'WM', 'bk0', 'bk2', 'ac0', 'ac2', 'ac', 'list_adj'
    figure('Position',[100,100, 1300,400]);
    subplot(1,3,1); plot_regress(ac0, ac2); 
    set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    [r, p]=corr(ac0, ac2); legstr=sprintf('r = %0.2f\np = %0.2e', r, p);
    text(83, 47, legstr, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    xlabel('Accuracy in 0-back (%)', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);  
    ylabel('Accuracy in 2-back (%)', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    subplot(1,3,2); hist(ac0, 50);  set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    xlabel('Accuracy in 0-back (%)', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16); 
    subplot(1,3,3); hist(ac2, 50);  set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    xlabel('Accuracy in 2-back (%)', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    set(gcf,'Color', 'w');export_fig([outfig 'Plot_ACC_corr_hist.tif'], '-dtiff', '-r300');
end

if  0
    %% Contrast analysis
    %% REST1 vs. REST2 as contrast level, prediction balls for each nodes on surface
    load([outdat 'corrR_Rest_WM_bk0_bk2_ACC.mat']); % 'Rest', 'WM', 'bk0', 'bk2', 'ac0', 'ac2', 'ac', 'list_adj'
    load([p_flex 'data_CompCor/revise_avg_corrR_REST2_remMActivation_Power264.mat']);  % corrR_REST2 = nan(nSubj,nROI,nROI);
    %% for avg 453, 2 outliers: 145834, 181131
    listREST2 = load([p_dyn 'hcp_S446_sex_age_avg_REST2.txt']);
   
    [C,ia,ib]=intersect(list_adj(:,1), listREST2(:,1));
    Rest = Rest(ia,:,:);
    ac0=ac0(ia); ac2=ac2(ia); ac=ac(ia);
    
    corrR_REST2=corrR_REST2(ib,:,:);    
    nSubj = length(ia);
    
    NR=nan(nSubj, nROI);
    for i=1:nSubj
        NR(i,:) = 1-abs(diag(corr(squeeze(corrR_REST2(i,:,:)), squeeze(Rest(i,:,:)))));
    end
    %% to plot the AV on the surface for each node by a ball
    [~,~,~,~, stats]=regress(ac0, [ones(nSubj,1),NR]); fprintf('ac0: %0.3f\n', stats(1)); %: 0.658
    [~,~,~,~, stats]=regress(ac2, [ones(nSubj,1),NR]); fprintf('ac2: %0.3f\n', stats(1)); %: 0.731
    [~,~,~,~, stats]=regress(ac, [ones(nSubj,1),NR]); fprintf('ac: %0.3f\n', stats(1));  %: 0.717
    ball_size=nan(nROI,3);
    for i=1:nROI
        [~,~,~,~, stats]=regress(ac0, [ones(nSubj,1),NR(:,i)]); ball_size(i,1)=stats(1);
        [~,~,~,~, stats]=regress(ac2, [ones(nSubj,1),NR(:,i)]); ball_size(i,2)=stats(1);
        [~,~,~,~, stats]=regress(ac, [ones(nSubj,1),NR(:,i)]); ball_size(i,3)=stats(1);
    end
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(powerPart, roiCoord, c264pool, ball_size(:,1)*100); %suptitle('ac0');
        set(gcf,'Color', 'w');export_fig([outfig 'Prediction_eachROI_ball_surface_rest1_rest2_ac0.tif'], '-dtiff', '-r300')
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(powerPart, roiCoord, c264pool, ball_size(:,2)*100); %suptitle('ac2');
        set(gcf,'Color', 'w');export_fig([outfig 'Prediction_eachROI_ball_surface_rest1_rest2_ac2.tif'], '-dtiff', '-r300')
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(powerPart, roiCoord, c264pool, ball_size(:,3)*100); %suptitle('ac');
        set(gcf,'Color', 'w');export_fig([outfig 'Prediction_eachROI_ball_surface_rest1_rest2_ac.tif'], '-dtiff', '-r300')
end


if 0
    %% to repeat the right part of Fig. 2, evaluate the individual contribution of each ROI (node)
    %% then to show the balls on the surface
    load([outdat 'corrR_Rest_WM_bk0_bk2_ACC.mat']); % 'Rest', 'WM', 'bk0', 'bk2', 'ac0', 'ac2', 'ac', 'list_adj'
    ACC=[ac0, ac2, ac2, ac];
    load([outdat 'NCD_4contrast.mat']); % NCD=nan(4, nSubj, nROI);
    nSubj=size(NCD,2);
    ball_size=nan(nROI,1);
    %rest>bk0
    ncd=squeeze(NCD(1,:,:));
    for i=1:nROI
        [~,~,~,~, stats]=regress(ac0, [ones(nSubj,1),ncd(:,i)]);
        ball_size(i)=stats(1);
    end
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(powerPart, roiCoord, c264pool, ball_size*100); %suptitle('rest vs. bk0');
        set(gcf,'Color', 'w');export_fig([outfig 'Prediction_eachROI_ball_surface_rest_bk0.tif'], '-dtiff', '-r300')

    %bk0>bk2
    ncd=squeeze(NCD(3,:,:));
    for i=1:nROI
        [~,~,~,~, stats]=regress(ac2, [ones(nSubj,1),ncd(:,i)]);
        ball_size(i)=stats(1);
    end
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(powerPart, roiCoord, c264pool, ball_size*200); %suptitle('bk0 vs. bk2 (3scale)');
        set(gcf,'Color', 'w');export_fig([outfig 'Prediction_eachROI_ball_surface_bk0_bk2_scale2.tif'], '-dtiff', '-r300')
        
end

if 1
    %% to repeat the Fig. 3, to evaluate the nodal (in 10 networks) AV and network AV, stem plot (with color patch)
    load([outdat 'corrR_Rest_WM_bk0_bk2_ACC.mat']); % 'Rest', 'WM', 'bk0', 'bk2', 'ac0', 'ac2', 'ac', 'list_adj'
    ACC=[ac0, ac2, ac2, ac];
    load([outdat 'NCD_4contrast.mat']); % NCD=nan(4, nSubj, nROI);
    nSubj=size(NCD,2);
    
    figure('Position',[100,100, 1600,700]);
    ball_size=nan(nROI,1); stem_size=nan(nNet,1);
    reSamp=310;
    errBar=zeros(nNet, 2); % two column, [lower bar, uper bar]
    nSamp=5000;
    
    %% rest>bk0
    ncd=squeeze(NCD(1,:,:));
    % Avg. Nodal AV, firt for each ROI and then average the AV
    fprintf('rest>bk0: Avg. Nodal AV\n');
    for i=1:nROI
        [~,~,~,~, stats]=regress(ACC(:,1), [ones(nSubj,1),ncd(:,i)]);
        ball_size(i)=stats(1);
    end
    for i=1:nNet
        stem_size(i)=mean(ball_size(powerPart==i));
    end
    tmp=zeros(nNet, nSamp);
    for J=1:nSamp % boot strapping
        nSel=randperm(nSubj, reSamp);
        ball_size=nan(nROI,1);
        for i=1:nROI
            [~,~,~,~, stats]=regress(ACC(nSel,1), [ones(reSamp,1),ncd(nSel,i)]);
            ball_size(i)=stats(1);
        end
        for i=1:nNet
            tmp(i, J)=mean(ball_size(powerPart==i));
        end
    end
    errBar(:,1) = stem_size - (mean(tmp, 2) - std(tmp, 0, 2));
    errBar(:,2) = mean(tmp, 2) + std(tmp, 0, 2) - stem_size;
    subplot(2,4,1); bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45); ylim([0, Inf]); % new from Matlab2017
    ylabel(['Avg. Nodal AV'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    stem_err(1).stem=stem_size;
    stem_err(1).errBar=errBar;
    stem_err(1).tmp = tmp;
    fprintf('rest>bk0: multiple regress\n');
    for i=1:nNet
        [~,~,~,~, stats]=regress(ACC(:,1), [ones(nSubj,1),ncd(:,powerPart==i)]);
        stem_size(i)=stats(1);
    end
    tmp=zeros(nNet, nSamp);
    for J=1:nSamp % boot strapping
        nSel=randperm(nSubj, reSamp);
        for i=1:nNet
            [~,~,~,~, stats]=regress(ACC(nSel,1), [ones(reSamp,1),ncd(nSel,powerPart==i)]);
            tmp(i, J)=stats(1);
        end
    end
    errBar(:,1) = stem_size - (mean(tmp, 2) - std(tmp, 0, 2));
    errBar(:,2) = mean(tmp, 2) + std(tmp, 0, 2) - stem_size;
    subplot(2,4,2), bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45);  ylim([0, Inf]);% new from Matlab2017
    ylabel(['Network AV'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    % Network AV: first mean NCD for each network nodes, then std regress
    stem_err(2).stem=stem_size;
    stem_err(2).errBar=errBar;
    stem_err(2).tmp = tmp;
    fprintf('rest>bk0: mean networkal NCD\n');
    rMat=nan(size(ncd,1),nNet);
    for i=1:nNet
        rMat(:,i)=mean(ncd(:,powerPart==i), 2);
    end
    [~, stem_size] = stdCorr(ACC(:,1), rMat);
    tmp=zeros(nNet, nSamp);
    for J=1:nSamp % boot strapping
        rMat2=zeros(reSamp, nNet);
        nSel=randperm(nSubj, reSamp);
        for i=1:nNet
            rMat2(:,i)=mean(ncd(nSel,powerPart==i), 2);
        end
        [~, tmp(:, J)] = stdCorr(ACC(nSel,1), rMat2);
    end
    errBar(:,1) = stem_size - (mean(tmp, 2) - std(tmp, 0, 2));
    errBar(:,2) = mean(tmp, 2) + std(tmp, 0, 2) - stem_size;
    subplot(2,4,3), bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45);  ylim([0, Inf]);% new from Matlab2017
    ylabel(['Tot. Network AV (std)'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    % Network AV: first mean NCD for each network nodes, then semipart regress
    % [~, stem_size] = semipartCorr(ACC(:,1), rMat);
    %stem_size = model_PRE(ACC(:,1), rMat);
    %subplot(2,4,4), bh=bar(stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    stem_err(3).stem=stem_size;
    stem_err(3).errBar=errBar;
    stem_err(3).tmp = tmp;
    fprintf('rest>bk0: modal selection, networkal NCD\n');
    [stem_size, errBar, tmp] = model_PRE(ACC(:,1), rMat, reSamp);
    subplot(2,4,4), bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b 
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45);  ylim([0, Inf]);% new from Matlab2017
    ylabel(['Tot. Network AV (semipart)'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    stem_err(4).stem=stem_size;
    stem_err(4).errBar=errBar;
    stem_err(4).tmp = tmp;
    %% bk0>bk2
    ncd=squeeze(NCD(3,:,:));
    % Avg. Nodal AV, firt for each ROI and then average the AV
    fprintf('bk0>bk2: Avg. Nodal AV\n');
    for i=1:nROI
        [~,~,~,~, stats]=regress(ACC(:,3), [ones(nSubj,1),ncd(:,i)]);
        ball_size(i)=stats(1);
    end
    for i=1:nNet
        stem_size(i)=mean(ball_size(powerPart==i));
    end
    tmp=zeros(nNet, nSamp);
    for J=1:nSamp % boot strapping
        nSel=randperm(nSubj, reSamp);
        ball_size=nan(nROI,1);
        for i=1:nROI
            [~,~,~,~, stats]=regress(ACC(nSel,3), [ones(reSamp,1),ncd(nSel,i)]);
            ball_size(i)=stats(1);
        end
        for i=1:nNet
            tmp(i, J)=mean(ball_size(powerPart==i));
        end
    end
    errBar(:,1) = stem_size - (mean(tmp, 2) - std(tmp, 0, 2));
    errBar(:,2) = mean(tmp, 2) + std(tmp, 0, 2) - stem_size;
    subplot(2,4,5), bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45);  ylim([0, Inf]);% new from Matlab2017
    ylabel(['Avg. Nodal AV'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    stem_err(5).stem=stem_size;
    stem_err(5).errBar=errBar;
    stem_err(5).tmp = tmp;
    % Network AV (regress to get AV based on each network)
    fprintf('bk0>bk2: multiple regress\n');
    for i=1:nNet
        [~,~,~,~, stats]=regress(ACC(:,3), [ones(nSubj,1),ncd(:,powerPart==i)]);
        stem_size(i)=stats(1);
    end
    tmp=zeros(nNet, nSamp);
    for J=1:nSamp % boot strapping
        nSel=randperm(nSubj, reSamp);
        for i=1:nNet
            [~,~,~,~, stats]=regress(ACC(nSel,3), [ones(reSamp,1),ncd(nSel,powerPart==i)]);
            tmp(i, J)=stats(1);
        end
    end
    errBar(:,1) = stem_size - (mean(tmp, 2) - std(tmp, 0, 2));
    errBar(:,2) = mean(tmp, 2) + std(tmp, 0, 2) - stem_size;
    subplot(2,4,6), bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45);  ylim([0, Inf]);% new from Matlab2017
    ylabel(['Network AV'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    stem_err(6).stem=stem_size;
    stem_err(6).errBar=errBar;
    stem_err(6).tmp = tmp;
    % Network AV: first mean NCD for each network nodes, then std regress
    fprintf('bk0>bk2: mean networkal NCD\n');
    rMat=nan(size(ncd,1),nNet);
    for i=1:nNet
        rMat(:,i)=mean(ncd(:,powerPart==i), 2);
    end
    [~, stem_size] = stdCorr(ACC(:,3), rMat);
    tmp=zeros(nNet, nSamp);
    for J=1:nSamp % boot strapping
        rMat2=zeros(reSamp, nNet);
        nSel=randperm(nSubj, reSamp);
        for i=1:nNet
            rMat2(:,i)=mean(ncd(nSel,powerPart==i), 2);
        end
        [~, tmp(:, J)] = stdCorr(ACC(nSel,3), rMat2);
    end
    errBar(:,1) = stem_size - (mean(tmp, 2) - std(tmp, 0, 2));
    errBar(:,2) = mean(tmp, 2) + std(tmp, 0, 2) - stem_size;
    subplot(2,4,7), bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45);  ylim([0, Inf]);% new from Matlab2017
    ylabel(['Tot. Network AV (std)'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    stem_err(7).stem=stem_size;
    stem_err(7).errBar=errBar;
    stem_err(7).tmp = tmp;
    % Network AV: first mean NCD for each network nodes, then semipart regress
    % [~, stem_size] = semipartCorr(ACC(:,3), rMat);
    fprintf('bk0>bk2: modal selection, networkal NCD\n');
    [stem_size, errBar, tmp] = model_PRE(ACC(:,3), rMat, reSamp);
    subplot(2,4,8), bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45);  ylim([0, Inf]);% new from Matlab2017
    ylabel(['Tot. Network AV (semipart)'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    stem_err(8).stem=stem_size;
    stem_err(8).errBar=errBar;
    stem_err(8).tmp = tmp;
    
    save([outdat 'Prediction_10Net_stemBar_AV_2contrast_stem_errBar.mat'], 'stem_err');
    
    set(gcf,'Color', 'w');export_fig([outfig 'Prediction_10Net_stemBar_AV_4contrast.tif'], '-dtiff', '-r300');
    
end

if 0
    %% Draw the adj/unadj mosaic figure
    load([outdat 'member_concensus_acrossSubject_mean_sp.mat']); %'memCensus_meanSubj', 'memCensus_sp'

    % Adj/Unadj mosaic map: mean across subjects
    figure('Position',[100,100, 1300,800]);
    strTitle={'Rest', '0-back', '2-back'};
    for i=1:3 % only rest, bk0, bk2
        tmp = memCensus_meanSubj(i).nmi_unadj; tmp(logical(eye(size(tmp))))=1;
        subaxis(2,3,i, 'SpacingVert', 0, 'SpacingHoriz',0.02);
        imshow(imresize(tmp, 20, 'nearest'), []);  title(strTitle{i}, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
        colormap(gca, redbluecmap); colorbar; 
        set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);        
        tmp = memCensus_meanSubj(i).nmi_adj; tmp(logical(eye(size(tmp))))=1;
        subaxis(2,3,i+3, 'SpacingVert', 0, 'SpacingHoriz',0.02);
        imshow(imresize(tmp, 20, 'nearest'), []);  title(strTitle{i}, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
        colormap(gca, redbluecmap); colorbar; 
        set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);        
    end
    %% save figure
    set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_AdjUnadj_acrossSubject_mean_rmT10.tif'], '-dtiff', '-r300');
    
    % Adj/Unadj mosaic map: spMat
    figure('Position',[100,100, 1800,500]);
    %% rest - bk0
    tmp = memCensus_sp(1).nmi_unadj; tmp(logical(eye(size(tmp))))=1;
    subaxis(1,4,1, 'SpacingVert', 0, 'SpacingHoriz',0.02);
    imshow(imresize(tmp, 20, 'nearest'), []);  title('Rest vs. 0-back', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
    colormap(gca, redbluecmap); colorbar; 
    set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
    tmp = memCensus_sp(1).nmi_adj; tmp(logical(eye(size(tmp))))=1;
    subaxis(1,4,2, 'SpacingVert', 0, 'SpacingHoriz', 0.02);
    imshow(imresize(tmp, 20, 'nearest'), []);   title('Rest vs. 0-back (Adj)', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
    colormap(gca, redbluecmap); colorbar; 
    set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
    %% bk0 - bk2
    tmp = memCensus_sp(3).nmi_unadj; tmp(logical(eye(size(tmp))))=1;
    subaxis(1,4,3, 'SpacingVert', 0, 'SpacingHoriz',0.02);
    imshow(imresize(tmp, 20, 'nearest'), []);  title('0-back vs. 2-back', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
    colormap(gca, redbluecmap); colorbar; 
    set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
    tmp = memCensus_sp(3).nmi_adj; tmp(logical(eye(size(tmp))))=1;
    subaxis(1,4,4, 'SpacingVert', 0, 'SpacingHoriz', 0.02);
    imshow(imresize(tmp, 20, 'nearest'), []);  title('0-back vs. 2-back (Adj)', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
    colormap(gca, redbluecmap); colorbar; 
    set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
    
    %% save figure
    set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_AdjUnadj_spMat_rmT10.tif'], '-dtiff', '-r300');
end

if 0
    %% Draw the comunity for showing the ball on surface
    mypos='_rmT10';
    load([outdat 'member_concensus_acrossSubject_mean_sp' mypos '.mat']); %'memCensus_meanSubj', 'memCensus_sp'
   %% spMat
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(memCensus_sp(1).S, roiCoord, [], ones(nROI,1)*5); %suptitle('REST1 vs. bk0');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_spMat_REST1_bk0' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(memCensus_sp(2).S, roiCoord, [], ones(nROI,1)*5); %suptitle('REST1 vs. bk2');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_spMat_REST1_bk2' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(memCensus_sp(3).S, roiCoord, [], ones(nROI,1)*5); %suptitle('bk0 vs. bk2');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_spMat_bk0_bk2' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(memCensus_sp(4).S, roiCoord, [], ones(nROI,1)*5); %suptitle('REST1 vs. WM');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_spMat_REST1_WM' mypos '.tif'], '-dtiff', '-r300');
        
    %% meanSubj
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(memCensus_meanSubj(1).S, roiCoord, [], ones(nROI,1)*5); %suptitle('REST1');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_acrossSubject_mean_REST1' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(memCensus_meanSubj(2).S, roiCoord, [], ones(nROI,1)*5); %suptitle('bk0');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_acrossSubject_mean_bk0' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(memCensus_meanSubj(3).S, roiCoord, [], ones(nROI,1)*5); %suptitle('bk2');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_acrossSubject_mean_bk2' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 900,700]), show_surface_roi_ModAssign(memCensus_meanSubj(4).S, roiCoord, [], ones(nROI,1)*5); %suptitle('WM');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_acrossSubject_mean_WM' mypos '.tif'], '-dtiff', '-r300');
end



end  % main function

function [R, R2] = semipartCorr(Y, X)  %% Y=[n,1]; X=[n,m] or [n,m,k]
% have been confirmed when m=2,k=1 by semipartialcorr in Matlab
% fileexchange by May 2014, Maik C. Stttgen (validated by SPSS V20)
% Lauriola M., The SAGE Encyclopedia of Social Research Methods[M]. 
% Thousand Oaks, CA: Sage Publications, Inc., 2004
    [nLen, M,N]=size(X);
    if N==1 % dim=2
       R=nan(M,1);
       R2=nan(M,1);
       for i=1:M
           tmp=X; tmp(:,i)=[];
           [~,~,tmp1]=regress(X(:,i), [ones(nLen,1), tmp]);
           R(i) = corr(Y, tmp1);
           [b, bint, r, rint, stats] = regress(Y, [ones(nLen,1),tmp1]);
           R2(i) = stats(1);
       end
    elseif N==M  % dim=3
       upInd= logical(triu(ones(M,M),1));
       X=reshape(X,nLen,[]);
       X=X(:, upInd(:));
       [Rt, R2t] = semipartCorr(Y,X);
       R=zeros(M,M); R2=zeros(M,M);
       R(upInd(:))=Rt; R=R+R';
       R2(upInd(:))=R2t; R2=R2+R2';
    else
        exit('Input matrix uncorrect dimension!');
    end
end

function [R, R2] = stdCorr(Y, X)  %% Y=[n,1]; X=[n,m] or [n,m,k]
% have been confirmed when m=2,k=1 by semipartialcorr in Matlab
% fileexchange by May 2014, Maik C. Stttgen (validated by SPSS V20)
    [nLen, M,N]=size(X);
    if N==1 % dim=2
       R=nan(M,1);
       R2=nan(M,1);
       for i=1:M
           R(i) = corr(Y, X(:,i));
           [b, bint, r, rint, stats] = regress(Y, [ones(nLen,1),X(:,i)]);  
           R2(i) = stats(1);
       end
    elseif N==M  % dim=3
       upInd= logical(triu(ones(M,M),1));
       X=reshape(X,nLen,[]);
       X=X(:, upInd(:));
       [Rt, R2t] = stdCorr(Y,X);
       R=zeros(M,M); R2=zeros(M,M);
       R(upInd(:))=Rt; R=R+R';
       R2(upInd(:))=R2t; R2=R2+R2';
    else
        exit('Input matrix uncorrect dimension!');
    end

end

function [PRE, errBar, tmp] = model_PRE(Y, X, reSamp)
    % https://www.researchgate.net/post/How_can_I_determine_the_relative_contribution_of_predictors_in_multiple_regression_models
    % PRE = (Residual Sum of Squares of M2 - Residual Sum of Squares of M1) / Residual Sum of Squares of M2
    % Judd, C.M., McClelland, G.H., & Ryan, C.S. (2008). Data analysis: A model comparison approach. Routledge.
    [nLen, M]=size(X);
    PRE=zeros(M,1);
    for i=1:M
         tmp=X; tmp(:, i)=[];
         [~,~, r1]=regress(Y, [ones(nLen,1), X]);
         [~,~, r2]=regress(Y, [ones(nLen,1), tmp]);
         PRE(i) = (sum(r2.*r2)-sum(r1.*r1))/sum(r2.*r2);
    end
    
    errBar=zeros(M, 2); % three columns: [lower error, uper error]
    if exist('reSamp', 'var')
        tmp=zeros(M, 50);  %% 5000 permutations
        for i=1:50
            nSel=randperm(nLen, reSamp);
            tmp(:,i) = model_PRE(Y(nSel), X(nSel, :));
        end
        errBar(:,1)= PRE - ( -std(tmp,0, 2) + mean(tmp, 2));
        errBar(:,2)=std(tmp,0, 2) + mean(tmp, 2) - PRE;
    end
end

