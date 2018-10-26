function flex_extendResults_rest_0bk_2bk_Power264AdjACC_Activation(nStart, nEnd)
% This is to extend the results presented in manuscript v10
% After the discussion with Prof. Yihong Yang on Oct. 17, 2017
% This code is shared by fig_extend_Power264AdjACC and  fig_extend_Power264AdjACCFisherz
% Test log: 
% 1. Raw ACC vs. Fisherz (z-score normalize): no difference
% 2. (ac2-ac0)./(ac2+ac0)*2.0 as the measure of bk2 vs. bk0

outdat='/datc/flex_8state/data_extend_Power264AdjACC_Act/';
outfig='/datc/flex_8state/fig_extend_Power264AdjACC_Act/';
p_dyn='/datc/dynNet/code/';
p_flex='/datc/flex/';
addpath(genpath(p_dyn));
addpath(genpath([p_flex 'code']));
addpath(genpath('./'));
addpath(genpath('/datc/software/matlabTools/'));
%addpath(genpath('/datc/software/brat/ccm')); %% compute clustering coeficient

nROI = 264;
listfile = load([p_dyn 'hcp_S453_sex_age_avg.txt']);

nSubj = size(listfile, 1);

tshort= {'Rest', 'Gambling', 'Motor', 'Social', ...
          'Emotion', 'Language', 'Relational', 'WM'};
roiCoord=load('/datc/dynNet/code/roi264_Power.txt');          
%% %% 
if 0 % prepare data
    disp('Reading original connectivity matrix ...');
    load([p_flex 'data_CompCor/revise_avg_corrR_all_remMActivation.mat']);  % corrR_all = nan(8,nSubj,nROI,nROI);
    Rest=squeeze(corrR_all(1,:,:,:)); WM=squeeze(corrR_all(8,:,:,:)); clear corrR_all;
    disp('Reading 0-back and 2-back correlation matrix');
    load('/datc/flex/data_CompCor/corrR_WM_block_original_avg.mat'); % corrR_WM_block = nan(2,nSubj,nROI,nROI);
    bk0=squeeze(corrR_WM_block(1,:,:,:)); bk2=squeeze(corrR_WM_block(2,:,:,:));  clear corrR_WM_block;
    %save([outdat 'corrR_Rest_WM_bk0_bk2.mat'], 'Rest', 'WM', 'bk0', 'bk2');

    %load([outdat 'corrR_Rest_WM_bk0_bk2.mat']); % 'Rest', 'WM', 'bk0', 'bk2'
    %% AC=load([p_flex 'data_CompCor/WM_ACC_Trials_avg.mat']); % bk0=[453,4], bk2=[453,4] %% deprecated!!!
    ac0=extract_behaveMeasure_fromS900csv('WM_Task_0bk_Acc');
    ac2=extract_behaveMeasure_fromS900csv('WM_Task_2bk_Acc');
    ac=extract_behaveMeasure_fromS900csv('WM_Task_Acc');

    %% remove the outliers
    [~, idx1] = delete_outliers(ac0);  [~, idx2] = delete_outliers(ac2);
    idx=union(idx1, idx2);
    if ~isempty(idx)
        disp('Outlier subjects:'); %% 2 outliers: 145834, 181131
        disp(listfile(idx,1));
        ac0(idx)=[]; ac2(idx)=[]; ac(idx)=[]; 
        Rest(idx,:,:)=[]; WM(idx,:,:)=[]; bk0(idx,:,:)=[]; bk2(idx,:,:)=[]; 
        list_adj=listfile;
        list_adj(idx, :)=[];
    end
    
    %% to remove the ACC<80 in the 0-back task, suggested by Yihong Yang, Dec. 21, 2017
    idx = (ac0<80);
    ac0(idx)=[]; ac2(idx)=[]; ac(idx)=[]; 
    Rest(idx,:,:)=[]; WM(idx,:,:)=[]; bk0(idx,:,:)=[]; bk2(idx,:,:)=[]; 
    list_adj(idx, :)=[];
    %% Fisher-z the scores to unify the data,
    %% according to Cole's paper, and email by Dr. Greg Burgess, Staff Scientist, Human Connectome Project
    % ACC = [nSubj, 4]; rest vs. bk0, rest vs. bk2, bk0 vs. bk2, rest vs. WM
%     ACC=zeros(size(Rest,1),4);
%     ACC(:,1)=fisherz(ac0/110); ACC(:,2)=fisherz(ac2/110); ACC(:,3)=fisherz((ac2-ac0)./(ac2+ac0)*2.0); ACC(:,4)=fisherz(ac/110); 
%     ACC = zscore(ACC);
%     
    %% to plot_regress() between ac0, ac2, ac
    figure('Position',[100,100, 1300,400]);
    subplot(1,3,1); plot_regress(ACC(:,1), ACC(:,2)); xlabel('ac0'); ylabel('ac2');
        [r, p]=corr(ACC(:,1), ACC(:,2)); title(sprintf('corr(ac0,ac2): r=%0.2f, p=%0.2e', r, p));
    subplot(1,3,2); plot_regress(ACC(:,1), ACC(:,3)); xlabel('ac0'); ylabel('ac20');
        [r, p]=corr(ACC(:,1), ACC(:,3)); title(sprintf('corr(ac0,ac20): r=%0.2f, p=%0.2e', r, p));
    subplot(1,3,3); plot_regress(ACC(:,2), ACC(:,3)); xlabel('ac2'); ylabel('ac20');
        [r, p]=corr(ACC(:,2), ACC(:,3)); title(sprintf('corr(ac2,ac20): r=%0.2f, p=%0.2e', r, p));
    set(gcf,'Color', 'w');export_fig([outfig 'Plot_corr_AC0_AC2_AC20.tif'], '-dtiff', '-r300');
    
   %% to plot the hist of ac0, ac2, ac
   figure('Position',[100,100, 1300,400]);
    subplot(1,3,1); hist(ACC(:,1), 50); xlabel('ac0'); 
    subplot(1,3,2); hist(ACC(:,2), 50); xlabel('ac2');
    subplot(1,3,3); hist(ACC(:,3), 50); xlabel('ac20'); 
    set(gcf,'Color', 'w');export_fig([outfig 'Plot_hist_AC0_AC2_AC20.tif'], '-dtiff', '-r300');
    
%    save([outdat 'corrR_Rest_WM_bk0_bk2_ACC.mat'], 'Rest', 'WM', 'bk0', 'bk2', 'ACC', 'list_adj');
%     
%     %% z-score the the accuracy, Cole et al, JN 2016. This is why we setup _Power264AdjACC
%     %% Results: no difference with the no adjustion. Reasonable since it is a linear transform. Dec. 17, 2017
%     %% However, in this case, norminv() will be approaching to Inf
%     % ac0=zscore(ac0);  ac2=zscore(ac2);  ac=zscore(ac);  
%     disp('0-back measures');
%     ac_bk0_body_tar=extract_behaveMeasure_fromS900csv('WM_Task_0bk_Body_Acc_Target')/100;
%     ac_bk0_body_non=extract_behaveMeasure_fromS900csv('WM_Task_0bk_Body_Acc_Nontarget')/100;
%     ac_bk0_face_tar=extract_behaveMeasure_fromS900csv('WM_Task_0bk_Face_Acc_Target')/100;
%     ac_bk0_face_non=extract_behaveMeasure_fromS900csv('WM_Task_0bk_Face_ACC_Nontarget')/100;
%     ac_bk0_place_tar=extract_behaveMeasure_fromS900csv('WM_Task_0bk_Place_Acc_Target')/100;
%     ac_bk0_place_non=extract_behaveMeasure_fromS900csv('WM_Task_0bk_Place_Acc_Nontarget')/100;
%     ac_bk0_tool_tar=extract_behaveMeasure_fromS900csv('WM_Task_0bk_Tool_Acc_Target')/100;
%     ac_bk0_tool_non=extract_behaveMeasure_fromS900csv('WM_Task_0bk_Tool_Acc_Nontarget')/100;
%     disp('2-back measures');
%     ac_bk2_body_tar=extract_behaveMeasure_fromS900csv('WM_Task_2bk_Body_Acc_Target')/100;
%     ac_bk2_body_non=extract_behaveMeasure_fromS900csv('WM_Task_2bk_Body_Acc_Nontarget')/100;
%     ac_bk2_face_tar=extract_behaveMeasure_fromS900csv('WM_Task_2bk_Face_Acc_Target')/100;
%     ac_bk2_face_non=extract_behaveMeasure_fromS900csv('WM_Task_2bk_Face_Acc_Nontarget')/100;
%     ac_bk2_place_tar=extract_behaveMeasure_fromS900csv('WM_Task_2bk_Place_Acc_Target')/100;
%     ac_bk2_place_non=extract_behaveMeasure_fromS900csv('WM_Task_2bk_Place_Acc_Nontarget')/100;
%     ac_bk2_tool_tar=extract_behaveMeasure_fromS900csv('WM_Task_2bk_Tool_Acc_Target')/100;
%     ac_bk2_tool_non=extract_behaveMeasure_fromS900csv('WM_Task_2bk_Tool_Acc_Nontarget')/100;
%     
%     %% Detect outliers
%     [~, idx1] = delete_outliers(ac_bk0_body_tar);  [~, idx2] = delete_outliers(ac_bk0_body_non);
%     [~, idx3] = delete_outliers(ac_bk0_face_tar); [~, idx4] = delete_outliers(ac_bk0_face_non);
%     [~, idx5] = delete_outliers(ac_bk0_place_tar); [~, idx6] = delete_outliers(ac_bk0_place_non);
%     [~, idx7] = delete_outliers(ac_bk0_tool_tar); [~, idx8] = delete_outliers(ac_bk0_tool_non);
%     [~, idx9] = delete_outliers(ac_bk2_body_tar); [~, idx10] = delete_outliers(ac_bk2_body_non);
%     [~, idx11] = delete_outliers(ac_bk2_face_tar); [~, idx12] = delete_outliers(ac_bk2_face_non);
%     [~, idx13] = delete_outliers(ac_bk2_place_tar); [~, idx14] = delete_outliers(ac_bk2_place_non);
%     [~, idx15] = delete_outliers(ac_bk2_tool_tar); [~, idx16] = delete_outliers(ac_bk2_tool_non);
%     ac0=(norminv(ac_bk0_body_tar)-norminv(1-ac_bk0_body_non)  + norminv(ac_bk0_face_tar)-norminv(1-ac_bk0_face_non) + ...
%             norminv(ac_bk0_place_tar)-norminv(1-ac_bk0_place_non) + norminv(ac_bk0_tool_tar)-norminv(1-ac_bk0_tool_non) )/4;
%     ac2=(norminv(ac_bk2_body_tar)-norminv(1-ac_bk2_body_non)  + norminv(ac_bk2_face_tar)-norminv(1-ac_bk2_face_non) + ...
%             norminv(ac_bk2_place_tar)-norminv(1-ac_bk2_place_non) + norminv(ac_bk2_tool_tar)-norminv(1-ac_bk2_tool_non) )/4;
%     ac=(ac0+ac2)/2;
%     idx=union(union(union(union(union(union(union(union(union(union(union(union(union(union(union(idx1, idx2), idx3), idx4), idx5), idx6), idx7), idx8), idx9), idx10), idx11), idx12), idx13), idx14), idx15), idx16);
%     if ~isempty(idx)
%         disp('Outlier subjects:'); %% 2 outliers: 145834, 181131
%         disp(listfile(idx,1));
%         ac0(idx)=[]; ac2(idx)=[]; ac(idx)=[]; 
%         Rest(idx,:,:)=[]; WM(idx,:,:)=[]; bk0(idx,:,:)=[]; bk2(idx,:,:)=[]; 
%     end    
%    save([outdat 'corrR_Rest_WM_bk0_bk2_ACC.mat'], 'Rest', 'WM', 'bk0', 'bk2', 'ac0', 'ac2', 'ac');
end
load(['/datc/flex_8state/data_extend_Power264AdjACC/corrR_Rest_WM_bk0_bk2_ACC.mat']); % 'Rest', 'WM', 'bk0', 'bk2', 'ac0', 'ac2', 'ac', 'list_adj'

nSubj=size(Rest,1);
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
powerPart(DMN)=4;
powerPart(Vatt)=5;
powerPart(Visual)=6;
powerPart(FPC)=7;
powerPart(Salience)=8;
powerPart(Datt)=9;

powerPart(Subcort)=10;
powerPart(Memory)=11;
powerPart(Cerebellar)=12;
powerPart(Uncertain)=13;

netSize=zeros(nNet,1);
for i=1:nNet
     netSize(i)=length(find(powerPart==i));  % [35    14    13    58     9    31    25    18    11    13]
end

Colormap=[0.8 0.8 0.8];
cpool=[1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; 0.5 0.5 0; 0.5 0 0.5; ...
           0 0.5 0.5; 0.5 0.5 0.5; 1 0.5 0; 1 0 0.5; 0 0.5 1];
c264pool=nan(nROI,3);
for i=1:nROI
    c264pool(i,:)=cpool(powerPart(i),:);
end

if 0  % REST1 vs. REST2 as contrast level
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
    [~,~,~,~, stats]=regress(ac0, [ones(nSubj,1),NR]); fprintf('ac0: %0.3f\n', stats(1));
    [~,~,~,~, stats]=regress(ac2, [ones(nSubj,1),NR]); fprintf('ac2: %0.3f\n', stats(1));
    [~,~,~,~, stats]=regress(ac, [ones(nSubj,1),NR]); fprintf('ac: %0.3f\n', stats(1));
    ball_size=nan(nROI,3);
    for i=1:nROI
        [~,~,~,~, stats]=regress(ac0, [ones(nSubj,1),NR(:,i)]); ball_size(i,1)=stats(1);
        [~,~,~,~, stats]=regress(ac2, [ones(nSubj,1),NR(:,i)]); ball_size(i,2)=stats(1);
        [~,~,~,~, stats]=regress(ac, [ones(nSubj,1),NR(:,i)]); ball_size(i,3)=stats(1);
    end
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(powerPart, roiCoord, c264pool, ball_size(:,1)*100); suptitle('ac0');
        set(gcf,'Color', 'w');export_fig([outfig 'Prediction_eachROI_ball_surface_rest1_rest2_ac0.tif'], '-dtiff', '-r300')
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(powerPart, roiCoord, c264pool, ball_size(:,2)*100); suptitle('ac2');
        set(gcf,'Color', 'w');export_fig([outfig 'Prediction_eachROI_ball_surface_rest1_rest2_ac2.tif'], '-dtiff', '-r300')
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(powerPart, roiCoord, c264pool, ball_size(:,3)*100); suptitle('ac');
        set(gcf,'Color', 'w');export_fig([outfig 'Prediction_eachROI_ball_surface_rest1_rest2_ac.tif'], '-dtiff', '-r300')
        
   %% DO NOT make sense  
    % perform the cross validation.  
    nVal=5; nPre=nSubj-nVal;
    nTest=200;
    valRes=nan(nTest, 3);
    for i=1:nTest
        samPool=randperm(nSubj);
        iVal=samPool(1:nVal); iPre=samPool(nVal+1:end);
        Yp = [NR(iVal,:), ones(nVal,1)] *([NR(iPre,:),ones(nPre,1)]\ac(iPre));
        [r,p]=corr(Yp, ac(iVal)); [~,~,~,~, stats] = regress(Yp, [ones(nVal,1),ac(iVal)]);
        valRes(i,:)=[r, p, stats(1)];
    end
    figure, plot(valRes); legend({'R', 'p', 'R2'});
    saveas(gcf, [outfig 'CrossValidation_Rest1_Rest2_ac.tif']);
    disp(mean(valRes,1));
    
    
    % plot the prediction plot for the whole cohort. % compare to Fig. 1
    figure('Position',[100,100, 1200,600]);
    Yp = [NR, ones(size(NR,1),1)] *([NR,ones(size(NR,1),1)]\ac0);
    [r,p]=corr(Yp, ac0); [~,~,~,~, stats] = regress(Yp, [ones(nSubj,1),ac0]);
    subplot(1,3,1); plot_regress(ac0, Yp); set(gca,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    title(sprintf('rest1 > rest2, r=%1.2f, p=%1.2e, R^2=%1.2f', r, p, stats(1)));
    %
    Yp = [NR, ones(size(NR,1),1)] *([NR,ones(size(NR,1),1)]\ac2);
    [r,p]=corr(Yp, ac2); [~,~,~,~, stats] = regress(Yp, [ones(nSubj,1),ac2]);
    subplot(1,3,2); plot_regress(ac2, Yp); set(gca,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    title(sprintf('rest1 > rest2, r=%1.2f, p=%1.2e, R^2=%1.2f', r, p, stats(1)));
    %
    Yp = [NR, ones(size(NR,1),1)] *([NR,ones(size(NR,1),1)]\ac);
    [r,p]=corr(Yp, ac); [~,~,~,~, stats] = regress(Yp, [ones(nSubj,1),ac]);
    subplot(1,3,3); plot_regress(ac, Yp); set(gca,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    title(sprintf('rest1 > rest2, r=%1.2f, p=%1.2e, R^2=%1.2f', r, p, stats(1)));
    
end

%% DO NOT make sense  
if 0 % perform the cross validation
    load([outdat 'NCD_4contrast.mat']); % NCD=nan(4, nSubj, nROI); % 4 contrasts, rest>bk0, rest>bk2, bk0>bk2, rest>WM
    nSubj=size(NCD,2);
    
    nVal=300; nPre=nVal;%nSubj-nVal; 
    nTest=200;
    valRes=nan(nTest, 3);
    NR=squeeze(NCD(4,:,:));
    for i=1:nTest
        samPool=randperm(nSubj);
        iVal=samPool(1:nVal); iPre=iVal; %samPool(nVal+1:end);
        Yp = [NR(iVal,:), ones(nVal,1)] *([NR(iPre,:),ones(nPre,1)]\ac(iPre));
        [r,p]=corr(Yp, ac(iVal)); [~,~,~,~, stats] = regress(Yp, [ones(nVal,1),ac(iVal)]);
        valRes(i,:)=[r, p, stats(1)];
        if stats(1)>0.3
            tmptmp = 1;
        end
    end
    figure, plot(valRes); legend({'R', 'p', 'R2'});
    saveas(gcf, [outfig 'CrossValidation_Rest1_WM_ac.tif']);
    disp(mean(valRes,1));
    
    newNR=zeros(size(NR));
    for i=1:nSubj
        newNR(i,:)=NR(i, randperm(nROI));
    end
    Yp = [newNR, ones(nSubj,1)] *([newNR,ones(nSubj,1)]\ac);
    [r,p]=corr(Yp, ac); [~,~,~,~, stats] = regress(Yp, [ones(nSubj,1),ac]);
    figure, plot_regress(ac, Yp);
    
end

if 0  %% this is to compute the concensus of the correlation matrix directly from [nSubj, NCD] matrix
    load([outdat 'corrMatrixFromNCD_5contrast.mat']);  % NCD5contr=zeros(5, nROI, nROI); %% 1 is REST1 vs. REST2; 2-4 is NCD
    thresh = [1 2 3 5 10]; T = length(thresh);
    gama=[0.4:0.2:1.6]; G = length(gama);
    nROI = size(NCD5contr,2);

    spMat = NCD5contr;
    memb=zeros(size(spMat,1), T,G,nROI);
    for i=1:size(spMat,1)
        fprintf('%d ', i);
        avMat=squeeze(spMat(i,:,:)); avMat=avMat .* (avMat>0);
        for j= 1:T
%            disp(thresh(j));
            thisMat = avMat .* network_thr_bin(avMat, thresh(j));
            %[memb, qual] = paco_mx(thisMat, 'quality', 2, 'nrep',100);
            for k=1:G
                [Q, tmp] = Mucha_2D(thisMat, 100, gama(k));
                memb(i, j,k,:) = tmp;
            end
        end
    end
    fprintf('\n');
    
    %% memb concensus
    for i=1:size(spMat,1)
        n_m=T*G;
        memb2 = reshape(squeeze(memb(i,:,:,:)), [n_m, nROI]);
        nmi_unadj=zeros(n_m, n_m);
        nmi_adj=zeros(n_m, n_m);
        fprintf('\n NMI across Thresh and Gamma \n');
        for j=1:n_m-1
            if mod(j,300) ==0 %% very slow for unadjusted; very^10 slow for adjusted
                fprintf('%d ', j);
            end
            for k=j+1:n_m
                nmi_adj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'adjusted'); 
                nmi_unadj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'unadjusted'); 
            end
        end
        nmi_unadj = nmi_unadj + nmi_unadj';
        nmi_adj = nmi_adj + nmi_adj';
        [~, ~, Xnew, qpc]=consensus_iterative(memb2); % we need >100 iterations to get a steady result
        [Q, S] = Mucha_2D(Xnew, 200);
        memCensus(i).nmi_unadj = nmi_unadj;
        memCensus(i).nmi_adj = nmi_adj;
        memCensus(i).S = S;
    end
    
    save([outdat 'corrMatrixFromNCD_5contrast_memb_Census.mat'], 'memb', 'memCensus');
    
end


if 0  % Generate 4 contrast NCD
    NCD=nan(4, nSubj, nROI); % 4 contrasts, rest>bk0, rest>bk2, bk0>bk2, rest>WM
    for i=1:nSubj
        tmp1=squeeze(Rest(i,:,:)); tmp2=squeeze(bk0(i,:,:)); tmp3=squeeze(bk2(i,:,:)); tmp4=squeeze(WM(i,:,:)); 
        NCD(1,i,:) = 1-abs(diag(corr(tmp1,tmp2))); % ABS is recommended (higher prediction), see flex_8state.m
        NCD(2,i,:) = 1-abs(diag(corr(tmp1,tmp3)));
        NCD(3,i,:) = 1-abs(diag(corr(tmp2,tmp3)));
        NCD(4,i,:) = 1-abs(diag(corr(tmp1,tmp4)));
    end
    save([outdat 'NCD_4contrast.mat'], 'NCD');
end
%load([outdat 'NCD_4contrast.mat']); % NCD=nan(4, nSubj, nROI);

%% for the Activation based analysis, NCD was replaced by Activation series
load('/datc/flex_8state/data_extend/Activation_WM_ROI_series_Power264.mat'); %% roiSeries =[[0-back/baseline, 2-back/0-back], nSubj, nROI]
NCD=nan(4, nSubj, nROI);
NCD([1,3],:,:)=roiSeries;
ACC=[ac0,ac2,ac2,ac];










%% DO NOT make sense  
if 0
    %% to repeat the Fig. 1, prediction vs. measurement, 4 contrasts, resting>bk0, resting>bk2, bk0>bk2, rest>WM
    figure('Position',[100,100, 1200,1200]);
    %rest>bk0
    ncd=squeeze(NCD(1,:,:));
    Yp = [ncd, ones(size(ncd,1),1)] *([ncd,ones(size(ncd,1),1)]\ac0);
    [r,p]=corr(Yp, ac0); [~,~,~,~, stats] = regress(Yp, [ones(nSubj,1),ac0]);
    subplot(2,2,1); plot_regress(ac0, Yp); set(gca,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    title(sprintf('rest > bk0, r=%1.2f, p=%1.2e, R^2=%1.2f', r, p, stats(1)));
    %rest>bk2
    ncd=squeeze(NCD(2,:,:));
    Yp = [ncd, ones(size(ncd,1),1)] *([ncd,ones(size(ncd,1),1)]\ac2);
    [r,p]=corr(Yp, ac2);  [~,~,~,~, stats] = regress(Yp, [ones(nSubj,1),ac2]);
    subplot(2,2,2); plot_regress(ac2, Yp); set(gca,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    title(sprintf('rest > bk2, r=%1.2f, p=%1.2e, R^2=%1.2f',r, p, stats(1)));
    %bk0>bk2
    ncd=squeeze(NCD(3,:,:));
    Yp = [ncd, ones(size(ncd,1),1)] *([ncd,ones(size(ncd,1),1)]\ac2);
    [r,p]=corr(Yp, ac2);  [~,~,~,~, stats] = regress(Yp, [ones(nSubj,1),ac2]);
    subplot(2,2,3); plot_regress(ac2, Yp); set(gca,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    title(sprintf('bk0 > bk2 (by ACC 2bk), r=%1.2f, p=%1.2e, R^2=%1.2f', r, p, stats(1)));
    %rest>WM
    ncd=squeeze(NCD(4,:,:));
    Yp = [ncd, ones(size(ncd,1),1)] *([ncd,ones(size(ncd,1),1)]\ac);
    [r,p]=corr(Yp, ac);  [~,~,~,~, stats] = regress(Yp, [ones(nSubj,1),ac]);
    subplot(2,2,4); plot_regress(ac, Yp); set(gca,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
    title(sprintf('rest > WM, r=%1.2f, p=%1.2e, R^2=%1.2f', r, p, stats(1)));

    set(gcf, 'PaperPositionMode', 'manual');  set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperPosition', [100 100 1200 800]);
    print(gcf, '-dtiff', '-r300', [outfig 'Prediction_overall_plot_regress_4contrast.tif']);
end

if 0
    %% to repeat the right part of Fig. 2, evaluate the individual contribution of each ROI (node)
    %% then to show the balls on the surface
    ball_size=nan(nROI,1);
    %rest>bk0
    ncd=squeeze(NCD(1,:,:));
    for i=1:nROI
        [~,~,~,~, stats]=regress(ACC(:,1), [ones(nSubj,1),ncd(:,i)]);
        ball_size(i)=stats(1);
    end
%    export_ModMemb_netfile_PowerPart(ball_size, 1, [outfig 'Prediction_eachROI_ball_surface_rest_bk0.net'], powerPart);
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(powerPart, roiCoord, c264pool, ball_size*100); suptitle('rest vs. bk0');
        set(gcf,'Color', 'w');export_fig([outfig 'Prediction_eachROI_ball_surface_rest_bk0.tif'], '-dtiff', '-r300')
    
    %bk0>bk2
    ncd=squeeze(NCD(3,:,:));
    for i=1:nROI
        [~,~,~,~, stats]=regress(ACC(:,3), [ones(nSubj,1),ncd(:,i)]);
        ball_size(i)=stats(1);
    end
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(powerPart, roiCoord, c264pool, ball_size*100); suptitle('bk0 vs. bk2');
        set(gcf,'Color', 'w');export_fig([outfig 'Prediction_eachROI_ball_surface_bk0_bk2.tif'], '-dtiff', '-r300')
            
end

if  1  %% to repeat the Fig. 3, to evaluate the nodal (in 10 networks) AV and network AV, stem plot (with color patch)
       %% This is similar to the following one, but adopt multiple variable regressing
       %% March 27, 2018
    nSubj=size(NCD,2);
    figure('Position',[100,100, 1200,700]);
    stem_size=nan(nNet,1);  
    %% rest>bk0
    ncd=squeeze(NCD(1,:,:));
    for i=1:nNet
        [~,~,~,~, stats]=regress(ACC(:,1), [ones(nSubj,1),ncd(:,powerPart==i)]);
        stem_size(i) = stats(1);
    end
    subplot(2,3,1); bh=bar(stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45); ylim([0, Inf]); % new from Matlab2017
    ylabel(['Network AV'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    stem1=stem_size;
    for i=1:nNet
        netIdx=find(powerPart==i);
        netOther=setdiff([1:nROI], netIdx);

        [~,~,~,~, s1] = regress(ACC(:,1), [ones(nSubj,1),ncd(:,netIdx)]);
        [~,~,~,~, s2] = regress(ACC(:,1), [ones(nSubj,1),ncd(:,netOther)]);
        [~,~,~,~, s3] = regress(ACC(:,1), [ones(nSubj,1),ncd]);
        stem_size(i) = s1(1)+s2(1) - s3(1);
    end
    subplot(2,3,2); bh=bar(stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45); ylim([0, Inf]); % new from Matlab2017
    ylabel(['Network AV (Shared)'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    subplot(2,3,3); bh=bar(stem_size./stem1); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45); ylim([0, Inf]); % new from Matlab2017
    ylabel(['Network AV (Shared Ratio)'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    %% bk0>bk2
    ncd=squeeze(NCD(3,:,:));
    for i=1:nNet
        [~,~,~,~, stats]=regress(ACC(:,3), [ones(nSubj,1),ncd(:,powerPart==i)]);
        stem_size(i) = stats(1);
    end
    subplot(2,3,4); bh=bar(stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45); ylim([0, Inf]); % new from Matlab2017
    ylabel(['Network AV'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    stem1=stem_size;
    for i=1:nNet
        netIdx=find(powerPart==i);
        netOther=setdiff([1:nROI], netIdx);

        [~,~,~,~, s1] = regress(ACC(:,3), [ones(nSubj,1),ncd(:,netIdx)]);
        [~,~,~,~, s2] = regress(ACC(:,3), [ones(nSubj,1),ncd(:,netOther)]);
        [~,~,~,~, s3] = regress(ACC(:,3), [ones(nSubj,1),ncd]);
        stem_size(i) = s1(1)+s2(1) - s3(1);
    end
    subplot(2,3,5); bh=bar(stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45); ylim([0, Inf]); % new from Matlab2017
    ylabel(['Network AV (Shared)'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    subplot(2,3,6); bh=bar(stem_size./stem1); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45); ylim([0, Inf]); % new from Matlab2017
    ylabel(['Network AV (Shared Ratio)'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    
    set(gcf,'Color', 'w');export_fig([outfig 'Prediction_10Net_stemBar_AV_2contrast_MultipleRegress.tif'], '-dtiff', '-r300');
    
    
    %%
    %% The above is no ErrorBar, and After confirming, then adapt the following code, March 27, 2108
    %%
    
    stem_size=nan(nNet,1);  
    reSamp=310; % 80% of nSubj
    errBar=zeros(nNet, 2); % two column, [lower bar, uper bar]
    nSamp=5000;
    recBS=[];
    
    %% rest>bk0
    ncd=squeeze(NCD(1,:,:));
    % Nodal AV, Multiple regression
    fprintf('rest>bk0: Network AV\n');   
    for i=1:nNet
        [~,~,~,~, stats]=regress(ACC(:,1), [ones(nSubj,1),ncd(:,powerPart==i)]);
        stem_size(i)=stats(1);
    end
    tmp=zeros(nNet, nSamp);
    for J=1:nSamp % boot strapping
        if mod(J, 500)==0, fprintf('%d ', J), end
        nSel=randperm(nSubj, reSamp);
        for i=1:nNet
            [~,~,~,~, stats]=regress(ACC(nSel,1), [ones(reSamp,1),ncd(nSel,powerPart==i)]);
            tmp(i, J)=stats(1);
        end
    end
    recBS(1).stem_size=stem_size;
    recBS(1).BS = tmp;
    fprintf('\nrest>bk0: multiple regress others\n');
    [~,~,~,~, s3] = regress(ACC(:,1), [ones(nSubj,1),ncd]);
    for i=1:nNet
        netIdx=find(powerPart==i);
        netOther=setdiff([1:nROI], netIdx);
        [~,~,~,~, s1] = regress(ACC(:,1), [ones(nSubj,1),ncd(:,netIdx)]);
        [~,~,~,~, s2] = regress(ACC(:,1), [ones(nSubj,1),ncd(:,netOther)]);
        stem_size(i) = s1(1)+s2(1) - s3(1);
    end
    tmp=zeros(nNet, nSamp);
    for J=1:nSamp % boot strapping
        if mod(J, 500)==0, fprintf('%d ', J), end
        nSel=randperm(nSubj, reSamp);
        [~,~,~,~, s3] = regress(ACC(nSel,1), [ones(reSamp,1),ncd(nSel, :)]);
        for i=1:nNet
            netIdx=find(powerPart==i);
            netOther=setdiff([1:nROI], netIdx);
            [~,~,~,~, s1] = regress(ACC(nSel,1), [ones(reSamp,1),ncd(nSel,netIdx)]);
            [~,~,~,~, s2] = regress(ACC(nSel,1), [ones(reSamp,1),ncd(nSel,netOther)]);       
            tmp(i, J) = s1(1)+s2(1) - s3(1);
        end
    end
    recBS(2).stem_size=stem_size;
    recBS(2).BS = tmp;
     %% bk0>bk2
    ncd=squeeze(NCD(3,:,:));
    % Nodal AV, Multiple regression
    fprintf('\n2bk>0bk: Network AV\n');
    
    for i=1:nNet
        [~,~,~,~, stats]=regress(ACC(:,3), [ones(nSubj,1),ncd(:,powerPart==i)]);
        stem_size(i)=stats(1);
    end
    tmp=zeros(nNet, nSamp);
    for J=1:nSamp % boot strapping
        if mod(J, 500)==0, fprintf('%d ', J), end
        nSel=randperm(nSubj, reSamp);
        for i=1:nNet
            [~,~,~,~, stats]=regress(ACC(nSel,3), [ones(reSamp,1),ncd(nSel,powerPart==i)]);
            tmp(i, J)=stats(1);
        end
    end
    recBS(3).stem_size=stem_size;
    recBS(3).BS = tmp;
    fprintf('\n2bk>0bk: multiple regress others\n');
    [~,~,~,~, s3] = regress(ACC(:,3), [ones(nSubj,1),ncd]);
    for i=1:nNet
        netIdx=find(powerPart==i);
        netOther=setdiff([1:nROI], netIdx);
        [~,~,~,~, s1] = regress(ACC(:,3), [ones(nSubj,1),ncd(:,netIdx)]);
        [~,~,~,~, s2] = regress(ACC(:,3), [ones(nSubj,1),ncd(:,netOther)]);
        stem_size(i) = s1(1)+s2(1) - s3(1);
    end
    tmp=zeros(nNet, nSamp);
    for J=1:nSamp % boot strapping
        if mod(J, 500)==0, fprintf('%d ', J), end
        nSel=randperm(nSubj, reSamp);
        [~,~,~,~, s3] = regress(ACC(nSel,3), [ones(reSamp,1),ncd(nSel, :)]);
        for i=1:nNet
            netIdx=find(powerPart==i);
            netOther=setdiff([1:nROI], netIdx);
            [~,~,~,~, s1] = regress(ACC(nSel,3), [ones(reSamp,1),ncd(nSel,netIdx)]);
            [~,~,~,~, s2] = regress(ACC(nSel,3), [ones(reSamp,1),ncd(nSel,netOther)]);       
            tmp(i, J) = s1(1)+s2(1) - s3(1);
        end
    end
    recBS(4).stem_size=stem_size;
    recBS(4).BS = tmp;
    
    %save([outdat 'Prediction_10Net_stemBar_AV_2contrast_MultipleRegress_2BootStrapping.mat'], 'recBS');
    load([outdat 'Prediction_10Net_stemBar_AV_2contrast_MultipleRegress_2BootStrapping.mat'], 'recBS');
    
    compItem={'Network AV', 'Network AV (Shared)', 'Network AV', 'Network AV (Shared)'};
    figure('Position',[100,100, 900, 700]);
    for i=1:4
        stem_size = recBS(i).stem_size;  tmp = recBS(i).BS;
        errBar(:,1) = stem_size - (mean(tmp, 2) - std(tmp, 0, 2)); % bottom
        errBar(:,2) = mean(tmp, 2) + std(tmp, 0, 2) - stem_size; % top
        subplot(2,2,i); bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
%        hold on; notBoxPlot(tmp');
        set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
        xtickangle(gca, 45); ylim([0, Inf]); % new from Matlab2017
        ylabel(compItem{i}, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    end   
    
    set(gcf,'Color', 'w');export_fig([outfig 'Prediction_10Net_stemBar_AV_2contrast_MultipleRegress_2BootStrapping.tif'], '-dtiff', '-r300');
    
end

if 0
    %% to repeat the Fig. 3, to evaluate the nodal (in 10 networks) AV and network AV, stem plot (with color patch)

%    load([outdat 'corrR_Rest_WM_bk0_bk2_ACC.mat']); % 'Rest', 'WM', 'bk0', 'bk2', 'ac0', 'ac2', 'ac', 'list_adj'
%   ACC=[ac0, ac2, ac2, ac];
%   load([outdat 'NCD_4contrast.mat']); % NCD=nan(4, nSubj, nROI);
    nSubj=size(NCD,2);
   
    figure('Position',[100,100, 1600,1200]);
    ball_size=nan(nROI,1); stem_size=nan(nNet,1);  
    reSamp=310; % 80% of nSubj
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
    stem_err(3).stem = stem_size;
    stem_err(3).errBar = errBar;
    stem_err(3).tmp = tmp;
    % Network AV: first mean NCD for each network nodes, then semipart regress
    % [~, stem_size] = semipartCorr(ACC(:,1), rMat);
    %stem_size = model_PRE(ACC(:,1), rMat);
    %subplot(2,4,4), bh=bar(stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    fprintf('rest>bk0: modal selection, networkal NCD\n');
    [stem_size, errBar] = model_PRE(ACC(:,1), rMat, reSamp);
    subplot(2,4,4), bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b   
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45);  ylim([0, Inf]);% new from Matlab2017
    ylabel(['Tot. Network AV (semipart)'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    
    
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
    % Network AV: first mean NCD for each network nodes, then semipart regress
    % [~, stem_size] = semipartCorr(ACC(:,3), rMat);
    fprintf('bk0>bk2: modal selection, networkal NCD\n');
    [stem_size, errBar] = model_PRE(ACC(:,3), rMat, reSamp);
    subplot(2,4,8), bh=barwitherr(errBar, stem_size); bh.FaceColor='flat'; bh.CData=cpool(1:nNet, :);  %% work only in Matlab 2017b
    set(gca, 'xticklabel', netName,'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    xtickangle(gca, 45);  ylim([0, Inf]);% new from Matlab2017
    ylabel(['Tot. Network AV (semipart)'], 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
          
    
    %% save figure
%    set(gcf, 'PaperPositionMode', 'manual');  set(gcf, 'PaperUnits', 'points');
%    set(gcf, 'PaperPosition', [100 100 1600 1200]);
%    print(gcf, '-dtiff', '-r300', [outfig 'Prediction_10Net_stemBar_AV_4contrast.tif']);
    set(gcf,'Color', 'w');export_fig([outfig 'Prediction_10Net_stemBar_AV_4contrast.tif'], '-dtiff', '-r300');
    
end


if 0
    %% compute the shared-prediction matrix between nROI and ACC
%    load([outdat 'NCD_4contrast.mat']); % NCD=nan(4, nSubj, nROI);
    spMat=zeros(3, nROI, nROI);
%    ACC=[ac0, ac2, ac2, ac]; % 4 contrasts, rest>bk0, rest>bk2, bk0>bk2, rest>WM
    for i=1:3
        if i==2
            continue;
        end
        fprintf('\n %d \n', i);
        for j=1:nROI
            if mod(j,50)==0
                fprintf('%d ', j);
            end
            for k=j:nROI
                [r1, v1]=stdCorrMult(ACC(:,i), squeeze(NCD(i,:,[j,k])) );
                [r2, v2]=semipartCorr(ACC(:,i), squeeze(NCD(i,:,[j,k])) );
                tmp=v1 - sum(v2); %% American Prof., prove AV(Y, A)-AV(Y,A|B)=AV(Y, B)-AV(Y,B|A)
                if tmp < 0
                    tmp=0; % someone suggests to remove tmp<0
                end
                spMat(i,j,k)=tmp;
            end 
        end
        tmp=squeeze(spMat(i,:,:)); spMat(i,:,:)=tmp+(triu(tmp,1))';
    end
    spMat(2,:,:)=[];
    save([outdat 'shared_prediction_matrix_4contrast.mat'], 'spMat');
end

if 0  %% to plot the stem bar, similar to the Fig. 3, but based on spMat, to group the 10 Networks
    spNet=nan(2, nNet,nNet);
    for i=1:2
        tmp=squeeze(spMat(i,:,:));
        for j=1:nNet
            tmp2=mean(tmp(powerPart==j,:), 1);
            for k=1:nNet
                spNet(i,j,k) = mean(tmp2(powerPart==k));
            end
        end
    end

    statePair={'0-back vs. baseline',  '2-back vs. 0-back'};
    
    figure('Position', [100, 100, 1200,800]);
    for J=1:2
        nScale=20;  iPos=0; tPos=[];
        subaxis(1,2,J, 'SpacingVert', 0, 'SpacingHoriz',0.13);
        imshow(imresize(squeeze(spNet(J,:,:)),nScale, 'nearest'), []), colormap(gca, redbluecmap);  axis on; box off;
        for i=1:nNet
            iPos=iPos+nScale;
            tPos=[tPos, iPos-0.5*nScale];
        %    set(gca, 'ytick', tPos, 'yticklabel', netName{i}); hold on;
        end
        set(gca, 'TickLength',[0,0], 'ytick', tPos, 'yticklabel', netName, 'xtick', [], 'xticklabel', {''});
        set(gca,  'FontName', 'Arial', 'FontSize',14, 'FontWeight', 'Bold');
        axis on; box off;
        title(statePair{J},  'FontName', 'Arial', 'FontSize',18, 'FontWeight', 'Bold');
        colorbar; set(gca,  'FontName', 'Arial', 'FontSize',16, 'FontWeight', 'Bold');   
    end
    set(gcf,'Color', 'w');export_fig([outfig 'Mosaic_spMat_10Net_2contrast.tif'], '-dtiff', '-r300');
    
    figure('Position', [100, 100, 1200,750]);
    for J=1:2
        subaxis(1,2,J, 'SpacingVert', 0, 'SpacingHoriz',0.13);
        tmp=nan(10,2);
        tmp2=squeeze(spNet(J,:,:));
        tmp(:,1)=diag(tmp2); tmp(:,2)=mean(triu(tmp2,1)+tril(tmp2,-1), 2);
        bar(tmp);
        set(gca, 'xticklabel', netName);  xtickangle(gca, 45);
        %set(gca, 'TickLength',[0,0], 'ytick', tPos, 'yticklabel', netName, 'xtick', [], 'xticklabel', {''});
        set(gca,  'FontName', 'Arial', 'FontSize',14, 'FontWeight', 'Bold');
        axis on; box off;
        legend({'Within network', 'With other networks'}, 'Location', 'North'); legend('boxoff');
        title(statePair{J},  'FontName', 'Arial', 'FontSize',18, 'FontWeight', 'Bold');
    end
    set(gcf,'Color', 'w');export_fig([outfig 'stemBar_inter_intra_spMat_10Net_2contrast.tif'], '-dtiff', '-r300');
    
end


if   0
    %% to repeat Mosaic_spMat_10Net_2contrast.tif, by multiple regression, March 27, 2018
    %rest>bk0
    spNet=zeros(2,nNet,nNet);
    NCDIdx=[1,3];
    for i=1:2
        ncd=squeeze(NCD(NCDIdx(i),:,:));
        for J=1:nNet
            fprintf('%d ', J);
            for K=J:nNet
                ncdJ = ncd(:,powerPart==J);
                ncdK = ncd(:,powerPart==K);
                [~,~,~,~, sJ]=regress(ACC(:,NCDIdx(i)), [ones(nSubj,1), ncdJ]);
                [~,~,~,~, sK]=regress(ACC(:,NCDIdx(i)), [ones(nSubj,1), ncdK]);
                [~,~,~,~, sJK]=regress(ACC(:,NCDIdx(i)), [ones(nSubj,1), ncdJ, ncdK]);
                spNet(i,J,K)=(sJ(1)+sK(1)-sJK(1));  %/(netSize(J) + netSize(K));
            end
        end    
        fprintf('\n');
        spNet(i,:,:) = squeeze(spNet(i,:,:))  + triu(squeeze(spNet(i,:,:)), 1)';
    end
    
    spNet = spNet .* (spNet>0);
    
    statePair={'0-back vs. Baseline',  '2-back vs. 0-back'};
    figure('Position', [100, 100, 1200,500]);
    for J=1:2
        nScale=20;  iPos=0; tPos=[];
        subaxis(1,2,J, 'SpacingVert', 0, 'SpacingHoriz',0.13);
        imshow(imresize(squeeze(spNet(J,:,:)),nScale, 'nearest'), []), colormap(gca, redbluecmap);  axis on; box off;
        for i=1:nNet
            iPos=iPos+nScale;
            tPos=[tPos, iPos-0.5*nScale];
        %    set(gca, 'ytick', tPos, 'yticklabel', netName{i}); hold on;
        end
        set(gca, 'TickLength',[0,0], 'ytick', tPos, 'yticklabel', netName, 'xtick', [], 'xticklabel', {''});
        set(gca,  'FontName', 'Arial', 'FontSize',14, 'FontWeight', 'Bold');
        axis on; box off;
        title(statePair{J},  'FontName', 'Arial', 'FontSize',18, 'FontWeight', 'Bold');
        cb=colorbar; XL=get(cb, 'Title'); set(XL, 'String', 'AV', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',16);
        set(gca,  'FontName', 'Arial', 'FontSize',16, 'FontWeight', 'Bold');   
    end
    set(gcf,'Color', 'w');export_fig([outfig 'Mosaic_spMat_10Net_2contrast_MultipleRegress.tif'], '-dtiff', '-r300');
    
    figure('Position', [100, 100, 1000,600]);
    for J=1:2
        subaxis(1,2,J, 'SpacingVert', 0, 'SpacingHoriz',0.13);
        tmp=nan(10,2);
        tmp2=squeeze(spNet(J,:,:));
        tmp(:,1)=diag(tmp2); tmp(:,2)=mean(triu(tmp2,1)+tril(tmp2,-1), 2);
        bar(tmp);
        set(gca, 'xticklabel', netName);  xtickangle(gca, 45);
        %set(gca, 'TickLength',[0,0], 'ytick', tPos, 'yticklabel', netName, 'xtick', [], 'xticklabel', {''});
        set(gca,  'FontName', 'Arial', 'FontSize',14, 'FontWeight', 'Bold');
        axis on; box off;
        legend({'Within network', 'With other networks'}, 'Location', 'North'); legend('boxoff');
        title(statePair{J},  'FontName', 'Arial', 'FontSize',18, 'FontWeight', 'Bold');
    end
    set(gcf,'Color', 'w');export_fig([outfig 'stemBar_inter_intra_spMat_10Net_2contrast_MultipleRegress.tif'], '-dtiff', '-r300');
    
end









if 0
    %% Compute modular partition, both the connectivity matrix and sp matrix
    load([outdat 'shared_prediction_matrix_4contrast.mat']);  % spMat=zeros(4, nROI, nROI);
    thresh = [1 2 3 5 10]; T = length(thresh);
    gama=[0.4:0.2:1.6]; G = length(gama);

    %% before using consensus_iterative.m (Bassett), we test the mean matrix across subjects
    connMat=cat(3, squeeze(mean(Rest,1)), squeeze(mean(bk0,1)), squeeze(mean(bk2,1)), squeeze(mean(WM,1)) );
    memb_meanSubj=zeros(size(connMat,3), T,G,nROI);
    for i=1:size(connMat,3)
        fprintf('%d ', i);
        avMat=squeeze(connMat(:,:,i)); avMat=avMat .* (avMat>0);
        for j= 1:T
%            disp(thresh(j));
            thisMat = avMat .* network_thr_bin(avMat, thresh(j));
            %[memb, qual] = paco_mx(thisMat, 'quality', 2, 'nrep',100);
            for k=1:G
                [Q, tmp] = Mucha_2D(thisMat, 100, gama(k));
                memb_meanSubj(i, j,k,:) = tmp;
            end
        end
    end
    fprintf('\n');
 
    %% spMat, 4 contrast
    memb_sp=zeros(size(spMat,1), T,G,nROI);
    for i=1:size(spMat,1)
        fprintf('%d ', i);
        avMat=squeeze(spMat(i,:,:)); avMat=avMat .* (avMat>0);
        for j= 1:T
%            disp(thresh(j));
            thisMat = avMat .* network_thr_bin(avMat, thresh(j));
            %[memb, qual] = paco_mx(thisMat, 'quality', 2, 'nrep',100);
            for k=1:G
                [Q, tmp] = Mucha_2D(thisMat, 100, gama(k));
                memb_sp(i, j,k,:) = tmp;
            end
        end
    end
    fprintf('\n');
    
    save([outdat 'member_modular_Subj_mean_spMat.mat'],  'memb_meanSubj', 'memb_sp');
%    save([outdat 'member_modular_Subj_mean_spMat.mat'], 'memb_Rest', 'memb_bk0', 'memb_bk2', 'memb_WM', 'memb_meanSubj', 'memb_sp');
    
end

if 0
    %% support the above function
    %% merge the splitted results (for qsub computing results)
    load([outdat 'shared_prediction_matrix_4contrast.mat']);  % spMat=zeros(4, nROI, nROI);
    thresh = [1 2 3 5 10]; T = length(thresh);
    gama=[0.4:0.2:1.6]; G = length(gama);
    memb_meanSubj=zeros(size(spMat,1), T,G,nROI);
    memb_sp=zeros(size(spMat,1), T,G,nROI);
    for i=1:4
        tmp=load([outdat 'member_modular_Subj_mean_spMat_' num2str(i) '.mat']);
        memb_meanSubj(i,:,:,:)=tmp.memb_meanSubj(i,:,:,:);
        memb_sp(i,:,:,:)=tmp.memb_sp(i,:,:,:);
    end
    save([outdat 'member_modular_Subj_mean_spMat.mat'],  'memb_meanSubj', 'memb_sp');
end

if 0
    %% support the above function
    %% merge the splitted results (for qsub computing results)
    load([outdat 'shared_prediction_matrix_4contrast.mat']);  % spMat=zeros(4, nROI, nROI);
    thresh = [1 2 3 5 10]; T = length(thresh);
    gama=[0.4:0.2:1.6]; G = length(gama);
    
    memb_Rest=zeros(nSubj, T,G,nROI);
    memb_bk0=zeros(nSubj, T,G,nROI);
    memb_bk2=zeros(nSubj, T,G,nROI);
    memb_WM=zeros(nSubj, T,G,nROI);
    memb_meanSubj=zeros(4, T,G,nROI);
    memb_sp=zeros(4, T,G,nROI);
    for i=1:12:451
        j=i+12-1;
        if j>451  j=451; end
        tmp=load([outdat sprintf('member_modular_Subj_mean_spMat_%d_%d.mat', i,j)]);
        memb_Rest(i:j,:,:,:)=tmp.memb_Rest(i:j,:,:,:);
        memb_bk0(i:j,:,:,:)=tmp.memb_bk0(i:j,:,:,:);
        memb_bk2(i:j,:,:,:)=tmp.memb_bk2(i:j,:,:,:);
        memb_WM(i:j,:,:,:)=tmp.memb_WM(i:j,:,:,:);
    end
    load([outdat 'member_modular_Subj_mean_spMat.mat']); %  'memb_meanSubj', 'memb_sp');
    
    % save all data
    save([outdat 'member_modular_Subj_mean_spMat.mat'], 'memb_Rest', 'memb_bk0', 'memb_bk2', 'memb_WM', 'memb_meanSubj', 'memb_sp');
end


if 0
    %% Compute the concensus across subjects or across [T*G] thresholds
    load([outdat 'member_modular_Subj_mean_spMat.mat']); %'memb_Rest', 'memb_bk0', 'memb_bk2', 'memb_WM', 'memb_meanSubj', 'memb_sp');
    thresh = [1 2 3 5 10]; T = length(thresh);
    gama=[0.4:0.2:1.6]; G = length(gama);
    
%    MEM=cat(5, memb_Rest, memb_bk0, memb_bk2, memb_WM);
    %% memb_Rest
%     for i=nStart:nEnd % nStart:nEnd % across nSubj is too slow!!!!
%         n_m=nSubj*T*G;
%         memb2 = reshape(squeeze(MEM(:,:,:,:,i)), [n_m, nROI]);
%         nmi_unadj=zeros(n_m, n_m);
%         nmi_adj=zeros(n_m, n_m);
%         fprintf('\n NMI across Thresh and Gamma \n');
%         for j=1:n_m-1
%             if mod(j,300) ==0 %% very slow for unadjusted; very^10 slow for adjusted
%                 fprintf('%d ', j);
%             end
%             for k=j+1:n_m
%                 nmi_adj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'adjusted'); 
%                 nmi_unadj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'unadjusted'); 
%             end
%         end
%         nmi_unadj = nmi_unadj + nmi_unadj';
%         nmi_adj = nmi_adj + nmi_adj';
%         [~, ~, Xnew, qpc]=consensus_iterative(memb2); % we need >100 iterations to get a steady result
%         [Q, S] = Mucha_2D(Xnew, 200);
%         memCensus(i).nmi_unadj = nmi_unadj;
%         memCensus(i).nmi_adj = nmi_adj;
%         memCensus(i).S = S;
%     end

    for i=1:4
        %% memb_meanSubj
        n_m=T*G;
        memb2 = reshape(squeeze(memb_meanSubj(i,:,:,:)), [n_m, nROI]);
        nmi_unadj=zeros(n_m, n_m);
        nmi_adj=zeros(n_m, n_m);
        fprintf('\n NMI across Thresh and Gamma \n');
        for j=1:n_m-1
            if mod(j,5) ==0 %% very slow for unadjusted; very^10 slow for adjusted
                fprintf('%d ', j);
            end
            for k=j+1:n_m
                nmi_adj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'adjusted'); 
                nmi_unadj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'unadjusted'); 
            end
        end
        nmi_unadj = nmi_unadj + nmi_unadj';
        nmi_adj = nmi_adj + nmi_adj';
        [~, ~, Xnew, qpc]=consensus_iterative(memb2); % we need >100 iterations to get a steady result
        [Q, S] = Mucha_2D(Xnew, 200);
        memCensus_meanSubj(i).nmi_unadj = nmi_unadj;
        memCensus_meanSubj(i).nmi_adj = nmi_adj;
        memCensus_meanSubj(i).S = S;
        
        %% memb_sp
        n_m=T*G;
        memb2 = reshape(squeeze(memb_sp(i,:,:,:)), [n_m, nROI]);
        nmi_unadj=zeros(n_m, n_m);
        nmi_adj=zeros(n_m, n_m);
        fprintf('\n NMI across Thresh and Gamma \n');
        for j=1:n_m-1
            if mod(j,5) ==0 %% very slow for unadjusted; very^10 slow for adjusted
                fprintf('%d ', j);
            end
            for k=j+1:n_m
                nmi_adj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'adjusted'); 
                nmi_unadj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'unadjusted'); 
            end
        end
        nmi_unadj = nmi_unadj + nmi_unadj';
        nmi_adj = nmi_adj + nmi_adj';
        [~, ~, Xnew, qpc]=consensus_iterative(memb2); % we need >100 iterations to get a steady result
        [Q, S] = Mucha_2D(Xnew, 200);
        memCensus_sp(i).nmi_unadj = nmi_unadj;
        memCensus_sp(i).nmi_adj = nmi_adj;
        memCensus_sp(i).S = S;
    
    end
    
    % save data
    save([outdat 'member_concensus_acrossSubject_mean_sp.mat'], 'memCensus_meanSubj', 'memCensus_sp');
    
end
%%support above function, to merge the splitted results
if 0
    thresh = [1 2 3 5 10]; T = length(thresh);
    gama=[0.4:0.2:1.6]; G = length(gama);
    %memCensus_meanSubj:nmi_unadj, nmi_adj, S
    %memCensus_sp
    for i=1:4
        tmp=load([outdat 'member_concensus_acrossSubject_mean_sp_' num2str(i) '_rmNeg.mat']); %, 'memCensus_meanSubj', 'memCensus_sp');
        memCensus_meanSubj(i)=tmp.memCensus_meanSubj(i);
        memCensus_sp(i)=tmp.memCensus_sp(i);
    end
end



%% This is to reduce the thresh range to [1,2,3,5] by removing [10],
%% since shorter range may generate different results ??
if 0
    %% Compute the concensus across subjects or across [T*G] thresholds
    load([outdat 'member_modular_Subj_mean_spMat.mat']); %'memb_Rest', 'memb_bk0', 'memb_bk2', 'memb_WM', 'memb_meanSubj', 'memb_sp');
    thresh = [1 2 3 5]; T = length(thresh);
    gama=[0.4:0.2:1.6]; G = length(gama);
       
    for i=1:4 %nStart:nEnd % 1:4
        %% memb_meanSubj
        n_m=T*G;
        memb2 = reshape(squeeze(memb_meanSubj(i,1:end-1,:,:)), [n_m, nROI]);
        nmi_unadj=zeros(n_m, n_m);
        nmi_adj=zeros(n_m, n_m);
        fprintf('\n NMI across Thresh and Gamma \n');
        for j=1:n_m-1
            if mod(j,5) ==0 %% very slow for unadjusted; very^10 slow for adjusted
                fprintf('%d ', j);
            end
            for k=j+1:n_m
                nmi_adj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'adjusted'); 
                nmi_unadj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'unadjusted'); 
            end
        end
        nmi_unadj = nmi_unadj + nmi_unadj';
        nmi_adj = nmi_adj + nmi_adj';
        [~, ~, Xnew, qpc]=consensus_iterative(memb2); % we need >100 iterations to get a steady result
        [Q, S] = Mucha_2D(Xnew, 200);
        memCensus_meanSubj(i).nmi_unadj = nmi_unadj;
        memCensus_meanSubj(i).nmi_adj = nmi_adj;
        memCensus_meanSubj(i).S = S;
        
        %% memb_sp
        n_m=T*G;
        memb2 = reshape(squeeze(memb_sp(i,1:end-1,:,:)), [n_m, nROI]);
        nmi_unadj=zeros(n_m, n_m);
        nmi_adj=zeros(n_m, n_m);
        fprintf('\n NMI across Thresh and Gamma \n');
        for j=1:n_m-1
            if mod(j,5) ==0 %% very slow for unadjusted; very^10 slow for adjusted
                fprintf('%d ', j);
            end
            for k=j+1:n_m
                nmi_adj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'adjusted'); 
                nmi_unadj(j,k)=normalized_mutual_information(squeeze(memb2(j,:)), squeeze(memb2(k,:)), 'unadjusted'); 
            end
        end
        nmi_unadj = nmi_unadj + nmi_unadj';
        nmi_adj = nmi_adj + nmi_adj';
        [~, ~, Xnew, qpc]=consensus_iterative(memb2); % we need >100 iterations to get a steady result
        [Q, S] = Mucha_2D(Xnew, 200);
        memCensus_sp(i).nmi_unadj = nmi_unadj;
        memCensus_sp(i).nmi_adj = nmi_adj;
        memCensus_sp(i).S = S;
    
    end
    
    % save data
    save([outdat 'member_concensus_acrossSubject_mean_sp_rmT10.mat'], 'memCensus_meanSubj', 'memCensus_sp');
    
end
if 0
    %% 1. Draw the adj/unadj mosaic figure
    %% 2. export to comunity .net file for showing the ball on surface in Python code
    load([outdat 'member_concensus_acrossSubject_mean_sp.mat']); %'memCensus_meanSubj', 'memCensus_sp'

    % Adj/Unadj mosaic map: mean across subjects
    figure('Position',[100,100, 1600,800]);
    for i=1:4
        tmp = memCensus_meanSubj(i).nmi_unadj; tmp(logical(eye(size(tmp))))=1;
        subaxis(2,4,i, 'SpacingVert', 0, 'SpacingHoriz',0.02);
        imshow(imresize(tmp, 20, 'nearest'), []);  colormap(gca, redbluecmap); colorbar; 
        set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);        
        tmp = memCensus_meanSubj(i).nmi_adj; tmp(logical(eye(size(tmp))))=1;
        subaxis(2,4,i+4, 'SpacingVert', 0, 'SpacingHoriz',0.02);
        imshow(imresize(tmp, 20, 'nearest'), []);  colormap(gca, redbluecmap); colorbar; 
        set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
        
    end
    %% save figure
    set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_AdjUnadj_acrossSubject_mean.tif'], '-dtiff', '-r300');
    
    % Adj/Unadj mosaic map: spMat
    figure('Position',[100,100, 1800,800]);
    for i=1:4
        tmp = memCensus_sp(i).nmi_unadj; tmp(logical(eye(size(tmp))))=1;
        subaxis(2,4,i, 'SpacingVert', 0, 'SpacingHoriz',0.02);
        imshow(imresize(tmp, 20, 'nearest'), []);  colormap(gca, redbluecmap); colorbar; 
        set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
        tmp = memCensus_sp(i).nmi_adj; tmp(logical(eye(size(tmp))))=1;
        subaxis(2,4,i+4, 'SpacingVert', 0, 'SpacingHoriz', 0.02);
        imshow(imresize(tmp, 20, 'nearest'), []);  colormap(gca, redbluecmap); colorbar; 
        set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',14);
    end  
    
    %% save figure
    set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_AdjUnadj_spMat.tif'], '-dtiff', '-r300');
end

if 0
    %% 2. export to comunity .net file for showing the ball on surface in Python code
    load([outdat 'member_concensus_acrossSubject_mean_sp.mat']); %'memCensus_meanSubj', 'memCensus_sp'
    statName={'Rest', 'bk0', 'bk2', 'WM'};
    pairName={'Rest_bk0', 'Rest_bk2', 'bk0_bk2', 'Rest_WM'};
    for i=1:4
        export_ModMemb_netfile(memCensus_meanSubj(i).S,  [outfig 'member_concensus_acrossSubject_mean_' statName{i} '.net']);
        export_ModMemb_netfile(memCensus_sp(i).S,  [outfig 'member_concensus_spMat_' pairName{i} '.net']);
    end
end

if 0  % Matlab 2017
    %% 2. export to comunity .net file for showing the ball on surface in Python code
    mypos='_rmT10';
    mypos=[];
    load([outdat 'member_concensus_acrossSubject_mean_sp' mypos '.mat']); %'memCensus_meanSubj', 'memCensus_sp'
   %% spMat
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(memCensus_sp(1).S, roiCoord, [], ones(nROI,1)*5); suptitle('REST1 vs. bk0');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_spMat_REST1_bk0' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(memCensus_sp(2).S, roiCoord, [], ones(nROI,1)*5); suptitle('REST1 vs. bk2');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_spMat_REST1_bk2' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(memCensus_sp(3).S, roiCoord, [], ones(nROI,1)*5); suptitle('bk0 vs. bk2');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_spMat_bk0_bk2' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(memCensus_sp(4).S, roiCoord, [], ones(nROI,1)*5); suptitle('REST1 vs. WM');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_spMat_REST1_WM' mypos '.tif'], '-dtiff', '-r300');
        
    %% meanSubj
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(memCensus_meanSubj(1).S, roiCoord, [], ones(nROI,1)*5); suptitle('REST1');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_acrossSubject_mean_REST1' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(memCensus_meanSubj(2).S, roiCoord, [], ones(nROI,1)*5); suptitle('bk0');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_acrossSubject_mean_bk0' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(memCensus_meanSubj(3).S, roiCoord, [], ones(nROI,1)*5); suptitle('bk2');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_acrossSubject_mean_bk2' mypos '.tif'], '-dtiff', '-r300');
    figure('Position',[100,100, 800,700]), show_surface_roi_ModAssign(memCensus_meanSubj(4).S, roiCoord, [], ones(nROI,1)*5); suptitle('WM');
        set(gcf,'Color', 'w');export_fig([outfig 'member_concensus_acrossSubject_mean_WM' mypos '.tif'], '-dtiff', '-r300');
end


if 0
    %% Statistics of the overlap for each community to each of 10 networks
    %% print the results as a table
    load([outdat 'member_concensus_acrossSubject_mean_sp.mat']); %'memCensus_meanSubj', 'memCensus_sp'
    memC=memCensus_sp(4).S;
    load([outdat 'NCD_4contrast.mat']); % NCD=nan(4, nSubj, nROI);
%    community_overlap_stat(memCensus_sp(4).S, ac, squeeze(NCD(4,:,:)));
    fprintf('\n\n');
    
    nNet=10;
    net10Ind=find(powerPart<=nNet);
    myrec=ac;
    NCD=squeeze(NCD(4,:,:));
    nROI=size(NCD,1);
    
    Thresh=[5:5:50]*0.01; % percent of the entire nodes
    fid = fopen([outfig 'Prediction_eachROI_ball_surface_rest_WM.net'], 'r');
    indR = textscan(fid, '%d %d %d %f %f %f %f', 'delimiter', '\n', 'HeaderLines', 1);
    fclose(fid);
    indR4=indR{4};
    [~, sortInd]=sort(indR4, 'descend');
    
    [nFreq, nBin]= hist(memC, [1:nROI]);
    nComm=2; % only stat the first nComm communities
    [~, sortF]=sort(nFreq, 'descend');
    iComm=nBin(sortF(1:nComm)); % get the first nComm label
    for i=1:nComm
        commPool{i}=find(memC==iComm(i));
    end
    
    for i=1:length(Thresh) 
        nCount=round(Thresh(i)*nROI);
        topNode=sortInd(1:nCount);
        [~,~,~,~,stats]=regress(ac, [ones(size(NCD,1),1), NCD(:, topNode)]);
        fprintf('%d%%(%0.2f%%)\t', Thresh(i)*100, stats(1)*100);
        tmp=zeros(nComm,1);
        for j=1:nComm
            intersectNode=intersect(topNode, commPool{j});
            tmp(j)=length(intersectNode)/length(topNode);
            fprintf('%0.2f\t', tmp(j)*100);
        end
        fprintf('%0.2f\t', sum(tmp)*100);
        fprintf('\n');
    end
    


end



end


function community_overlap_stat(memb, myrec,NCD, thresh, gama)
    netName={'Sensory', 'CON', 'Auditory','DMN', 'VAN', 'Visual', ...
             'FPN', 'Salience', 'DAN', 'Subcort.', 'Memory','Cerebellar', 'Uncertain'};   
    nROI = 264;
    if ~exist('thresh', 'var')
        thresh=10;
    end
    if ~exist('gama', 'var')
        gama=1.0;
    end
    
    Sensory=[13:46,255]; % 35
    Cingulo=[47:60]; % 14
    Auditory=[61:73]; % 13
    DMN=[74:83,86:131,137,139]; % 58
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
    powerPart(DMN)=4;
    powerPart(Vatt)=5;
    powerPart(Visual)=6;
    powerPart(FPC)=7;
    powerPart(Salience)=8;
    powerPart(Datt)=9;

    powerPart(Subcort)=10;
    powerPart(Memory)=11;
    powerPart(Cerebellar)=12;
    powerPart(Uncertain)=13;
    
    nNet=10;
    
    nComm=10; % the first 10 communities;
    nSize=5; % the smallest community
    %nBig=zeros(nComm,2); % 2=[roi index, comm size]
    
    s0=sprintf('thresh=%d; gamma=%0.1f\n', thresh,gama);
    s_all=[];
    sn=[];
    %sn=sprintf('%s', '\n', 'interpreter', 'latex');
    
    [nCount, nBin]=hist(memb, max(memb));
    [cSort, cInd]=sort(nCount, 'descend');
    for i=1:nComm
        if cSort(i)>nSize
            tmpInd=find(memb==cInd(i));
            [~,~,~,~,stats]=regress(myrec, [ones(size(myrec,1),1) NCD(:,tmpInd)]);
            comm_size=length(tmpInd);
            fprintf('Comm %d; Size: %d (AV:%0.2f%%)\n', i, comm_size, stats(1)*100);
            
            s3=[];
            for j=1:nNet
                thisNet=find(powerPart==j);
                [C, jF,jL]=intersect(tmpInd, thisNet); %% CAUTION: index of index
                if ~isempty(C)
                    [~,~,~,~,stats]=regress(myrec, [ones(size(myrec,1),1) NCD(:,thisNet(jL))]);
                    fprintf('%s(%0.2f%%): %0.2f%%(AV:%0.2f%%)\t', netName{j}, length(C)/length(thisNet)*100, length(C)/comm_size*100, stats(1)*100);
                end
            end
            fprintf('\n');
        end
    end
    %title(sprintf('%s%s',s0, s_all ));
    %title(s0);

end


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
        tmp=zeros(M, 5000);
        for i=1:5000
            nSel=randperm(nLen, reSamp);
            tmp(:,i) = model_PRE(Y(nSel), X(nSel, :));
        end
        errBar(:,1)= PRE - ( -std(tmp,0, 2) + mean(tmp, 2));
        errBar(:,2)=std(tmp,0, 2) + mean(tmp, 2) - PRE;
    end
end


function [R, R2] = stdCorrMult(Y, X)  %% Y=[n,1]; X=[n,m] or [n,m,k]
% https://stats.stackexchange.com/questions/100211/why-is-my-shared-variance-negative
% Tabachnick, B. G., & Fidell, L. S. (2001). Using multivariate statistics (4th ed.). Boston: Allyn and Bacon.
    [nLen, M,N]=size(X);
    if N==1 % dim=2
       [b, bint, r, rint, stats] = regress(Y, [ones(nLen,1),X]);
       R2 = stats(1);
       R = corr(Y, Y-r);
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

function z = fisherz(r)
    z=0.5 * log((1+r)./(1-r));
end



