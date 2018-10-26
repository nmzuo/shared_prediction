function show_surface_Activation_prediction

outdat='/datc/flex_8state/data_extend_Power264AdjACC_Act/';
outfig='/datc/flex_8state/fig_extend_Power264AdjACC_Act/';
p_dyn='/datc/dynNet/code/';
p_flex='/datc/flex/';
addpath(genpath(p_dyn));
addpath(genpath([p_flex 'code']));
addpath(genpath('./'));
addpath(genpath('/datc/software/matlabTools/'));
%addpath(genpath('/datc/software/brat/ccm')); %% compute clustering coeficient

load(['/datc/flex_8state/data_extend_Power264AdjACC/corrR_Rest_WM_bk0_bk2_ACC.mat']); % 'Rest', 'WM', 'bk0', 'bk2', 'ac0', 'ac2', 'ac', 'list_adj'
nSubj=size(Rest,1);

addpath('/datc/software/SurfStat');
addpath('/datc/software/NIfTI_20140122');

%% Activation mapped to Surface
%temp='/datc/software/spm12/toolbox/cat12/templates_surfaces_32k/';
%avsurf=SurfStatReadSurf({[temp 'lh.central.Template_T1_IXI555_MNI152_GS.gii'], [temp 'rh.central.Template_T1_IXI555_MNI152_GS.gii']});
%avsurfSM = SurfStatInflate_zuo(avsurf, 0.2, {[temp 'lh.sphere.freesurfer.gii'], [temp 'rh.sphere.freesurfer.gii']});
temp='/datc/software/BrainNetViewer_20170403/Data/SurfTemplate/';
avsurf=SurfStatReadSurf([temp 'BrainMesh_ICBM152.gii']);
avsurfSM=SurfStatReadSurf([temp 'BrainMesh_ICBM152_smooth.gii']);

MASK=load_nii('/datc/Cam-CAN/code/aal_2mm_fsl_mask.nii.gz'); 
[nx,ny,nz]=size(MASK.img);
MASK=find(MASK.img>0.1); LEN=length(MASK);

%% Activation on the Surface
if  0
    % cope3: 0-back vs. baseline
    c3=load_nii('/datc/dynNet/groupActi_avg/tfMRI_WM_avg/groupmean.gfeat/cope3.feat/stats/zstat1.nii.gz');
    vol.data=c3.img; vol.origin=[-90, -126, -72]; vol.vox=[2,2,2];
    mapS = SurfStatVol2Surf(vol, avsurf);
    minNR = min(mapS(mapS<0)); maxNR = max(mapS(mapS<0)); 
    minPR = min(mapS(mapS>0)); maxPR = max(mapS(mapS>0)); 
    R1=vol.data;
    figure, SurfStatView_2pieceColor(mapS, avsurfSM, 'Z (Activation)', [minNR, -2.32, 2.32, maxPR],  ...
         [min(R1(R1<0)), -2.32, 2.32, max(R1(R1>0))], ['0-back vs. Baseline']);
    set(gcf,'Color', 'w');export_fig([outfig 'Activation_cope3_0bk-baseline.tif'], '-dtiff', '-r300'); 
    % cope1: 2-back vs. 1-back
    c1=load_nii('/datc/dynNet/groupActi_avg/tfMRI_WM_avg/groupmean.gfeat/cope1.feat/stats/zstat1.nii.gz');
    vol.data=c1.img; vol.origin=[-90, -126, -72]; vol.vox=[2,2,2];
    mapS = SurfStatVol2Surf(vol, avsurf);
    minNR = min(mapS(mapS<0)); maxNR = max(mapS(mapS<0)); 
    minPR = min(mapS(mapS>0)); maxPR = max(mapS(mapS>0)); 
    R1=vol.data;
    figure, SurfStatView_2pieceColor(mapS, avsurfSM, 'Z (Activation)', [minNR, -2.32, 2.32, maxPR],  ...
         [min(R1(R1<0)), -2.32, 2.32, max(R1(R1>0))], ['2-back vs. 0-back']);
    set(gcf,'Color', 'w');export_fig([outfig 'Activation_cope1_2bk-0bk.tif'], '-dtiff', '-r300'); 
end

%% Voxel-wise prediction along the Surface
if  0
    
    ACT3=zeros(LEN, nSubj);
    ACT1=zeros(LEN, nSubj);
    for i=[] %1:nSubj
        if mod(i, 20)==0, fprintf('%d ', i); end
        % cope3: 0-back vs. baseline
        thispath=['/datc/DATA239/hcp_S500_defil/' num2str(list_adj(i,1)) '/MNINonLinear/Results/tfMRI_WM_LR/hp200_s4_level1/level1.feat/stats/zstat3.nii.gz'];
        thismap=load_nii(thispath); thismap=thismap.img(MASK);
        tmp=['/datc/DATA239/hcp_S500_defil/' num2str(list_adj(i,1)) '/MNINonLinear/Results/tfMRI_WM_RL/hp200_s4_level1/level1.feat/stats/zstat3.nii.gz'];
        tmp=load_nii(tmp); thismap=(tmp.img(MASK) + thismap)*0.5;
        ACT3(:,i) = thismap;
        clear tmp thismap
        
        % cope1: 2-back vs. 1-back
        thispath=['/datc/DATA239/hcp_S500_defil/' num2str(list_adj(i,1)) '/MNINonLinear/Results/tfMRI_WM_LR/hp200_s4_level1/level1.feat/stats/zstat1.nii.gz'];
        thismap=load_nii(thispath); thismap=thismap.img(MASK);
        tmp=['/datc/DATA239/hcp_S500_defil/' num2str(list_adj(i,1)) '/MNINonLinear/Results/tfMRI_WM_RL/hp200_s4_level1/level1.feat/stats/zstat1.nii.gz'];
        tmp=load_nii(tmp); thismap=(tmp.img(MASK) + thismap)*0.5;
        ACT1(:,i) = thismap;
        clear tmp thismap
    end
    fprintf('\n');
    %save(['/datc/dynNet/groupActi_avg/tfMRI_WM_avg/ActivationMap_WM_cope3_cope1_385Subject.mat'], 'ACT3', 'ACT1');
    load(['/datc/dynNet/groupActi_avg/tfMRI_WM_avg/ActivationMap_WM_cope3_cope1_385Subject.mat'], 'ACT3', 'ACT1');
    
    [r3, p3]=corr(ACT3', ac0);
    [r1, p1]=corr(ACT1', ac2);
%    r3(p3>0.05)=0; r1(p1>0.05)=0; 
    vol.data=zeros([nx,ny,nz]); vol.data(MASK)=r3; vol.origin=[-90, -126, -72]; vol.vox=[2,2,2];
    mapS = SurfStatVol2Surf(vol, avsurf);
    minNR = min(mapS(mapS<0)); maxNR = max(mapS(mapS<0)); 
    minPR = min(mapS(mapS>0)); maxPR = max(mapS(mapS>0)); 
    figure, [ab, cb]=SurfStatView(mapS, avsurfSM, 'r: corr(Activation, Accuracy)');
    XL=get(cb, 'YLabel'); set(XL, 'String', '0-back vs. Baseline', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
    set(cb,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
    set(gcf,'Color', 'w');export_fig([outfig 'Correlation_Activation_ACC0_cope3_0bk-baseline.tif'], '-dtiff', '-r300'); 
    
    vol.data=zeros([nx,ny,nz]); vol.data(MASK)=r1; vol.origin=[-90, -126, -72]; vol.vox=[2,2,2];
    mapS = SurfStatVol2Surf(vol, avsurf);
    minNR = min(mapS(mapS<0)); maxNR = max(mapS(mapS<0)); 
    minPR = min(mapS(mapS>0)); maxPR = max(mapS(mapS>0)); 
    figure, [ab, cb]=SurfStatView(mapS, avsurfSM, 'r: corr(Activation, Accuracy)');
    XL=get(cb, 'YLabel'); set(XL, 'String', '2-back vs. 0-back', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
    set(cb,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
    set(gcf,'Color', 'w');export_fig([outfig 'Correlation_Activation_ACC2_cope1_2bk-0bk.tif'], '-dtiff', '-r300'); 
end
%% Voxel-wise prediction along the Surface: but after threshold (p<0.05), 2piece display
if  0
    
    ACT3=zeros(LEN, nSubj);
    ACT1=zeros(LEN, nSubj);

    load(['/datc/dynNet/groupActi_avg/tfMRI_WM_avg/ActivationMap_WM_cope3_cope1_385Subject.mat'], 'ACT3', 'ACT1');
    
    [r3, p3]=corr(ACT3', ac0);
    [r1, p1]=corr(ACT1', ac2);
    r3(p3>0.05)=0; r1(p1>0.05)=0; 
    vol.data=zeros([nx,ny,nz]); vol.data(MASK)=r3; vol.origin=[-90, -126, -72]; vol.vox=[2,2,2];
    mapS = SurfStatVol2Surf(vol, avsurf);
    minNR = min(mapS(mapS<0)); maxNR = max(mapS(mapS<0)); 
    minPR = min(mapS(mapS>0)); maxPR = max(mapS(mapS>0)); 
    %figure, [ab, cb]=SurfStatView(mapS, avsurfSM, 'r: corr(Activation, Accuracy)');
    figure, SurfStatView_2pieceColor(mapS, avsurfSM, 'r: corr(Activation, Accuracy)', [minNR, maxNR, minPR, maxPR],  ...
         [min(r3), max(r3(r3<0)), min(r3(r3>0)), max(r3)], ['0-back vs. Baseline']);
%    XL=get(cb, 'YLabel'); set(XL, 'String', '0-back vs. Baseline', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
%    set(cb,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
    set(gcf,'Color', 'w');export_fig([outfig 'Correlation_Activation_ACC0_cope3_0bk-baseline_p0.05.tif'], '-dtiff', '-r300'); 
    
    vol.data=zeros([nx,ny,nz]); vol.data(MASK)=r1; vol.origin=[-90, -126, -72]; vol.vox=[2,2,2];
    mapS = SurfStatVol2Surf(vol, avsurf);
    minNR = min(mapS(mapS<0)); maxNR = max(mapS(mapS<0)); 
    minPR = min(mapS(mapS>0)); maxPR = max(mapS(mapS>0)); 
    %figure, [ab, cb]=SurfStatView(mapS, avsurfSM, 'R: corr(Activation, Accuracy)');
    figure, SurfStatView_2pieceColor(mapS, avsurfSM, 'r: corr(Activation, Accuracy)', [minNR, maxNR, minPR, maxPR],  ...
         [min(r3), max(r3(r3<0)), min(r3(r3>0)), max(r3)], ['2-back vs. 0-back']);
%    XL=get(cb, 'YLabel'); set(XL, 'String', '2-back vs. 0-back', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
%    set(cb,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
    set(gcf,'Color', 'w');export_fig([outfig 'Correlation_Activation_ACC2_cope1_2bk-0bk_p0.05.tif'], '-dtiff', '-r300'); 
end

%% Overlap between Activation and Prediction
if   1
    load(['/datc/dynNet/groupActi_avg/tfMRI_WM_avg/ActivationMap_WM_cope3_cope1_385Subject.mat'], 'ACT3', 'ACT1');
    
    [r3, p3]=corr(ACT3', ac0);
    [r1, p1]=corr(ACT1', ac2);
    tmp=zeros(nx,ny,nz);  tmp(MASK)=r3; r3=tmp; tmp(MASK)=r1; r1=tmp;
    tmp=ones(nx,ny,nz);  tmp(MASK)=p3; p3=tmp; tmp(MASK)=p1; p1=tmp;
    clear tmp;
    c3=load_nii('/datc/dynNet/groupActi_avg/tfMRI_WM_avg/groupmean.gfeat/cope3.feat/stats/zstat1.nii.gz'); c3=c3.img;
    c1=load_nii('/datc/dynNet/groupActi_avg/tfMRI_WM_avg/groupmean.gfeat/cope1.feat/stats/zstat1.nii.gz'); c1=c1.img;

    %% 0-back vs. Baseline
    noThr=zeros(nx,ny,nz); 
    noThr(r3>0 & c3>0)=2; noThr(r3<0 & c3<0)=1;
    vol.data=noThr; vol.origin=[-90, -126, -72]; vol.vox=[2,2,2];
    mapS = SurfStatVol2Surf(vol, avsurf);
    figure, [ab, cb]=SurfStatView(mapS, avsurfSM, 'Overlap (Activation vs. Prediction)');
    SurfStatColormap([0.95 0.95 0.95; 0 0 1; 1 0 0]);
    set(cb,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',8);
    XL=get(cb, 'YLabel'); set(XL, 'String', '0-back vs. Baseline', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
    set(cb, 'YTick',[0.5, 1, 1.7], 'YTickLabel', {'', 'Neg.', 'Pos.'});
    set(gcf,'Color', 'w');export_fig([outfig 'Overlap_Correlation_Activation_ACC0_cope3_0bk-baseline.tif'], '-dtiff', '-r300'); 
    posSize=length(find(r3>0 | c3>0)); negSize=length(find(r3<0 | c3<0)); 
    fprintf('Overlap: %f (Pos.), %f (Neg.)\n', length(find(noThr(MASK)>1.5))/posSize,  length(find(noThr(MASK)<1.5 & noThr(MASK)>0.5 ))/negSize );
    
    withThr=zeros(nx,ny,nz); 
    withThr(r3>0 & p3<0.05 & c3>2.32)=2;  withThr(r3<0 & p3<0.05 & c3<-2.32)=1; 
    vol.data=withThr; vol.origin=[-90, -126, -72]; vol.vox=[2,2,2];
    mapS = SurfStatVol2Surf(vol, avsurf);
    figure, [ab, cb]=SurfStatView(mapS, avsurfSM, 'Overlap (Sig., Activation vs. Prediction)');
    SurfStatColormap([0.95 0.95 0.95; 0 0 1; 1 0 0]);
    set(cb,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',8);
    XL=get(cb, 'YLabel'); set(XL, 'String', '0-back vs. Baseline', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
    set(cb, 'YTick',[0.5, 1, 1.7], 'YTickLabel', {'', 'Neg.', 'Pos.'});
    set(gcf,'Color', 'w');export_fig([outfig 'Overlap_Correlation_Activation_ACC0_cope3_0bk-baseline_Significant.tif'], '-dtiff', '-r300'); 
    posSize=length(find((r3>0 & p3<0.05) | c3>2.32)); negSize=length(find((r3<0 & p3<0.05) | c3<-2.32)); 
    fprintf('Overlap: %f (Pos.), %f (Neg.)\n', length(find(withThr(MASK)>1.5))/posSize,  length(find(withThr(MASK)<1.5 & withThr(MASK)>0.5 ))/negSize );
    
     %% 2-back vs. 0-back
    noThr=zeros(nx,ny,nz); 
    noThr(r1>0 & c1>0)=2; noThr(r1<0 & c1<0)=1;
    vol.data=noThr; vol.origin=[-90, -126, -72]; vol.vox=[2,2,2];
    mapS = SurfStatVol2Surf(vol, avsurf);
    figure, [ab, cb]=SurfStatView(mapS, avsurfSM, 'Overlap (Activation vs. Prediction)');
    SurfStatColormap([0.95 0.95 0.95; 0 0 1; 1 0 0]);
    set(cb,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',8);
    XL=get(cb, 'YLabel'); set(XL, 'String', '2-back vs. 0-back', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
    set(cb, 'YTick',[0.5, 1, 1.7], 'YTickLabel', {'', 'Neg.', 'Pos.'});
    set(gcf,'Color', 'w');export_fig([outfig 'Overlap_Correlation_Activation_ACC2_cope1_2bk-0bk.tif'], '-dtiff', '-r300'); 
    posSize=length(find(r1>0 | c1>0)); negSize=length(find(r1<0 | c1<0)); 
    fprintf('Overlap: %f (Pos.), %f (Neg.)\n', length(find(noThr(MASK)>1.5))/posSize,  length(find(noThr(MASK)<1.5 & noThr(MASK)>0.5 ))/negSize );
    
    withThr=zeros(nx,ny,nz); 
    withThr(r1>0 & p1<0.05 & c1>2.32)=2;  withThr(r1<0 & p1<0.05 & c1<-2.32)=1; 
    vol.data=withThr; vol.origin=[-90, -126, -72]; vol.vox=[2,2,2];
    mapS = SurfStatVol2Surf(vol, avsurf);
    figure, [ab, cb]=SurfStatView(mapS, avsurfSM, 'Overlap (Sig., Activation vs. Prediction)');
    SurfStatColormap([0.95 0.95 0.95; 0 0 1; 1 0 0]);
    set(cb,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',8);
    XL=get(cb, 'YLabel'); set(XL, 'String', '2-back vs. 0-back', 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',10);
    set(cb, 'YTick',[0.5, 1, 1.7], 'YTickLabel', {'', 'Neg.', 'Pos.'});
    set(gcf,'Color', 'w');export_fig([outfig 'Overlap_Correlation_Activation_ACC2_cope1_2bk-0bk_Significant.tif'], '-dtiff', '-r300'); 
    posSize=length(find((r1>0 & p1<0.05) | c1>2.32)); negSize=length(find((r1<0 & p1<0.05) | c1<-2.32)); 
    fprintf('Overlap: %f (Pos.), %f (Neg.)\n', length(find(withThr(MASK)>1.5))/posSize,  length(find(withThr(MASK)<1.5 & withThr(MASK)>0.5 ))/negSize );
    
end

end



