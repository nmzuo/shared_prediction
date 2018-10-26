function ROI_convert_Region2Coord
%% This is to convert the region-based ROI definitions to central coordinates based
%% namely, to extract the central coordinates for each ROI

addpath('/datc/dynNet/code/'); %% xyz2ijk_MNI2mm.m

Yeo=load_untouch_nii('Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz');
Yimg=Yeo.img;
nROI=length(unique(Yimg))-1; % remove 0

YeoCoord=nan(nROI, 3);

for i=1:nROI
    IJK=find(Yimg==i);
    [I,J,K]=ind2sub(size(Yimg), IJK);
    I=round(mean(I)); J=round(mean(J)); K=round(mean(K)); 
    YeoCoord(i,:)=[I,J,K] -1;
end

YeoCoord = xyz2ijk_MNI2mm(YeoCoord, 2);
YeoCoord=round(YeoCoord);

%save('roi400_Yeo.txt', 'YeoCoord', '-ASCII')
dlmwrite('roi200_Yeo.txt', YeoCoord, 'delimiter', '\t', 'precision', '%3d');

end