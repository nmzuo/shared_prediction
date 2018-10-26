function ROI_convert_Coord2Region
%% This is to convert the region-based ROI definitions to central coordinates based
%% namely, to extract the central coordinates for each ROI

addpath('/datc/dynNet/code/'); %% xyz2ijk_MNI2mm.m

Yeo=load_untouch_nii('/datc/flex_8state/code/BN_Atlas_246_3mm.nii.gz');
dx=Yeo.hdr.dime.pixdim(2); dy=Yeo.hdr.dime.pixdim(3); dz=Yeo.hdr.dime.pixdim(4);
NX=Yeo.hdr.dime.dim(2); NY=Yeo.hdr.dime.dim(3); NZ=Yeo.hdr.dime.dim(4);
Rad = 5; % 5mm
xoff=round(Rad/dx); yoff=round(Rad/dy); zoff=round(Rad/dz); 
Yimg=Yeo.img;

YeoCoord=load('/datc/flex_8state/code/BN_Atlas_246_coord.txt');
nROI=size(YeoCoord,1);
roiImg=zeros(size(Yimg));

for i=1:nROI
    iCount=0;
    coord=xyz2ijk_MNI2mm(YeoCoord(i,:), 1, '/datc/flex_8state/code/BN_Atlas_246_3mm.nii.gz') + 1;
    for ix=-xoff:xoff
        for iy=-yoff:yoff
            for iz=-zoff:zoff
                if (ix*dx)^2 + (iy*dy)^2 + (iz*dz)^2 <= Rad*Rad
                    thisCoord=coord+[ix,iy,iz];
                    if thisCoord(1)>0 && thisCoord(1)<=NX && thisCoord(2)>0 && thisCoord(2)<=NY && thisCoord(3)>0 && thisCoord(3)<=NZ
                        roiImg(thisCoord(1), thisCoord(2),thisCoord(3))=i;
                        iCount=iCount+1;
                    end
                end
            end
        end
    end
end

Yeo.img = roiImg;
save_untouch_nii(Yeo, '/datc/flex_8state/code/BN_Atlas_246_3mm_ball5mm.nii.gz');

end