function show_surface_roi_ModAssign(memb, roiCoord, roiColor, roiWeit, Factor)
    %% This is calling show_surface_roi(roiCoord, roiColor, roiWeit, Factor)
    
    if ~exist('roiCoord', 'var')
        roiCoord=load('/datc/flex_8state/code/roi400_Yeo.txt');
    end
    cpool=[1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; 0.5 0.5 0; 0.5 0 0.5; ...
           0 0.5 0.5; 0.5 0.5 0.5; 1 0.5 0; 1 0 0.5; 0 0.5 1];
    nComm=13; % the first 8 communities;
    nSize=10;
    
    if ~isempty(roiColor)
        roiColor2=roiColor;
        colorSet = 1;
    else
        colorSet = 0;
    end

    roiColor=zeros(size(roiCoord));
    [nCount, nBin]=hist(memb, max(memb));
    [cSort, cInd]=sort(nCount, 'descend');
    indPool=[];
    for i=1:nComm
        if i <= length(cSort)
            if cSort(i)>nSize
                tmpInd=find(memb==cInd(i));
                roiColor(tmpInd,:) = repmat(cpool(i,:), [length(tmpInd),1]);
                indPool=[indPool; tmpInd];
            end
        end
    end

    if ~exist('Factor', 'var')
        Factor = 1;
    end
    if ~exist('roiWeit', 'var')
        roiWeit=ones(size(roiCoord,1),1);
    end
    if colorSet
        show_surface_roi(roiCoord, roiColor2,  roiWeit, Factor);
    else
        show_surface_roi(roiCoord(indPool,:), roiColor(indPool,:),  roiWeit(indPool,:), Factor);
    end

end