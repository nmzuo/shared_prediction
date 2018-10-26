function show_surface_roi(roiCoord, roiColor, roiWeit, Factor)

addpath(genpath('/datc/software/SurfStat'));

%% adjust the coord for show
pAdj=abs(roiCoord(:,1))<5.5; nAdj=abs(roiCoord(:,1))>6.5;
roiCoord(pAdj,1)=roiCoord(pAdj,1)*0.8;
roiCoord(nAdj,1)=roiCoord(nAdj,1)*1.1;

mysurf='surf_smoothed.txt';
%mysurf='surf_standard.txt';
fin=fopen(mysurf, 'r');
nVert=fgetl(fin); nVert=str2num(nVert);
%vertMat=nan(nVert, 3);
[verMat, pos]=textscan(fin, '%f %f %f', nVert); %'headerlines', 1
verMat=[verMat{1}(:), verMat{2}(:), verMat{3}(:)];
%% shrink the surface for no hiding ROI balls
%verMat = verMat * 0.8;
nEdge=fgetl(fin); nEdge=fgetl(fin); nEdge=fgetl(fin); 
nEdge=str2num(nEdge);
edgeMat=nan(nEdge,3);
[edgeMat, pos]=textscan(fin, '%d %d %d', nEdge);
edgeMat=[edgeMat{1}(:), edgeMat{2}(:),edgeMat{3}(:),];

% construct the structure for SurfStat
bksurf.tri=edgeMat; bksurf.coord=verMat;
orgSurf=patch('Faces',bksurf.tri,'Vertices',bksurf.coord);
cutSurf=reducepatch(orgSurf, 0.5);
clf(gcf); % patch will generate rubbish

clear edgeMat verMat ;

%% call SurfStat
t=size(cutSurf.faces,1);
v=size(cutSurf.vertices,1);
cut=t/2;
cuv=v/2;

tl=1:cut;
tr=(cut+1):t;
vl=1:cuv;
vr=(cuv+1):v;

h=0.5;
w=0.5;
    
faceColor=[0.95 0.95 0.95];
faceAlpha=1; % Fast in Matlab 2017 but VERY slow in 2012
edgeAlpha=0.6;

% roiCoord=[
%   -2  -37 44
%   11  -54 17
%   52  -59 36
%   23  33  48
%   -10 39  52
%  -16 29  53 ];
% roiWeit=rand(6,1);
if ~exist('Factor', 'var')
    Factor = 5;
end
if ~exist('roiWeit', 'var')
    roiWeit=ones(size(roiCoord,1),1);
end
roiWeit = roiWeit * Factor;
if ~exist('roiColor', 'var')
    roiColor=repmat([1,0,0], [size(roiCoord,1),1]);
end

load('/datc/dynNet/code/redbluecolormap.mat'); % redblue=[64,3]
% minC=1; maxC=64;
% if any(roiWeit)
%     Weit=(roiWeit-min(roiWeit))/(max(roiWeit)-min(roiWeit)) * (maxC-minC) + minC;
%     roiColor=interp1([1:64], redblue, Weit);
% end % roiColor =[n,3]


[x,y,z]=sphere(100);

%% figure,
%% Left lateral
a(1)=axes('position',[0.06 0.5 h  w]);
%subplot(2,2,1);
hh=trisurf(cutSurf.faces(tl,:),cutSurf.vertices(vl,1),cutSurf.vertices(vl,2),cutSurf.vertices(vl,3),...
    double(ones(size(vl))),'EdgeColor','none');
load('/datc/flex_8state/code/surf_color_CM.mat');
colormap(CM);
%set(hh, 'FaceAlpha', faceAlpha);  % Fast in Matlab 2017 but VERY slow in 2012
view(-90,0); 
daspect([1 1 1]); 
whitebg(gcf, [1 1 1]);
set(gcf,'Color',[1 1 1],'InvertHardcopy','off');
lighting phong; material([0.5 0.5 0.5]); shading interp; 
set(hh, 'FaceColor', faceColor, 'FaceAlpha', faceAlpha);
axis tight; axis vis3d off;
camlight('right'); 

hold on;
for i=1:size(roiCoord,1)
    if roiCoord(i,1)<2
        hh=surf(x*roiWeit(i)+roiCoord(i,1),y*roiWeit(i)+roiCoord(i,2),z*roiWeit(i)+roiCoord(i,3)); 
        set(hh, 'FaceColor', roiColor(i,:),'EdgeColor', 'none');
        hold on;
    end
end
hold off;

%% Right lateral
a(2)=axes('position',[0.45 0.5 h  w]);
%subplot(2,2,2);
hh=trisurf(cutSurf.faces(tr,:)-cuv,cutSurf.vertices(vr,1),cutSurf.vertices(vr,2),cutSurf.vertices(vr,3),...
    double(ones(size(vr))),'EdgeColor','none');
colormap(CM);
%set(hh, 'FaceAlpha', faceAlpha);  % Fast in Matlab 2017 but VERY slow in 2012
view(90,0); 
daspect([1 1 1]); 
whitebg(gcf, [1 1 1]);
set(gcf,'Color',[1 1 1],'InvertHardcopy','off');
lighting phong; material([0.5 0.5 0.5]); shading interp; 
set(hh, 'FaceColor', faceColor, 'FaceAlpha', faceAlpha);
axis tight; axis vis3d off;
camlight('right'); 

hold on;
for i=1:size(roiCoord,1)
    if roiCoord(i,1)>-2
        hh=surf(x*roiWeit(i)+roiCoord(i,1),y*roiWeit(i)+roiCoord(i,2),z*roiWeit(i)+roiCoord(i,3)); 
        set(hh, 'FaceColor', roiColor(i,:),'EdgeColor', 'none');
        hold on;
    end
end
hold off;

%% Left medial
a(3)=axes('position',[0.06 0.13 h  w]);
%subplot(2,2,1);
hh=trisurf(cutSurf.faces(tl,:),cutSurf.vertices(vl,1),cutSurf.vertices(vl,2),cutSurf.vertices(vl,3),...
    double(ones(size(vl))),'EdgeColor','none');
colormap(CM);
%set(hh, 'FaceAlpha', faceAlpha);  % Fast in Matlab 2017 but VERY slow in 2012
view(90,0); 
daspect([1 1 1]); 
whitebg(gcf, [1 1 1]);
set(gcf,'Color',[1 1 1],'InvertHardcopy','off');
lighting phong; material([0.5 0.5 0.5]); shading interp; 
set(hh, 'FaceColor', faceColor, 'FaceAlpha', faceAlpha);
axis tight; axis vis3d off;
camlight('right'); 

hold on;
for i=1:size(roiCoord,1)
    if roiCoord(i,1)<2
        hh=surf(x*roiWeit(i)+roiCoord(i,1),y*roiWeit(i)+roiCoord(i,2),z*roiWeit(i)+roiCoord(i,3)); 
        set(hh, 'FaceColor', roiColor(i,:),'EdgeColor', 'none');
        hold on;
    end
end
hold off;

%% Right medial
a(4)=axes('position',[0.45 0.13 h  w]);
%subplot(2,2,2);
hh=trisurf(cutSurf.faces(tr,:)-cuv,cutSurf.vertices(vr,1),cutSurf.vertices(vr,2),cutSurf.vertices(vr,3),...
    double(ones(size(vr))),'EdgeColor','none');
colormap(CM);
%set(hh, 'FaceAlpha', faceAlpha);  % Fast in Matlab 2017 but VERY slow in 2012
view(-90,0); 
daspect([1 1 1]); 
whitebg(gcf, [1 1 1]);
set(gcf,'Color',[1 1 1],'InvertHardcopy','off');
lighting phong; material([0.5 0.5 0.5]); shading interp; 
set(hh, 'FaceColor', faceColor, 'FaceAlpha', faceAlpha);
axis tight; axis vis3d off;
camlight('right'); 

hold on;
for i=1:size(roiCoord,1)
    if roiCoord(i,1)>-2
        hh=surf(x*roiWeit(i)+roiCoord(i,1),y*roiWeit(i)+roiCoord(i,2),z*roiWeit(i)+roiCoord(i,3)); 
        set(hh, 'FaceColor', roiColor(i,:),'EdgeColor', 'none');
        hold on;
    end
end
hold off;

if 0
    %% to add the colorbar
    a(5)=axes('position',[0.3, 0.1, 0.4,0.1]);
    barimg=interp1([1:64],redblue, [1:0.5:64]);
    barimg=repmat(barimg',[1,1,12]); 
    barimg=permute(barimg, [3 2 1]);
    imshow(barimg, []);
    %set(gca, 'xtick', [2,127], 'xticklabel', [min(roiWeit), max(roiWeit)], 'ytick',[]);
    %set(gca,  'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',12);
    %text(2, 20, num2str(min(roiWeit)), 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',22);
    %text(122, 20, num2str(max(roiWeit)), 'FontName', 'Arial', 'FontWeight', 'bold', 'Fontsize',22);
end
    
    
end % main function

