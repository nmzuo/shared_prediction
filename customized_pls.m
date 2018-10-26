function [PLS,DR_X,DR_tX,PLSR_ty]=customized_pls(X,y,tX,ty,A,method)
if nargin<6;method='center';end
if nargin<5;A=2;end


[Mx,Nx]=size(X);

%+++ check effectiveness of A.
A=min([Mx Nx A]);


%+++ data pretreatment
[Xs,xpara1,xpara2]=pretreat(X,method);
[ys,ypara1,ypara2]=pretreat(y,'center');

%+++ Use the pretreated data to build a PLS model
[B,Wstar,T,P,Q,W,R2X,R2Y]=plsnipals(Xs,ys,A); % notice that here, B is the regression coefficients linking Xs and ys.

%+++ perform target projection;
[tpt,tpw,tpp,SR]=tp(Xs,B);     

%+++ get regression coefficients that link X and y (original data) ************
coef=zeros(Nx+1,A);
for j=1:A
    Bj=Wstar(:,1:j)*Q(1:j);
    C=ypara2*Bj./xpara2';
    coef(:,j)=[C;ypara1-xpara1*C;];
end

%+++ ********************************************
x_expand=[X ones(Mx,1)];
ypred=x_expand*coef(:,end);
error=ypred-y;
%********************************************
SST=sum((y-mean(y)).^2);
SSR=sum((ypred-mean(y)).^2);
SSE=sum((y-ypred).^2);
R2=1-SSE/SST;


%+++ Output************************************** 
PLS.method=method;
PLS.X_pretreat=Xs;
PLS.y_pretreat=ys;
PLS.regcoef_pretreat=B;
PLS.regcoef_original_all=coef;
PLS.regcoef_original=coef(:,end);
PLS.X_scores=T;
PLS.X_loadings=P;
PLS.W=W;
PLS.Wstar=Wstar;
PLS.y_fit=ypred;
PLS.fitError=error;
PLS.tpscores=tpt;
PLS.tploadings=tpp;
PLS.SR=SR;
PLS.SST=SST;
PLS.SSR=SSR;
PLS.SSE=SSE;
PLS.RMSEF=sqrt(SSE/Mx);
PLS.R2=R2;
%+++ END ++++++++++++++++++++++++++++++++++++
DR_X=T; 
[tMx,~]=size(tX);
tx_expand=[tX ones(tMx,1)];
PLSR_ty=tx_expand*coef(:,end);
tXs=pretreat(tX,method,xpara1,xpara2);
tys=pretreat(ty,'center',ypara1,ypara2);
[~,~,DR_tX,~,~,~,~,~]=plsnipals(tXs,tys,A);
