%------------------------------------------
%
%    fun_calc_curvature.m
%         S.Hasegawa
%
%------------------------------------------


function coe=fun_calc_curvature(HF,HFZ,z)

global delta lambda nme h k

[hq wq]=size(HF);
K=hq;

%%%%%%%
L0x=delta*wq;L0y=delta*hq;

fex=wq/L0x;fey=hq/L0y;%
da=-z;

fx=[-fex/2:fex/wq:fex/2-fex/wq];
fy=[-fey/2:fey/hq:fey/2-fey/hq];
[FX,FY]=meshgrid(fx,fy);

E=exp(i*k*da*sqrt(1-(h*FX).^2-(h*FY).^2)); % Angular spectrum transfer function

    [Hrr]=ifft2(HF.*E);
    [Hrr0]=ifft2(HFZ.*E);
    [haa waa]=size(Hrr);

 
hmax=max(max(abs(Hrr)));
[yc,xc]=find(abs(Hrr)==hmax); %fprintf('xc=%d yc=%d\n',xc,yc);
mar=500;
Hrre=Hrr(yc-mar:yc+mar,xc-mar:xc+mar);
Hrr0e=Hrr0(yc-mar:yc+mar,xc-mar:xc+mar);
[haa waa]=size(Hrre);


phw=(angle(Hrre)-angle(Hrr0e));
ta=100;
phw=phw(floor(haa/2)-ta:1:floor(haa/2)+ta,floor(waa/2)-ta:1:floor(waa/2)+ta);    
Hrr=Hrre(floor(haa/2)-ta:1:floor(haa/2)+ta,floor(waa/2)-ta:1:floor(waa/2)+ta);   
 
HR=Hrr/hmax;

% Unwrappung 
mex Miguel_2D_unwrapper.cpp
%Miguel_2D_unwrapper.cpp
WrappedPhase = single(phw);
UnwrappedPhase = Miguel_2D_unwrapper(WrappedPhase);
phwa=double(UnwrappedPhase);
phwas = medfilt2(phwa);
[haa waa]=size(phwas);
medphase=phwas;
[haa waa]=size(medphase);
qle=floor(-da*1000);
ta=20;

phwaa=medphase(floor(waa/2)-ta:1:floor(waa/2)+ta,floor(haa/2)-ta:1:floor(haa/2)+ta);
n=floor(waa/2)-ta:1:floor(waa/2)+ta;m=floor(haa/2)-ta:1:floor(haa/2)+ta;
ln=length(n);
lm=length(m);

n=1:1:ln;
m=1:1:lm;
[xs ys]=meshgrid((-ta+(n-1))*delta,(-ta+(m-1))*delta);

xn=xs(ta+1,:);phc=phwaa(ta+1,:);
coe=fun_chebychev_240525(xn,phc);



