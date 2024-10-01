%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    fun_recovery using single-shot phase-shifting method
%          S.Hasegawa
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Hf,K]=fun_recovery(holl,del)

[hh,ww]=size(holl);

L0=del*(ww-1);

mab=0;
km=mab;
ms=km+1;
ms=3;

for kh=1:1:hh
for ks=1:ms:ww-ms
xa1(ks)=-L0/2+L0/ww*(ks-1);
xa2(ks)=-L0/2+L0/ww*(ks);
xa3(ks)=-L0/2+L0/ww*(ks+1);
ya1(ks)=holl(kh,ks);
ya2(ks)=holl(kh,ks+1);
ya3(ks)=holl(kh,ks+2);
end

xaa1=xa1(1:ms:ww-ms);
xaa2=xa2(1:ms:ww-ms);
xaa3=xa3(1:ms:ww-ms);
yaa1=ya1(1:ms:ww-ms);
yaa2=ya2(1:ms:ww-ms);
yaa3=ya3(1:ms:ww-ms);

kbb=1:1:ww-ms;
xi=-L0/2+L0/ww*(kbb-1);
holl1(kh,kbb)=interp1(xaa1,yaa1,xi,'spline');
holl2(kh,kbb)=interp1(xaa2,yaa2,xi,'spline');
holl3(kh,kbb)=interp1(xaa3,yaa3,xi,'spline');

end
wave=zeros(hh,ww);
for ny=1:1:hh
for nx=1:1:ww-ms
s1(ny,nx)=holl1(ny,nx)*cos(0*2*pi/3)+holl2(ny,nx)*cos(1*2*pi/3)+holl3(ny,nx)*cos(2*2*pi/3);
si(ny,nx)=s1(ny,nx);
c1(ny,nx)=holl1(ny,nx)*sin(0*2*pi/3)+holl2(ny,nx)*sin(1*2*pi/3)+holl3(ny,nx)*sin(2*2*pi/3);
ci(ny,nx)=c1(ny,nx);
wave(ny,nx)=1/6*((2*holl1(ny,nx)-holl2(ny,nx)-holl3(ny,nx))+sqrt(3)*1i*(holl2(ny,nx)-holl3(ny,nx)));
end;
end;

X=conj(wave);

[M,N]=size(X);
X=double(X);
K=2*max(M,N);

% Zeros-padding to get KxK image
Z1=zeros(K,floor((K-N)/2));
Z2=zeros(floor((K-M)/2),N);
Z3=zeros(K-M-floor((K-M)/2),N);
Z4=zeros(K,K-N-floor((K-N)/2));

Xp=[Z1,[Z2;X;Z3],Z4];
U1=double(Xp);
Hf=fftshift(fft2(U1,K,K));