%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%    fun_maxz_5_190728_rev.m
%         S.Hasegawa
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function zq=fun_maxz_5_190728_rev(y1,y2,x1,x2,Hf,K,zc);

global delta lambda nme h k

zm=21;
for ii=1:1:zm;
      
z0(ii)=zc-5+(ii-1)*0.05*10;

da=-z0(ii);

L0=delta*K;

%---------------Diffraction calculation by D-FFT
% Spectrum of the initial field
fex=K/L0;fey=fex;% sampling of frequency plane

fx=[-fex/2:fex/K:fex/2-fex/K];
fy=[-fey/2:fey/K:fey/2-fey/K];

[FX,FY]=meshgrid(fx,fy);
E=exp(i*k*da*sqrt(1-(h*FX).^2-(h*FY).^2)); % Angular spectrum transfer function
% Diffraction   
[Hrr]=ifft2(Hf.*E);
M(ii) = max(max(abs(Hrr(y1:y2,x1:x2)).^2));

end

  [MM,zi]=max(M);

zq=zc-5+(zi-1)*0.05*10;
fprintf('Maximum intensinty zq=%f\n',zq);
figure(20),plot(z0,M);title('Intensity as a function of z (mm))');
xlabel(' z (mm)') ;ylabel('Intensity');

      



