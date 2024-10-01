%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Guess zp using curvature along optical axis in the prosescing of experiment 
%       
%      Shin-ya Hasegawa 
%
%       fun_recovery.m
%       fun_maxz_5_190728_rev.m
%       fun_converge.m
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;close all;
global delta lambda nme h k

sm=1;
z3=input('Approximate z [mm] ?  e.g. 158.0 ');
%z3=156;
delta=0.0022;               % [mm]
lambda=0.532*1e-3;          % [mm]
nme=1.3337;                 % Anmbient refractive index
k=2*pi/lambda*nme;          % Wave number

%  load Hologram 
load('holl_100_100');
load('holl0_100_100');

figure(1);imagesc(holl);

% Wave recovery by phase-shifting optical setup.  
[Hf,K]=fun_recovery(holl,delta); % particle 
figure(2),imagesc(abs(Hf(floor(K/2)-500:1:floor(K/2)+500,floor(K/2)-500:1:floor(K/2)+500)));
[HFZE,K]=fun_recovery(holl0,delta); % without particle ( for phase calculation ).

L0=delta*K;
da=-z3;
h=lambda./nme;

% Angular Spectrum Method

fex=K/L0;fey=fex;
fx=[-fex/2:fex/K:fex/2-fex/K];
fy=[-fey/2:fey/K:fey/2-fey/K];
[FX,FY]=meshgrid(fx,fy);
E=exp(i*k*da*sqrt(1-(h*FX).^2-(h*FY).^2)); 

% Find maximum intensity along z.

[Hrr]=ifft2(Hf.*E);
figure(5),imagesc(abs(Hrr).^2);
fprintf('Click the position of a particle in Fig.5.\n');
wxy=ginput(1);mar=100;  % Select the particle
x1=floor(wxy(1,1))-mar;
x2=floor(wxy(1,1))+mar;
y1=floor(wxy(1,2))-mar;
y2=floor(wxy(1,2))+mar;

Hrr0=Hrr(y1:y2,x1:x2);
figure(6),imagesc(abs(Hrr0).^2);
zc=fun_maxz_5_190728_rev(y1,y2,x1,x2,Hf,K,z3);

% Approximate curvature zero clossing point along z.

delerror=0.05; % zero crossing error torerance [mm]
zs=fun_converge(Hf,HFZE,zc,delerror);
fprintf(' Guess z by curvature = %f, error<= %f \n',zs,delerror);
