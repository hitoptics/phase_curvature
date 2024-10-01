function coe=fun_chebychev_240525(xn,phc)
%clear A % just need to clean this out as otherwise the A created above would cause an error

xgrid=xn'; % Turn up the interval to [0,10] or [0,100] and 5th order Chebyshev will start to struggle to approximate.
ygrid=phc';
% We could fit a polynomial of order m.
m=5;
% xgrid=-1:0.1:1; xgrid=xgrid';
% ygrid=exp(xgrid);
numdatapoints=length(xgrid);

% Create zgrid by moving xgrid from [a,b] onto [-1,1]
a=min(xgrid); b=max(xgrid);
zgrid = (2*xgrid-a-b)/(b-a);

% Fit chebyshev polynomial of order m
A(:,1) = ones(numdatapoints,1);
if m > 1
   A(:,2) = zgrid;
end
if m > 2
  for k = 3:m+1
     A(:,k) = 2*zgrid.*A(:,k-1) - A(:,k-2);  %% recurrence relation
  end
end
fittedchebyshevcoeffs = A \ ygrid;
% Note: other than adding zgrid line this is unchanged (A depends on zgrid
% instead of xgrid, but this is largely irrelevant).

% Evaluate the fitted chebyshev polynomial on our zgrid
b1 = zeros(numdatapoints,1);
b0 = zeros(numdatapoints,1);
for jj=m:-1:0
    b2=b1;
    b1=b0;
    b0=fittedchebyshevcoeffs(jj+1)+2*zgrid.*b1-b2; % only change is now uses zgrid
end
ygrid_fittedchebyshev= 0.5*(fittedchebyshevcoeffs(1)+b0-b2);

% Now graph to take a look at the fit
% figure(25),plot(xgrid,ygrid,'*',xgrid,ygrid_fittedchebyshev,'-')
% title('Fitted Chebyshev on interval [a,b]')
% legend('Original function', 'Fitted chebyshev approximation')
coe=fittedchebyshevcoeffs(3);

%
% When people talk about 'hypercubes' they are talking about this trick of
% switching problems from [a,b] to [-1,1] and solving there, then switching
% back. It has the advantage that all your codes written to solve models
% can just work with the [-1,1] hypercube (or more accurately, the obvious
% extension of this to [-1,1]^d in higher dimensions, where d is number of dimensions).