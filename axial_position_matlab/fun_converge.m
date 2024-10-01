function zs=fun_converge(Hf,HFZE,z,delerror)

a=z-3;
b=z+3;
c=(a+b)/2;
qc=0;
while abs(a-b)>=delerror
    qc=qc+1;
    fa=fun_calc_curvature(Hf,HFZE,a);
    fc=fun_calc_curvature(Hf,HFZE,c);
    
    if fa*fc>0 a=c;    
        elseif fa*fc<0 b=c;end
            c=(a+b)/2;
    fprintf('iteration = %d curvature=%f \n',qc,c);        
end
zs=c;