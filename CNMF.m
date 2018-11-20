function [F,G] = CNMF(X, Finit, Ginit, ratio, maxiter)
% min ||X-FG^T||_cappednorm
%X: dxn
%ratio: the ratio of outliers, set it according to the paper

disp('Capped ONMF');

[numOfFeatures, numOfSamples] = size(X);
D = eye(numOfSamples, numOfSamples);


G = Ginit;
F = Finit;

preObj = 0;


%theta = 100000;


for it=1:maxiter

    F = F.*((X*D*G)./(F*G'*D*G+eps));
    G = G.*sqrt((D*X'*F)./(D*G*G'*X'*F+eps));
    
    
    Delta = X - F*G';
    DeltaN = sqrt(sum(Delta.*Delta, 1));
    
    if it == 1
        DeltaS = sort(DeltaN);
        len = length(DeltaS);
        theta = DeltaS(floor(ratio*len))
    end
    

    DiagD = 0.5./(DeltaN+eps);
    Idx = find(DeltaN>theta);
    DiagD(Idx) = 0;
    D = diag(DiagD);
    
    
    
    fprintf('numOfOutliers = %d\n', length(Idx));
    
    %--------obj----------
    T = X - F*G';
    obj = sum(diag(T*D*T'));
    myobj(it) = obj;
    
    if mod(it, 10) == 0
        fprintf('numOfOutliers = %d\n', length(Idx));
        fprintf('%dth iteration, obj = %f \n', it, obj);
    end
    %fprintf('%d-th iteration, obj = %f\n', it, obj);
    
    
   if abs(preObj - obj) < 0.00001*preObj
        break;
   end

    preObj = obj;
end
%save ORLObj.mat myobj
assert(sum(sum(F<0)) == 0);
assert(sum(sum(G<0)) == 0);
fprintf('total iterations:%d\n', it);