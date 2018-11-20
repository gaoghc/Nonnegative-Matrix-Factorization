function [F, G, S, P, obj] = BNMF_manifold(X, Finit, Ginit, c, neighbor, lambda, maxiter)

% min_{F, G, P}||X-FG^T||_F^2 + \lambda Tr(P^TLP)
% s.t. F\geq 0, ||g||_0\leq k, P^TP = I

[d, n] = size(X);
F = Finit;
G = Ginit;

k = size(G, 2);  

preObj = 0;

ratio = 2;

for it = 1:maxiter
    
    %%===========update P=========
    S = [zeros(n, n), G; G', zeros(k, k)];
    D = diag(sum(S, 2));
    L = D - S;
    
    [V, E] = eig(L);
    [~, idx] = sort(diag(E));
    P = V(:, idx(2:c+1));

   
    
    %%==========update F =========
    F = F.*((X*G)./(F*G'*G + eps));
    assert(sum(sum(F<0))==0);
    
   
    %%==========update G==========
    for i=1:n+k
        for j=1:n+k
            Q(i, j) = norm(P(i,:)-P(j,:))^2;
        end
    end
    
    Q12 = Q(1:n, n+1:end);
    Q21 = Q(n+1:end, 1:n);
    A = Q21'+Q12;
    alpha = lambda/2;
    
    % min_{||g||_0<k} ||g-grad||_2^2
    
    FtF = F'*F;
    
    for i=1:n
        
        g = G(i, :)';
        a = A(i, :)';
        x = X(:, i);

%         g = linesearch(F, x, g, a, neighbor, alpha);
%         G(i,:) = g';
  
        grad = alpha*a - F'*x + FtF*g;
        gtemp = g - 0.00001*2*grad; 
        
        gtemp = max(gtemp, 0);
        [~, idx] = sort(gtemp, 'descend');
        g = zeros(k, 1);
        g(idx(1:neighbor)) = gtemp(idx(1:neighbor));
        
        G(i, :) = g;
        
        
        
    end
    
    S = [zeros(n, n), G; G', zeros(k, k)];
    D = diag(sum(S, 2));
    L = D - S;
    
    
    %%===========obj===========

    
    obj(it) = norm(X-F*G', 'fro')^2 ;%+ 2*lambda*sum(diag(P'*L*P));
    fprintf('=====%d-th iteration, obj=%f=====\n', it, obj(it));
    

    [eigVec, eigVal] = eig(L);
    evD = diag(eigVal);
    [~,idx] = sort(evD, 'ascend');
    fn1 = sum(evD(idx(1:c)));
    fn2 = sum(evD(idx(1:c+1)));
    
    if fn1 > 1e-11
        ratio = 2;
        lambda = ratio*lambda;
        fprintf('lambda = %f\n', lambda);
    elseif fn2 < 1e-11
        ratio = 2;
        lambda = lambda/ratio;
        fprintf('lambda = %f\n', lambda);
    else
        fprintf('lambda = %f\n', lambda);
        break;
    end

end

