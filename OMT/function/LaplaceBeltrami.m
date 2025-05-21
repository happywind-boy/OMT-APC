function L = LaplaceBeltrami(F, V, Sigma)
if nargin < 3
    Sigma = 1;
end
Vno = size(V,1);
if size(V,2)==2
    V = [V, zeros(Vno,1)];
end
E12 = V(F(:,2),:) - V(F(:,1),:);
E23 = V(F(:,3),:) - V(F(:,2),:);
E31 = V(F(:,1),:) - V(F(:,3),:);
Fno = size(F,1);
W = zeros(Fno,3);
W(:,1) = -0.5*sum(E31.*E23, 2) ./ sqrt( sum(cross(E31,E23).^2, 2) ) ./ Sigma;
W(:,2) = -0.5*sum(E12.*E31, 2) ./ sqrt( sum(cross(E12,E31).^2, 2) ) ./ Sigma;
W(:,3) = -0.5*sum(E23.*E12, 2) ./ sqrt( sum(cross(E23,E12).^2, 2) ) ./ Sigma;
K = sparse(F, F(:,[2, 3, 1]), W, Vno, Vno);
K = K + K.';
L = diag( sum(K, 2) ) - K;