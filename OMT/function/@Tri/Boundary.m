function [VB, VI] = Boundary(F)
Vno = max(max(F));
if nargin == 1
    V = zeros(Vno,2);
end
M = triangulation(F, V);
VB = freeBoundary(M);
VB = VB(:,1);
if nargout > 1
    VI = setdiff((1:Vno).', VB);
end