function [RP, R] = RotOMT(P, Q, w)
if nargin < 3
    M = P.'*Q;
else
    % M = P.'*diag(w)*Q;
    M = bsxfun(@times, Q, w/max(w));
    M = P.'*M;
end

if sum(sum(isnan(M)))+sum(sum(isinf(M))) == 0
    [U, ~, V] = svd(M);
    R = V*U.';
else
    R = eye(3);
end
if det(R)>0
    RP = P*R.';
else
    RP = P;
end

