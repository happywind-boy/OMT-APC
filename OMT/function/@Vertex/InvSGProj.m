function S = InvSGProj(C)
B = 1+C(:,1).^2+C(:,2).^2;
S = [2*C(:,1)./B, 2*C(:,2)./B, (B-2)./B];