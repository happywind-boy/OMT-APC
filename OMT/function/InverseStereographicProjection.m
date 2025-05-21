function S = InverseStereographicProjection(C, option)
if nargin < 2
    option = 'north';
end
B = 1+C(:,1).^2+C(:,2).^2;
S = [2*C(:,1)./B, 2*C(:,2)./B, (B-2)./B];
if strcmpi(option, 'south')
    S(:,3) = -S(:,3);
end