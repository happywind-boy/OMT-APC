function [InnerIdx, OuterIdx] = InnerOuterIndex(uv, Radius)
if nargin < 2
    Radius = 1.2;
end
InnerIdx = VecNorm(uv) < Radius;
OuterIdx = ~InnerIdx;
InnerIdx = find(InnerIdx);
OuterIdx = find(OuterIdx);