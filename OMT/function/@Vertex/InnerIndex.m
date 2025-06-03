function [InnerIdx, OuterIdx] = InnerIndex(uv, Radius)
if nargin < 2
    Radius = 1.2;
end
InnerIdx = Vertex.Norm(uv) < Radius;
if nargout == 2
    OuterIdx = ~InnerIdx;
    OuterIdx = find(OuterIdx);
end
InnerIdx = find(InnerIdx);
