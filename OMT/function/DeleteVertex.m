function [F, V] = DeleteVertex(F, V, Vid)
Vno = size(V,1);
Fid = sum(ismember(F, Vid), 2);
Fid = Fid > 0;
F(Fid,:) = [];
nVid = length(Vid);
RemoveVid = Vno:-1:Vno-nVid+1;
SourseVid = setdiff(RemoveVid, Vid);
TargetVid = setdiff(Vid, RemoveVid);
V(TargetVid,:) = V(SourseVid,:);
V(RemoveVid,:) = [];
[a, b] = ismember(F, SourseVid);
t = find(a~=0);
F(t) = TargetVid(b(t));