function [VI, VB, I, L] = FindCubeFace(F, V, Cid, Pid)
%   8 --- 7 
% 5 --- 6
%   4 --- 3
% 1 --- 2 
L = LaplaceBeltrami(F, V);
B = cell(3,2);
I = cell(3,2);
B{1,1} = [Cid(1); Pid{1,4}; Cid(4); Pid{4,8}; Cid(8); Pid{8,5}; Cid(5); Pid{5,1}];
B{1,2} = [Cid(2); Pid{2,3}; Cid(3); Pid{3,7}; Cid(7); Pid{7,6}; Cid(6); Pid{6,2}];
[I{1,1}, I{1,2}] = B2I(B{1,1}, B{1,2}, L);
B{2,1} = [Cid(1); Pid{1,2}; Cid(2); Pid{2,6}; Cid(6); Pid{6,5}; Cid(5); Pid{5,1}];
B{2,2} = [Cid(4); Pid{4,3}; Cid(3); Pid{3,7}; Cid(7); Pid{7,8}; Cid(8); Pid{8,4}];
[I{2,1}, I{2,2}] = B2I(B{2,1}, B{2,2}, L);
B{3,1} = [Cid(1); Pid{1,2}; Cid(2); Pid{2,3}; Cid(3); Pid{3,4}; Cid(4); Pid{4,1}];
B{3,2} = [Cid(5); Pid{5,6}; Cid(6); Pid{6,7}; Cid(7); Pid{7,8}; Cid(8); Pid{8,5}];
[I{3,1}, I{3,2}] = B2I(B{3,1}, B{3,2}, L);
Vno = size(V, 1);
Vid = (1:Vno).';
VB = cell(3,1);
VI = cell(3,1);
for k = 1:3
    VB{k} = [I{k,1}; I{k,2}];
    VI{k} = setdiff(Vid, VB{k});
end


function [I1234, I5678] = B2I(B1234, B5678, L)
Vno = size(L,1);
Vid = (1:Vno).';
B = [B1234; B5678];
I = setdiff(Vid, B);
f = zeros(Vno, 1);
f(B5678) = 1;
rhs = -L(I,B)*f(B);
f(I) = L(I,I)\rhs;
I1234 = find(f==0);
f = zeros(Vno, 1);
f(B1234) = 1;
rhs = -L(I,B)*f(B);
f(I) = L(I,I)\rhs;
I5678 = find(f==0);
