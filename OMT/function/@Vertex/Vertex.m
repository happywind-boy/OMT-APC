% Vertex is a class of functions for vertex operations.
%
% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group

classdef Vertex
   methods (Static)
       V = Add(V, c);
       [V, Center] = Centralize(V);
       V = Divide(V, c);
       [InnerIdx, OuterIdx] = InnerIndex(uv, Radius);
       V = Inv(V);
       S = InvSGProj(C);
       NormV = Norm(V);
       NormV2 = Norm2(V);
       V = Normalize(V);
       V = R3(V);
       [RP, R] = RotOMT(P, Q, w);
       uv = SGProj(S);
       V = Times(V, c);
   end
end
