% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group 2020

function [S, Bdry] = VOMT(T, V, Bdry, VB, VI)
Bdry.S = Sphere.AOMT(Bdry.F, Bdry.V);
V = Sphere.VolumeNormalize(T, V);
Bdry.V = V(VB,:);
S = Sphere.Homotopy(T, V, Bdry, VB, VI);
Vol = Tet.VertexVolume(T, V);
S = Vertex.RotOMT(S, V, Vol);