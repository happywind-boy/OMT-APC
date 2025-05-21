function [S, Bdry] = VOMT_CubeHomotopy(T, V, Bdry, VB, VI, Weight, Cube, p) 
 % AOMT 
 Bdry.S = Sphere.AOMT_ProjGrad(Bdry.F, Bdry.V, Bdry.Weight); 
 Bdry.S = InverseBallMap(Cube.T, Cube.V, Cube.S, Bdry.S); 
   % VOMT 
 Vno = size(V,1); 
 S = zeros(Vno,3); 
 TimeStep = 1/p; 
 L = Tet.Laplacian(T, V); 
   % Homotopy 
 for t = TimeStep:TimeStep:1 
 S(VB,:) = t*Bdry.S + (1-t)*Bdry.V; 
 [S, L] = SolveInner(T, S, Weight, L, VI, VB); 
 end 
   % run with a fix boundary, correct it's laplacian matrix 
 for t = 1 : (10-p) 
 [S, L] = SolveInner(T, S, Weight, L, VI, VB); 
 end 
end 
 function [S, L] = SolveInner(T, S, Weight, L, VI, VB) 
 tol = 1e-8; 
 cmgPASS = false(3,1); cmgIter = 50; 
 ssorPASS = false(3,1); ssorIter = 100; 
   rhs = -L(VI,VB)*S(VB,:); 
 try 
 pfun = cmg_sdd(L(VI,VI)); 
 S(VI,1) = pcg(L(VI,VI), rhs(:,1), tol, cmgIter, pfun); cmgPASS(1) = true; 
 S(VI,2) = pcg(L(VI,VI), rhs(:,2), tol, cmgIter, pfun); cmgPASS(2) = true; 
 S(VI,3) = pcg(L(VI,VI), rhs(:,3), tol, cmgIter, pfun); cmgPASS(3) = true; 
 catch ME 
 warning([ME.identifier ' CMG not work, try SSOR\n']); 
 [M1, M2] = SSOR_Precond(L(VI,VI), 1.2); 
 S(VI,1) = pcg(L(VI,VI), rhs(:,1), tol, ssorIter, M1, M2); ssorPASS(1) = true; 
 S(VI,2) = pcg(L(VI,VI), rhs(:,2), tol, ssorIter, M1, M2); ssorPASS(2) = true; 
 S(VI,3) = pcg(L(VI,VI), rhs(:,3), tol, ssorIter, M1, M2); ssorPASS(3) = true; 
 end 
 if ~all(cmgPASS | ssorPASS) 
 save('ErrorCase.mat', 'L', 'VI', 'VB', 'S'); 
 error('Both CMG and SSOR not work.\n'); 
 end 
   Vol_S = abs(Tet.Volume(T, S)); 
 Vol_S = Vol_S / sum(Vol_S) * (4*pi/3); 
 Sigma = Weight.T ./ Vol_S; 
   L = Tet.Laplacian(T, S, Sigma); 
 end 