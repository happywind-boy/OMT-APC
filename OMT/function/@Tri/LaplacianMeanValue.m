function L = LaplacianMeanValue(F, V)
% if size(V,2) == 2
%     V = [V, 0*V(:,1)];
% end
Fno = size(F,1);
Vno = size(V,1);
W = zeros(Fno,3);
v_ki = V(F(:,1), :) - V(F(:,3), :);
v_kj = V(F(:,2), :) - V(F(:,3), :);
v_ij = V(F(:,2), :) - V(F(:,1), :);

% Compute edge lengths of each face, L = [Lij, Ljk, Lki] ------------------
edgeLen = zeros(Fno,3);
edgeLen(:,1) = sqrt( sum( v_ij.^2, 2 ) );
edgeLen(:,2) = sqrt( sum( v_kj.^2, 2 ) );
edgeLen(:,3) = sqrt( sum( v_ki.^2, 2 ) );

% Compute tan( theta_k/2 ) ------------------------------------------------
num  = sum( v_ki.*v_kj, 2 );
temp = edgeLen(:,2) .* edgeLen(:,3);
num  = num + temp;
temp = cross(v_ki, v_kj);
den  = sqrt( sum( temp.^2, 2 ) );
W(:,3) = den./num;

% Compute half weight of Wki and Wkj --------------------------------------
temp = W(:,3)./edgeLen(:,3);
K    = sparse(F(:,3),F(:,1),temp,Vno,Vno);
temp = W(:,3)./edgeLen(:,2);
temp = sparse(F(:,3),F(:,2),temp,Vno,Vno);
K = K + temp;

% Compute tan( theta_i/2 ) ------------------------------------------------
v_ki = -v_ki; % v_ki is now equal to v_ik = v_k - v_i
num  = sum( v_ij.*v_ki, 2 );
temp = edgeLen(:,1) .* edgeLen(:,3);
num  = num + temp;
temp = cross(v_ij, v_ki);
den  = sqrt( sum( temp.^2, 2 ) );
W(:,1) = den./num;

% Compute half weight of Wij and Wik --------------------------------------
temp = W(:,1)./edgeLen(:,1);
temp = sparse(F(:,1),F(:,2),temp,Vno,Vno);
K = K + temp;
temp = W(:,1)./edgeLen(:,3);
temp = sparse(F(:,1),F(:,3),temp,Vno,Vno);
K = K + temp;

% Compute tan( theta_j/2 ) ------------------------------------------------
v_kj = -v_kj;     % v_kj is equal to v_jk = v_k - v_j
v_ij = -v_ij;     % v_ij is equal to v_ji = v_i - v_j
num  = sum( v_kj.*v_ij, 2 );
temp = edgeLen(:,1) .* edgeLen(:,2);
num  = num + temp;
temp = cross(v_kj, v_ij);
den  = sqrt( sum( temp.^2, 2 ) );
W(:,2) = den./num;

% Compute half weight of Wji and Wjk --------------------------------------
temp = W(:,2)./edgeLen(:,1);
temp = sparse(F(:,2),F(:,1),temp,Vno,Vno);
K = K + temp;
temp = W(:,2)./edgeLen(:,2);
temp = sparse(F(:,2),F(:,3),temp,Vno,Vno);
K = K + temp;
L = diag( sum(K, 2) ) - K;
