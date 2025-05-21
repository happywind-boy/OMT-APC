function L = VolumeLaplacian(T, V, Sigma)

n = size(V,1);
% cotangents of dihedral angles
if nargin > 2
    C = cotWeight(T, V, Sigma);
else
    C = cotWeight(T, V);
end

%% Zero-out almost zeros to help sparsity
%C(abs(C)<10*eps) = 0;
% add to entries
L = sparse(T(:,[2 3 1 4 4 4]), T(:,[3 1 2 1 2 3]), C, n, n);
% add in other direction
L = L + L.';
% diagonal is minus sum of offdiagonal entries
L = diag(sum(L,2)) - L;
%% divide by factor so that regular grid laplacian matches finite-difference laplacian in interior
%L = L./(4+2/3*sqrt(3));
%% multiply by factor so that matches legacy laplacian in sign and "off-by-factor-of-two-ness"
%L = L*0.5;
% flip sign to match cotmatix.m
% if(all(diag(L)>0))
% 	warning('Flipping sign of cotmatrix3, so that diag is negative');
% 	L = -L;
% end


function C = cotWeight(T, V, Sigma)

% lengths of edges opposite *face* pairs: 23 31 12 41 42 43
l = [ sqrt(sum((V(T(:,4),:)-V(T(:,1),:)).^2,2)) ...
      sqrt(sum((V(T(:,4),:)-V(T(:,2),:)).^2,2)) ...
      sqrt(sum((V(T(:,4),:)-V(T(:,3),:)).^2,2)) ...
      sqrt(sum((V(T(:,2),:)-V(T(:,3),:)).^2,2)) ...
      sqrt(sum((V(T(:,3),:)-V(T(:,1),:)).^2,2)) ...
      sqrt(sum((V(T(:,1),:)-V(T(:,2),:)).^2,2)) ];

% (unsigned) face Areas (opposite vertices: 1 2 3 4)
s = 0.5*[ doublearea_intrinsic(l(:,[2 3 4])) ...
          doublearea_intrinsic(l(:,[1 3 5])) ...
          doublearea_intrinsic(l(:,[1 2 6])) ...
          doublearea_intrinsic(l(:,[4 5 6])) ];


%     [~,cos_theta] = dihedral_angles([],[],'SideLengths',l,'FaceAreas',s);
H_sqr = (1/16) * ...
    (4*l(:,[4 5 6 1 2 3]).^2.* l(:,[1 2 3 4 5 6]).^2 - ...
    ((l(:, [2 3 4 5 6 1]).^2 + l(:,[5 6 1 2 3 4]).^2) - ...
     (l(:, [3 4 5 6 1 2]).^2 + l(:,[6 1 2 3 4 5]).^2)).^2);
cos_theta= (H_sqr - s(:,[2 3 1 4 4 4]).^2 - ...
                    s(:,[3 1 2 1 2 3]).^2)./ ...
                  (-2*s(:,[2 3 1 4 4 4]).* ...
                      s(:,[3 1 2 1 2 3]));

vol = volume_intrinsic(l);
vol = abs(vol);
%% To retrieve dihedral angles stop here...
%theta = acos(cos_theta);
%theta/pi*180

% Law of sines
% http://mathworld.wolfram.com/Tetrahedron.html
sin_theta = bsxfun(@rdivide,vol,(2./(3*l)) .* s(:,[2 3 1 4 4 4]) .* s(:,[3 1 2 1 2 3]));
%% Using sin for dihedral angles gets into trouble with signs
%theta = asin(sin_theta);
%theta/pi*180
% http://arxiv.org/pdf/1208.0354.pdf Page 18
C = 1/6 * l .* cos_theta ./ sin_theta;

if nargin > 2
    C = C ./ Sigma(:,ones(1,6));
end

function [vol] = volume_intrinsic(l)
  u = l(:,1); v = l(:,2); w = l(:,3); 
  U = l(:,4); V = l(:,5); W = l(:,6); 
  X = (w - U + v).*(U + v + w);
  x = (U - v + w).*(v - w + U);
  Y = (u - V + w).*(V + w + u);
  y = (V - w + u).*(w - u + V);
  Z = (v - W + u).*(W + u + v);
  z = (W - u + v).*(u - v + W);
  a = sqrt(x.*Y.*Z);
  b = sqrt(y.*Z.*X);
  c = sqrt(z.*X.*Y);
  d = sqrt(x.*y.*z);
  vol = sqrt( ...
    (-a + b + c + d).* ...
    ( a - b + c + d).* ...
    ( a + b - c + d).* ...
    ( a + b + c - d))./ ...
    (192.*u.*v.*w);


function dblA = doublearea_intrinsic(l)
  % DOUBLEAREA_INTRINSIC Compute the double area of the triangles of a mesh
  % dblA = doublearea_intrinsic(l)
  % Inputs:
  %   l  #F by 3, array of edge lengths of edges opposite each face in F
  % Outputs:
  %   dblA   #F list of twice the area of each corresponding face
  % Copyright 2011, Alec Jacobson (jacobson@inf.ethz.ch), and Daniele Panozzo
  %

  l = sort(l,2,'descend');
  l1 = l(:,1); l2 = l(:,2); l3 = l(:,3);
  % Kahan's assertion: "Miscalculating Area and Angles of a Needle-like
  % Triangle" Section 2.
  % http://www.cs.berkeley.edu/~wkahan/Triangle.pdf
%   if any(l3-(l1-l2)<0)
%     warning( 'Failed Kahan''s assertion');
%   end
  %% semiperimeters
  %s = (l1 + l2 + l3)*0.5;
  %% Heron's formula for area
  %dblA = 2*sqrt( s.*(s-l1).*(s-l2).*(s-l3));
  % Kahan's heron's formula
  dblA = 2*0.25*sqrt((l1+(l2+l3)).*(l3-(l1-l2)).*(l3+(l1-l2)).*(l1+(l2-l3)));




