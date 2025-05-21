function L = Laplacian(T, V, Sigma)
Vno = size(V, 1);
EdgeLen = [ Vertex.Norm(V(T(:,4),:)-V(T(:,1),:)), ...
            Vertex.Norm(V(T(:,4),:)-V(T(:,2),:)), ...
            Vertex.Norm(V(T(:,4),:)-V(T(:,3),:)), ...
            Vertex.Norm(V(T(:,2),:)-V(T(:,3),:)), ...
            Vertex.Norm(V(T(:,3),:)-V(T(:,1),:)), ...
            Vertex.Norm(V(T(:,1),:)-V(T(:,2),:)) ];
s = 0.5*[ FaceArea2(EdgeLen(:,[2 3 4])), FaceArea2(EdgeLen(:,[1 3 5])), ...
          FaceArea2(EdgeLen(:,[1 2 6])), FaceArea2(EdgeLen(:,[4 5 6])) ];
H2 = (1/16) * (4*EdgeLen(:,[4 5 6 1 2 3]).^2.* EdgeLen(:,[1 2 3 4 5 6]).^2 - ...
             ((EdgeLen(:, [2 3 4 5 6 1]).^2 + EdgeLen(:,[5 6 1 2 3 4]).^2) - ...
             (EdgeLen(:, [3 4 5 6 1 2]).^2 + EdgeLen(:,[6 1 2 3 4 5]).^2)).^2);
cos_theta = (H2 - s(:,[2 3 1 4 4 4]).^2 - s(:,[3 1 2 1 2 3]).^2) ./ ...
            (-2*s(:,[2 3 1 4 4 4]).* s(:,[3 1 2 1 2 3]));
Vol = EdgeLen2Volume(EdgeLen);
Vol = abs(Vol);
sin_theta = bsxfun(@rdivide, Vol, (2./(3*EdgeLen)).*s(:,[2 3 1 4 4 4]).*s(:,[3 1 2 1 2 3]));
C = 1/6 * EdgeLen .* cos_theta ./ sin_theta;
if exist('Sigma', 'var')
    C = C ./ Sigma(:,ones(1,6));
end
C(abs(C)<10*eps) = 0;
L = sparse(T(:,[2 3 1 4 4 4]), T(:,[3 1 2 1 2 3]), C, Vno, Vno);
L = L + L.';
L = diag(sum(L,2)) - L;
end

function Vol = EdgeLen2Volume(EdgeLen)
u = EdgeLen(:,1); v = EdgeLen(:,2); w = EdgeLen(:,3); 
U = EdgeLen(:,4); V = EdgeLen(:,5); W = EdgeLen(:,6); 
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
Vol = sqrt( (-a + b + c + d).* (a - b + c + d) .* (a + b - c + d).* (a + b + c - d)) ./ (192.*u.*v.*w);
end

function FA2 = FaceArea2(EdgeLen)
EdgeLen = sort(EdgeLen,2,'descend');
E1 = EdgeLen(:,1);
E2 = EdgeLen(:,2);
E3 = EdgeLen(:,3);
s = 0.5*(E1 + E2 + E3);
FA2 = 2*sqrt( s.*(s-E1).*(s-E2).*(s-E3));
end