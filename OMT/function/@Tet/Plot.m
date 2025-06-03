function Plot(T, V)
ip = unique(T(:));
dc = max(V(ip,1),[],1) - min(V(ip,1),[],1);
% dc = max(V(ip,:),[],1) - min(V(ip,:),[],1);
[dd,id] = min( dc) ;
ok = false(size(V,1),1);
ok(ip) = V(ip,id) < mean(V(ip,id)) + .20*dd;
ti = all(ok(T),2);

ec = [.20,.20,.20];
ei = [.25,.25,.25];
fe = [138, 241, 255]/255;
fi = [.95,.95,.50];

F1 = BoundaryFace(T( ti,:));
F2 = BoundaryFace(T(~ti,:));
c1 = ismember(sort(F1,2), sort(F2,2),'rows'); % common facets
c2 = ismember(sort(F2,2), sort(F1,2),'rows');

% draw external surface
patch('faces', F1(~c1,:), 'vertices', V, 'facecolor', fe,...
      'edgecolor', ec, 'linewidth', 0.67, 'facealpha', 1);
% draw internal surface
patch('faces', F1( c1,:), 'vertices', V, 'facecolor', fi,...
      'edgecolor', ei, 'linewidth', 0.67, 'facealpha', 1);
% draw transparent part
patch('faces', F2(~c2,:), 'vertices', V, 'facecolor', fe,...
      'edgecolor', 'none', 'linewidth', 0.67, 'facealpha', 0.2);
axis equal off

function Fb = BoundaryFace(T)
F = [T(:,[1,2,3]); T(:,[1,4,2]); T(:,[2,4,3]); T(:,[3,4,1])] ; 
[~, ii, jj] = unique (sort(F,2),'rows');
F = F(ii,:);
ss = accumarray(jj, ones(size(jj)),[size(F,1),1]) ;
Fb = F(ss==+1,:);
