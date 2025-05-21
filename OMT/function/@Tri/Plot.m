function P = Plot(F, V, U)
NF = Tri.Normal(F,V);
Vno = size(V,1);
e = ones(Vno,1);
rgb = [129/255, 159/255, 247/255];
Vrgb = rgb(e,:);
if exist('U', 'var')
    P = patch('Faces', F, 'Vertices', U, 'FaceVertexCData', Vrgb, 'EdgeColor','w','FaceColor','interp', 'EdgeAlpha', 0.5, 'EdgeLighting', 'flat');
else
    P = patch('Faces', F, 'Vertices', V, 'FaceVertexCData', Vrgb, 'EdgeColor','none','FaceColor','interp', 'EdgeAlpha', 0.5, 'EdgeLighting', 'flat');
    P.FaceLighting = 'phong';
end
P.FaceNormals = -NF;
camlight('headlight');
light('Position', [1,1,1]);
light('Position', -[1,1,1]);
set(gcf, 'color', [0 0 0]);
axis equal off