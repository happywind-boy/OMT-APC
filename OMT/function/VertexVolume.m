function [VertVol, Vol, Gvt] = VertexVolume(T, V)
Vno = size(V,1);
Tno = size(T,1);
Vol = TetVolume(T, V);
Tidx = repmat(1:Tno,1,4);
Gvt = sparse(T, Tidx, 1, Vno, Tno);
VertVol = 0.25*Gvt*Vol;