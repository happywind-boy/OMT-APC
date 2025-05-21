function [Gray, Bdry] = Img2GrayFlair(Img, Raw, T, V, Bdry, VB, Bin)
Img = double(Img);
Img = ImgNormalize(Img);
ImgMean = Img(:,:,:,1);
Raw.Gray.V = ImgMean(:);
if exist('Bin', 'var')
    Raw.Gray.V(~Bin) = 0;
end
Gray.V = PiecewiseAffineMap(Raw.T, Raw.V, Raw.Gray.V, V);
Gray.V = Gray.V(:,1);

TC = TetCenter(T, V);
Gray.T = PiecewiseAffineMap(Raw.T, Raw.V, Raw.Gray.V, TC);
Gray.T = Gray.T(:,1);
Bdry.Gray.V = Gray.V(VB);

FC = FaceCenter(Bdry.F, Bdry.V);
Bdry.Gray.F = PiecewiseAffineMap(Raw.T, Raw.V, Raw.Gray.V, FC);
Bdry.Gray.F = Bdry.Gray.F(:,1);
