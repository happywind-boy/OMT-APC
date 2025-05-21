function DImg = Binary(Eps, Mask)
    DImg = double(Mask);
    DImg(DImg == 0) = Eps;
end