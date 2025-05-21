function Img = ImgNormalize(Img)
ImgMean = mean(Img, [1,2,3]);
ImgStd = std(Img,0,[1 2 3]);
Img = (Img - ImgMean)./ImgStd;

rangeMin = -5;
rangeMax = 5;
Img(Img > rangeMax) = rangeMax;
Img(Img < rangeMin) = rangeMin;

% Rescale the data to the range [0, 1].
Img = (Img - rangeMin) / (rangeMax - rangeMin);
