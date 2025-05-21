function DImg = Conv(DImg, ConvN)
    % Convolution on density image in order to obtain a smoother 
    % density image
    % Input:
    %   DImg (3d double tensor): Density image
    %   ConvN (odd int): Convolution window size
    % Output:
    %   DImg (3d double tensor): Density image after convolution
    DImg = imfilter(DImg, ones(ConvN,ConvN,ConvN)/ConvN^3, 1,...
                    'same', 'conv');
end