function DImg = GenerateDImg(Img, Mask, DensityName, DensityParam,...
                             varargin,DilateN, ConvN)
    p = inputParser;
    isOddInt = @(x) (mod(x,1) == 0) && (mod(x,2) == 1);
    addOptional(p, 'DilateN', 9, isOddInt);
    addOptional(p, 'ConvN', 5, isOddInt);
    parse(p);

    DilateN = p.Results.DilateN;
    ConvN   = p.Results.ConvN;
    
    if ~isempty(Mask) && ~isa(Mask, 'logical')
        error("Input mask should be a binary matrix or empty")
    end

    % Dilate
    if ~isempty(Mask) && exist('DilateN', 'var')
        Mask = Density.Dilate(Mask, DilateN);
    end

    % Define Density image
    switch (upper(DensityName))
        case 'VP'
            DImg = Density.VP(Img);
        case 'BINARY'
            DImg = Density.Binary(DensityParam, Mask);
        case 'Exp-Flair'
            DImg = Density.Exp_Flair(Img, DensityParam, Mask);
        case 'EXP-HE-FLAIR'
            DImg = Density.Exp_HE_Flair(Img, DensityParam, Mask);
        otherwise
            error("Not implemented");
    end

    % Convolution
    if ~isempty(Mask) && exist('ConvN', 'var')
        DImg = Density.Conv(DImg, ConvN);
    end
end