classdef Density
    properties (Constant)
        DensityType = {
            'VP', ...
            'EXP-HE-FLAIR', ...
            'Exp-FLAIR', ...
            'BINARY'
        };
    end
    
    methods (Static, Access = private)
        DImg = Binary(Eps, Mask);
        DImg = Exp_Flair(Img, r, Mask);
        DImg = Exp_HE_Flair(Img, r, Mask);
        DImg = VP(Img);

        Mask = Dilate(Mask, DilateN);
        DImg = Conv(DImg, ConvN);
    end

    methods (Static)
        DImg = GenerateDImg(Img, Mask, DensityName, DensityParam,...
                            DilateN, ConvN);
        Valid = isValidName(DensityName);
    end
end