function Mask = Dilate(Mask, DilateN)
    % Enlarge the prediction mask by dilation
    % Input:
    %   Mask (3d logical tensor): prediction mask
    %   DilateN (odd int): The size of structure element for dilation
    % Output:
    %   Mask (3d logical tensor): Mask after dilation
    
    Mask = imdilate(Mask, strel('cube', DilateN));
end