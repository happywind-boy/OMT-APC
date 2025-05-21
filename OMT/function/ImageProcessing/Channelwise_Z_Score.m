function img_out = Channelwise_Z_Score(img_in)
    % Perform channelwise Z-score -> clipping -> min-max scaling on image
    % Input:
    %   img_in(array): image
    % Output:
    %   img_out(double array): image after applying the transforms above.
    
    img_in   = double(img_in);
    img_mean = mean(img_in, [1 2 3]);
    img_std  = std(img_in, 0, [1 2 3]);
    
    img_out = (img_in - img_mean) ./ img_std;
    
    img_max =  5;
    img_min = -5;
    
    img_out(img_out > img_max) = img_max;
    img_out(img_out < img_min) = img_min;
    img_out = (img_out - img_min) / (img_max - img_min);
end