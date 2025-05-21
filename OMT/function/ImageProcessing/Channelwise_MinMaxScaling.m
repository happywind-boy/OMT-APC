function img_out = Channelwise_MinMaxScaling(img_in)
    % Perform channel-wise Min Max Scaling for the image
    % Input:
    %   img_in(array): image
    % Output:
    %   img_out(double array): image after channel-wise min-max scaling
    
    img_in  = double(img_in);
    img_min = min(img_in, [], [1,2,3]);
    img_max = max(img_in, [], [1,2,3]);
    img_out = (img_in - img_min) ./ (img_max - img_min);
end