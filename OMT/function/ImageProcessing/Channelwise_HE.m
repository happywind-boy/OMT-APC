function img_out = Channelwise_HE(img_in)
    % Perform channel-wise histogram equalization (HE)
    % on img_in only on nonzero region
    %
    % Input:
    %   img_in(array): image
    % Output:
    %   img_out(double array): image after channel-wise HE
    
    img_in        = double(img_in);
    [nx,ny,nz,nd] = size(img_in);
    
    Idx     = find(mean(img_in, 4) > 0);
    img_in  = reshape(img_in, nx*ny*nz, nd);
    img_out = zeros(size(img_in)); 
    for k = 1:nd
        img_out(Idx, k) = histeq(img_in(Idx, k));
    end
    img_out = reshape(img_out, nx, ny, nz, nd);
end