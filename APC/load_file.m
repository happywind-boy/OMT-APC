function [] = load_file(folderPath, mode)
addpath(' ');
fileList = dir(fullfile(folderPath));
F = numel(fileList);
for ev = 15
    X = zeros(384*ev,F);
    for i = 3:F
        filePath = fullfile(folderPath, fileList(i).name, [fileList(i).name '.nii.gz']);
        Q = niftiread(filePath);
        Q = double(Q);
        Q = Q(:,:,:,mode); 
        Q = ImgNormalize(Q);
        [n1, n2, n3] = size(Q);
        time_HOSVD = tic;
        [S1, S2, S3, U1, U2, U3] = HOSVD(Q,ev);
        fprintf('运行到第%d个文件\n', i);
        U_1 = U1 * S1;
        U_2 = U2 * S2;
        U_3 = U3 * S3;
        X(:,i) = [U_1(:);U_2(:);U_3(:)];
        time_HOSVD = toc(time_HOSVD);
    end    
    filename = strcat('1p19q-11-12_X_rank',num2str(ev),...
        '_t',num2str(mode),'.mat');
    save(filename, 'X')
end
end


