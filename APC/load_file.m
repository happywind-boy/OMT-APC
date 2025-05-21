function [] = load_file(folderPath, mode)
addpath('D:\python_01\tensor_feature');
% folderPath  = "E:\hyd\Medical_image\tensor\data\Data\TCGA\IDH-training";
% 获取文件夹中的所有文件
% folderPath  = "E:\hyd\Medical_image\tensor\data_paper\Raw_Crop_train";
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
        %             X = zeros((n1+n2+n3)*ev,F);
        % Compute the high order SVD of a three-way tensor
        time_HOSVD = tic;
        [S1, S2, S3, U1, U2, U3] = HOSVD(Q,ev);
        fprintf('运行到第%d个文件\n', i);
        %         X(1:n1,(i*ev-(ev-1)):(i*ev)) = U1(:,1:ev);
        %         X(n1+1:2*n1,(i*ev-(ev-1)):(i*ev)) = U2(:,1:ev);
        %         X((2*n1+1):3*n1,(i*ev-(ev-1)):(i*ev)) = U3(:,1:ev);
        U_1 = U1 * S1;
        U_2 = U2 * S2;
        U_3 = U3 * S3;
        X(:,i) = [U_1(:);U_2(:);U_3(:)];
        %         S_ev(1:ev,i) =  diag(S1(1:ev, 1:ev));
        %         S_ev(ev+1:2*ev,i) =  diag(S2(1:ev, 1:ev));
        %         S_ev(2*ev+1:3*ev,i) =  diag(S3(1:ev, 1:ev));
        time_HOSVD = toc(time_HOSVD);
    end    
    %     if flag
    %         X = X - 1/N *(X*(ones(F,1))) * (ones(F,1))';
    %     else
    %         X = X;  % 不中心化
    %     end
    %     filename = strcat('IDH-test-weival-weitest_X_rank',num2str(ev),...
    %         '_t',num2str(mode) ,'.mat');
    %     filename = strcat('1p19q-weitest_X_rank',num2str(ev),...
    %         '_t',num2str(mode) ,'.mat');
    %       filename = strcat('IDH-guan_image_train_X_rank',num2str(ev),...
    %           '_t',num2str(mode) ,'ce','.mat');
    filename = strcat('1p19q-11-12_X_rank',num2str(ev),...
        '_t',num2str(mode),'.mat');
    save(filename, 'X')
end
end


