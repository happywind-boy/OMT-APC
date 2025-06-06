clear;clc;
tic
% directory =  " ";
% folderPath  = " ";
addpath(" ");
folderPath  = " ";
% folderPath  = " ";
% folderPath  = " ";
% load(' ');
% load(' ');
% load(' ');
% load(' ');
% load(' ');

%% get file info
fileList = dir(fullfile(folderPath));
F = numel(fileList)-2;
k = 10;
n1 = 64;
feature = cell(k,3);
accuracy = zeros(k, 3);
sensitivity = zeros(k, 3);
specificity = zeros(k, 3);
count = zeros(k,1);
result = zeros(k,1);
mode = 2;
flag = 0;  
for ev = 10
    N = F*ev;
    sorted_ev = 2;
    X = zeros(384*ev,F);
    % 循环读取每个文件
    for i = 1:F
        filePath = fullfile(folderPath, fileList(i).name);
        Q = niftiread(filePath);
        Q = Q(:,:,:,1);
        Q = ImgNormalize(Q);
        [n1, n2, n3] = size(Q);
                    
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
        load('IDH-training-weival-weitest_X_rank30_t2.mat')
   
        load('IDH-guan_rawcrop_train+test_X_rank10_t2.mat')
        load('IDH-training-weival-weitest_X_rank30_t1ce.mat')
        X = [X(1:128*ev, :); X(128*30+1:128*(10+ev), :); X(128*20+1:128*(20+ev), :)];
        X = [X(1:80*ev, :); X(80*10+1:80*(10+ev), :); X(80*20+1:80*(20+ev), :)];
    X = [X(1:128*ev, :); X(128*30+1:128*(30+ev), :); X(128*60+1:128*(60+ev), :)];
    load('1p19q-11-13_X15_t2.mat');  
    X = [X(1:64*ev, :); X(64*15+1:64*(15+ev), :); X(64*30+1:64*(30+ev), :)];
    X_1 = X(1:n1*ev,:);
    X_2 = X(n1*ev+1:2*n1*ev,:);
    X_3 = X(2*n1*ev+1:3*n1*ev,:);
    filename_2 = strcat('NEW1p19q-truetrain-trueval-weitest_P_W_lam1e-4_rank',num2str(ev),...
        '_t',num2str(mode),'.mat');
    load( filename_2 , 'P', 'W');
    load('1p19q-11-13_X15_P_W_lam1e-4_rank10_t2.mat', 'P', 'W')
        filename_2 = strcat('IDH-guan_rawcrop_train_PW_rank',num2str(ev),...
            '_t',num2str(mode),'.mat');
        load( filename_2 , 'P', 'W');
        filename_2 = strcat('NEWIDH-training-weival-weitest_P_W_lam1e-4_rank',num2str(ev),...
            '_t',num2str(mode),'.mat');
        load( filename_2 , 'P', 'W');
    Y_norm = norm(Y, 'fro')^2;
    
    for s = 1:3
        P_s = P((s-1)*n1*ev+1:s*n1*ev, :);
                X_s = X((s-1)*n1*ev+1:s*n1*ev, :);
        result = result + norm(Y - W' * (P_s.' * X_s), 'fro')^2/ Y_norm ;
        feature(2*s-1:2*s,:) = W'*P_s'*X_s;
        feature{1,s} = W'*(P_s.' * X_s);
        feature{ev,s} = W'*(P_s.' * X_s);
        V1 =  feature{ev,s};
        count(ev) = 0;
        V1(1,:) = 1./(1+exp(-(V1(1,:)-(V1(1,:)+V1(2,:))/2)/0.2));
        V1(2,:) = 1 - V1(1,:);
        feature{ev,s} = V1;
       
    end

    outputData = [];
    for i = 1:3
        featureData = feature{ev,i};
        outputData = [outputData; featureData];
    end
    excelFile = '1p19q_rank10_t2_truetrainweitest-WPXtikonov-diag_1e-4.xlsx';
    xlswrite(excelFile, outputData);
end
