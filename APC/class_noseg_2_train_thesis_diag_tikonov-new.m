clear;clc;
tic
% directory =  'E:\hyd\Medical_image\tensor\data\ALL_change\WHO_all_train';
% folderPath  = "E:\hyd\Medical_image\tensor\data\Data\train-val-test\training-weitest_chuntuxiang-weival_chuntuxiang";
folderPath  = "D:\thesis_data-fuben\DTI_MGMT\DTI_T2WI_train";
folderList = dir(folderPath);
folderList = folderList(3:end);
% 初始化图像数据容器
imageData = cell(1, length(folderList));
F = numel(imageData);
k = 15;
% n1 = 128;
n1 = 64;
feature = cell(k,3);
accuracy = zeros(k, 3);
sensitivity = zeros(k, 3);
specificity = zeros(k, 3);
auc_values = zeros(k, 3);
count = zeros(k,1);
result = zeros(k,1);
mode = 1;
for ev = 1:15
    N = F*ev;
    sorted_ev = 2;
%     filename = strcat('IDH-training-weival-weitest_X_rank',num2str(ev),...
%         '_t',num2str(mode),'.mat');
%     load('WHO-2728_X_rank15_t1ce.mat');    
     load('DTI_MGMT_thesis_train_511_T2WI_X_rank15.mat');
%      load('DCE_65_train_Vp_X_rank15.mat');
%     load('IDH-training-weival-weitest_X_rank30_t1ce.mat')
%     X = [X(1:128*ev, :); X(128*15+1:128*(15+ev), :); X(128*30+1:128*(30+ev), :)];
      X = [X(1:64*ev, :); X(64*15+1:64*(15+ev), :); X(64*30+1:64*(30+ev), :)];
    % xlsxFile = "E:\Users\liuxi\Desktop\WHO_grade\WHO-Grade.xlsx";
%     xlsxFile = "E:\hyd\Medical_image\tensor\data\Data\train-val-test\turn_training-weitest_chuntuxiang-weival_chuntuxiang.xlsx";
    xlsxFile = "D:\thesis_data-fuben\DTI_MGMT\MGMT_train_turn.xlsx";
    dataTable = readtable(xlsxFile);
    [lam_1, lam_2, lam_3] = deal(1e-4, 1e-4, 1e-4);
    % 获取名为 "IDH" 的列数据（第二行到最后一行）
    % idhData = dataTable.WHO_Grade;
%     idhData = dataTable.IDH;
    idhData = dataTable.MGMT_mythelation;
    Y = zeros(2,length(idhData));
    % 将列向量中的数据转换为one-hot二维向量
    for i = 1:length(idhData)
        if idhData(i) == 0
            Y(:, i) = [1; 0];
        elseif idhData(i) == 1
            Y(:, i) = [0; 1];
        end
    end
    % for i = 1:length(idhData)
    %     if idhData(i) == 2
    %         Y(:, i) = [1; 0];
    %     elseif idhData(i) == 3
    %         Y(:, i) = [1; 0];
    %     else idhData(i) == 4
    %         Y(:, i) = [0; 1];
    %     end
    % end
    Y = Y(:,1:F);
    % filename = 'E:\hyd\Medical_image\tensor论文\Y_egd.mat';
    % save(filename, 'Y');
    
    indices1 = find(Y(2,:) == 1);
    indices0 = find(Y(2,:) == 0);
    n_acc1 = numel(indices1);
    n_acc0 = numel(indices0);
    
    D = zeros(1,F);
    D(indices1) =  1/n_acc1;
    D(indices0) =  1/n_acc0;
    D = sqrt(D);
    D = diag(D);
%     D_inv = diag(1 ./ diag(D)); % 计算对角矩阵 D 的逆元素
    Y_D = Y * D;
    X_D = X * D;
    
    X_D_1 = X_D(1:n1*ev,:);
    X_D_2 = X_D(n1*ev+1:2*n1*ev,:);
    X_D_3 = X_D(2*n1*ev+1:3*n1*ev,:);
    A = (X_D*Y_D')*(Y_D*X_D');
    C_block = blkdiag(X_D_1*X_D_1',X_D_2*X_D_2', X_D_3*X_D_3');
    % 求解广义特征值问题
    I = eye(n1*ev);
    B = C_block + blkdiag(lam_1*I, lam_2*I, lam_3*I);
    [P, Lambda] = eigs(A, B,sorted_ev);
    V = A * P - C_block * P * Lambda;
    residuals = vecnorm(V);
    % [P, Lambda] = eig(A, C_block);
    % eigenvalues = diag(Lambda);
    %
    % % 对特征值进行排序
    % [sorted_eigenvalues, index] = sort(eigenvalues, 'descend');
    %
    % top_eigenvalues = sorted_eigenvalues(1:sorted_ev);
    % top_eigenvectors = P(:, index(1:sorted_ev));
    % P2 = top_eigenvectors;
    W = (P'* (C_block+ blkdiag(lam_1*I, lam_2*I, lam_3*I)) * P)\(P'* X_D * Y_D');
    %result = norm(Y - W'*P2'*X, 'fro')^2;
    % save('IDH-training_3_t2_tikonov_diag.mat', 'P', 'W');
%     filename_2 = strcat('NEWIDH-training-weival-weitest_P_W_lam1e-1_rank',num2str(ev),...
%         '_t',num2str(mode),'ce','.mat');
%     filename_2 = strcat('WHO-truetrain-weitest_P_W_lam1e-4_rank',num2str(ev),...
%       '_t',num2str(mode),'ce','.mat');
    filename_2 = strcat('DTI_MGMT_T2WI_train_511_P_W_lam1e-4_rank',num2str(ev),...
      '.mat');
%     filename_2 = strcat('DCE_Vp_train_65_P_W_lam1e-1_rank',num2str(ev),...
%       '.mat');
    save( filename_2 , 'P', 'W');
    Y_norm = norm(Y, 'fro')^2;
    for s = 1:3
        % 从矩阵 P 中获取第 s 行
        P_s = P((s-1)*n1*ev+1:s*n1*ev, :);
        % 从矩阵 X 中获取第 s 行
        X_s = X((s-1)*n1*ev+1:s*n1*ev, :);
        
        % 计算当前迭代的部分结果
        result(ev) = result(ev) + norm(Y - W' * (P_s.' * X_s), 'fro')^2/ Y_norm ;
        %     feature(2*s-1:2*s,:) = W'*P_s'*X_s;
        feature{ev,s} = W'*(P_s.' * X_s);
        
        V1 =  feature{ev,s};
        %      count(ev) = 0;
        %      for i = 1:size(V1,2);
        %        if any(V1(:,i) < 0)
        %          % 如果存在负数，则使用 softmax 转换
        %           count(ev) = count(ev)+1;
        %           V1(:,i) = exp(V1(:,i));
        %        end
        %       V1(:,i) = V1(:,i) ./ sum(V1(:,i), 1);
        %      end
        V1(1,:) = 1./(1+exp(-(V1(1,:)-(V1(1,:)+V1(2,:))/2)/0.2));
        V1(2,:) = 1 - V1(1,:);
        
        feature{ev,s} = V1;
    end
    
    % 存储转换后的数据
    binary_data = cell(1, 3);
    
    for i = 1:3
        data =  feature{ev,i};
        binary_data{1,i}(1,:) = data(1,:) >= 0.5;
        binary_data{1,i}(2,:) = ~binary_data{1,i}(1,:);
    end
    
    % 初始化一个向量，用于存储每个cell的ACC精度
    for i = 1:3
        data = binary_data{i};
        % 计算准确率
        correct_predictions = sum(data == Y, 'all');
        total_predictions = numel(data);
        accuracy(ev,i) = correct_predictions / total_predictions;
       [~, ~, ~, auc_values(ev,i)] = perfcurve(Y(2,:),feature{ev, i}(2, :),1);  % 计算 AUC
    end
    % 初始化一个向量，用于存储每个cell的sensitivity精度
    for i = 1:3
        data = binary_data{i};
        TP = sum(Y(2,:) == 1 & data(2,:) == 1);  % 计算 TP
        FN = sum(Y(2,:) == 1 & data(2,:) == 0);  % 计算 FN
        sensitivity(ev,i) = TP / (TP + FN);  % 计算 TPR
    end
    % 初始化一个向量，用于存储每个cell的specificity精度
    for i = 1:3
        data = binary_data{i};
        FP = sum(Y(2,:) == 0 & data(2,:) == 1);  % 计算 TP
        TN = sum(Y(2,:) == 0 & data(2,:) == 0);  % 计算 FN
        specificity(ev,i) = TN / (FP + TN);  % 计算 TPR
    end
    % 初始化一个向量，用于存储每个 cell 的 AUC 值

end
elapsed_time = toc;

% 输出运行时间
disp(['代码执行时间：', num2str(elapsed_time), '秒']);