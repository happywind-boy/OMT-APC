clear;clc;
tic

folderPath  = " ";
folderList = dir(folderPath);
folderList = folderList(3:end);

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
    load('DTI_MGMT_thesis_train_511_T2WI_X_rank15.mat');
    X = [X(1:64*ev, :); X(64*15+1:64*(15+ev), :); X(64*30+1:64*(30+ev), :)];

    xlsxFile = "D:\thesis_data-fuben\DTI_MGMT\MGMT_train_turn.xlsx";
    dataTable = readtable(xlsxFile);
    [lam_1, lam_2, lam_3] = deal(1e-4, 1e-4, 1e-4);
    idhData = dataTable.MGMT_mythelation;
    Y = zeros(2,length(idhData));

    for i = 1:length(idhData)
        if idhData(i) == 0
            Y(:, i) = [1; 0];
        elseif idhData(i) == 1
            Y(:, i) = [0; 1];
        end
    end
    Y = Y(:,1:F);
    
    indices1 = find(Y(2,:) == 1);
    indices0 = find(Y(2,:) == 0);
    n_acc1 = numel(indices1);
    n_acc0 = numel(indices0);
    
    D = zeros(1,F);
    D(indices1) =  1/n_acc1;
    D(indices0) =  1/n_acc0;
    D = sqrt(D);
    D = diag(D);
    Y_D = Y * D;
    X_D = X * D;
    
    X_D_1 = X_D(1:n1*ev,:);
    X_D_2 = X_D(n1*ev+1:2*n1*ev,:);
    X_D_3 = X_D(2*n1*ev+1:3*n1*ev,:);
    A = (X_D*Y_D')*(Y_D*X_D');
    C_block = blkdiag(X_D_1*X_D_1',X_D_2*X_D_2', X_D_3*X_D_3');

    I = eye(n1*ev);
    B = C_block + blkdiag(lam_1*I, lam_2*I, lam_3*I);
    [P, Lambda] = eigs(A, B,sorted_ev);
    V = A * P - C_block * P * Lambda;
    residuals = vecnorm(V);
    W = (P'* (C_block+ blkdiag(lam_1*I, lam_2*I, lam_3*I)) * P)\(P'* X_D * Y_D');
    filename_2 = strcat('DTI_MGMT_T2WI_train_511_P_W_lam1e-4_rank',num2str(ev),...
      '.mat');
    save( filename_2 , 'P', 'W');
    Y_norm = norm(Y, 'fro')^2;
    for s = 1:3
        P_s = P((s-1)*n1*ev+1:s*n1*ev, :);
        X_s = X((s-1)*n1*ev+1:s*n1*ev, :);
        
        result(ev) = result(ev) + norm(Y - W' * (P_s.' * X_s), 'fro')^2/ Y_norm ;
        feature{ev,s} = W'*(P_s.' * X_s);
        
        V1 =  feature{ev,s};
        V1(1,:) = 1./(1+exp(-(V1(1,:)-(V1(1,:)+V1(2,:))/2)/0.2));
        V1(2,:) = 1 - V1(1,:);
        
        feature{ev,s} = V1;
    end
    binary_data = cell(1, 3);

    for i = 1:3
        data =  feature{ev,i};
        binary_data{1,i}(1,:) = data(1,:) >= 0.5;
        binary_data{1,i}(2,:) = ~binary_data{1,i}(1,:);
    end
    
    for i = 1:3
        data = binary_data{i};
        correct_predictions = sum(data == Y, 'all');
        total_predictions = numel(data);
        accuracy(ev,i) = correct_predictions / total_predictions;
       [~, ~, ~, auc_values(ev,i)] = perfcurve(Y(2,:),feature{ev, i}(2, :),1);  % 计算 AUC
    end

    for i = 1:3
        data = binary_data{i};
        TP = sum(Y(2,:) == 1 & data(2,:) == 1);  % 计算 TP
        FN = sum(Y(2,:) == 1 & data(2,:) == 0);  % 计算 FN
        sensitivity(ev,i) = TP / (TP + FN);  % 计算 TPR
    end

    for i = 1:3
        data = binary_data{i};
        FP = sum(Y(2,:) == 0 & data(2,:) == 1);  % 计算 TP
        TN = sum(Y(2,:) == 0 & data(2,:) == 0);  % 计算 FN
        specificity(ev,i) = TN / (FP + TN);  % 计算 TPR
    end

end
elapsed_time = toc;

disp(['代码执行时间：', num2str(elapsed_time), '秒']);
