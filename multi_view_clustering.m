%% 设置路径和超参数  
clear; clc;  
addpath('measure', 'tools', 'gspbox-0.7.0');  

resultdir = 'Results/';  
if (~exist(resultdir, 'dir'))  
    mkdir(resultdir);  
    addpath(genpath(resultdir));  
end  
  
logdir = 'exp_log/';  
if (~exist(logdir, 'dir'))  
    mkdir(logdir);  
end  
  
datadir = 'datasets/';  
dataname = {'bbcsport-4view'};  
del = [0.1]; % del = [0.1, 0.3, 0.5, 0.7];  
lamda1 = [2e-15, 2e-13, 2e-11, 2e-9, 2e-7, 2e-5, 2e-3, 2e-1, 2e1, 2e3, 2e5, 2e7, 2e9, 2e11, 2e13, 2e15];  
lamda2 = [2e-15, 2e-13, 2e-11, 2e-9, 2e-7, 2e-5, 2e-3, 2e-1, 2e1, 2e3, 2e5, 2e7, 2e9, 2e11, 2e13, 2e15];  
  
% 获取当前脚本名称并创建日志文件夹  
[~, scriptName, ~] = fileparts(mfilename('fullpath'));  
logSubdir = fullfile(logdir, scriptName);  
if (~exist(logSubdir, 'dir'))  
    mkdir(logSubdir);  
end  
  
ResBest = [];  
ResStd = [];  
  
for idata = 1:length(dataname)  
    % 构建数据文件路径  
    datafile = fullfile(datadir, [char(dataname(idata)), '.mat']);  
    disp(['Loading data file: ', datafile]);  
      
    % 遍历每个缺失比例  
    for perMising = 1:length(del)  
        datafolds = fullfile(datadir, [char(dataname(idata)), '_Per', num2str(del(perMising)), '.mat']);  
        disp(['Loading data fold file: ', datafolds]);  
        load(datafolds);  
          
        if exist('Y', 'var')  
            Tlable = Y;  
            numclass = length(unique(Tlable));  
            k = numclass;  
            N = size(X{1}, 2);  
            numview = length(X);  
            fold = folds;  
            X1 = cell(length(X), 1);  
            index = cell(length(X), 1);  
  
            % 创建日志文件  
            logFileName = fullfile(logSubdir, [datestr(now, 'YYYY-mm-DD-HH-MM'), '-', char(dataname(idata)), '.txt']);  
            fid = fopen(logFileName, 'a');  
              
            for i = 1:length(lamda1)  
                for j = 1:length(lamda2)  
                    % 预处理数据  
                    for iv = 1:length(X)  
                        index{iv} = find(fold{1}(:, iv) == 1);  
                        X1{iv} = NormalizeData(X{iv});  
                        ind_0 = find(fold{1}(:, iv) == 0);  
                        X1{iv}(:, ind_0) = 0;  
                    end  
                      
                    % 打印日志  
                    fprintf(fid, '%s_Per%0.1f lamda1=%e lamda2=%e\n', char(dataname(idata)), del(perMising), lamda1(i), lamda2(j));  
                      
                    % 初始化参数  
                    [A, np] = constructA(X, index);  
                    [Z0, Z, Q1, M1, G, G0, G1, W, W0, W1, E, E0, Y, tensor_G, tensor_W, tensor_Z, beta, U, Q, M] = initializeParams(X, A, np, numview, N, numclass);  
                      
                    mu1 = 10e-5; max_mu1 = 10e10; pho_mu1 = 2;  
                    mu2 = 10e-5; max_mu2 = 10e10; pho_mu2 = 2;  
                    mu3 = 10e-5; max_mu3 = 10e10; pho_mu3 = 2;  
                    iter = 0; start = 1; tic;  
                    epson = 1e-7; Isconverg = 0;  
                      
                    while (Isconverg == 0)  
                        iter = iter + 1;  
                        fprintf(fid, '----processing iter %d--------\n', iter);  
                          
                        % 更新L^k  
                        if start == 1  
                            Weight = constructW_PKN((abs(U) + abs(U')) / 2, 3);  
                            Diag_tmp = diag(sum(Weight));  
                            L = Diag_tmp - Weight;  
                        else  
                            P = (abs(abs(U) + abs(U'))) / 2;  
                            param.k = 3;  
                            HG = gsp_nn_hypergraph(P', param);  
                            L = HG.L;  
                        end  
                        start = 0;  
                          
                        % 更新Z0{i}  
                        for iz = 1:numview  
                            beta(iz) = 1 / (2 * norm(Z0{iz} - A{iz}' * U * A{iz}, 'fro') + eps);  
                            Q1{iz} = Q - A{iz} * A{iz}' * Q * A{iz} * A{iz}';  
                            M1{iz} = M - A{iz} * A{iz}' * M * A{iz} * A{iz}';  
                            G1{iz} = G{iz} - A{iz} * A{iz}' * G{iz} * A{iz} * A{iz}';  
                            W1{iz} = W{iz} - A{iz} * A{iz}' * W{iz} * A{iz} * A{iz}';  
                            Y1{iz} = (mu1 * Q1{iz} - M1{iz} + mu3 * G1{iz} - W1{iz});  
                            Z0{iz} = ((2 * beta(iz) + mu3) * eye(np(iz)) + mu2 * A{iz}' * X{iz}' * X{iz} * A{iz}) \ (A{iz}' * (mu2 * X{iz}' * X{iz} + 2 * beta(iz) * U + mu3 * G{iz} - W{iz}) * A{iz} + A{iz}' * X{iz}' * (Y{iz} - mu2 * E0{iz}));  
                        end  
                          
                        % 更新U  
                        sumU = 0;  
                        for iu = 1:numview  
                            U1p = (2 * numview * beta(iu) * Z0{iu} + A{iu}' * (mu1 * Q - M) * A{iu}) / (2 * numview * beta(iu) + mu1);  
                            U2p = (mu1 * Q1{iu} - M1{iu} + numview * (mu3 * G1{iz} - W1{iz})) / (mu1 + numview * mu3);  
                            Up = A{iu} * U1p * A{iu}' + U2p;  
                            sumU = sumU + Up;  
                        end  
                        U = sumU / numview;  
                          
                        % 更新tensor_Z  
                        for ii = 1:numview  
                            ZpU{ii} = A{ii} * (Z0{ii} - A{ii}' * U * A{ii}) * A{ii}' + U;  
                            tensor_Z(:, :, ii) = ZpU{ii};  
                        end  
                        z = tensor_Z(:);  
                          
                        % 更新E  
                        F = [];  
                        for k = 1:numview  
                            tmp = X{k} * A{k} - X{k} * A{k} * Z0{k} + Y{k} / mu2;  
                            F = [F; tmp * A{k}'];  
                        end  
                        [Econcat] = solve_l1l2(F, lamda1(i) / mu2);  
                        start = 1;  
                        for k = 1:numview  
                            E{k} = Econcat(start:start + d(k) - 1, :);  
                            E0{k} = E{k} * A{k};  
                            start = start + d(k);  
                        end  
                          
                        % 更新Q  
                        Q = (mu1 * U + M) / (mu1 * eye(numsample) + 2 * lamda2(j) * L);  
                          
                        % 更新tensor_G  
                        w = tensor_W(:);  
                        [g, objv] = wshrinkObj(z + 1 / mu3 * w, 1 / mu3, sx, 0, 1);  
                        tensor_G = reshape(g, sx);  
                          
                        % 更新M, Y  
                        M = M + mu1 * (U - Q);  
                        for im = 1:numview  
                            Y{im} = Y{im} + mu2 * (X{im} * A{im} - X{im} * A{im} * Z0{im} - E0{im});  
                        end  
                          
                        % 更新tensor_W  
                        w = w + mu3 * (z - g);  
                        tensor_W = reshape(w, sx);  
                          
                        % 更新mu  
                        mu1 = min(mu1 * pho_mu1, max_mu1);  
                        mu2 = min(mu2 * pho_mu2, max_mu2);  
                        mu3 = min(mu3 * pho_mu3, max_mu3);  
                          
                        % 记录迭代信息  
                        history.objval(iter + 1) = objv;  
                          
                        % 收敛条件  
                        Isconverg = 1;  
                        for ic = 1:numview  
                            if (norm(X{ic} * A{ic} - X{ic} * A{ic} * Z0{ic} - E0{ic}, inf) > epson)  
                                history.norm_Z0 = norm(X{ic} * A{ic} - X{ic} * A{ic} * Z0{ic} - E0{ic}, inf);  
                                fprintf(fid, '    norm_Z0 %7.10f    ', history.norm_Z0);  
                                Isconverg = 0;  
                            end  
                            if (norm(Z{ic} - G{ic}, inf) > epson)  
                                history.norm_Z_G = norm(Z{ic} - G{ic}, inf);  
                                fprintf(fid, 'norm_Z_G %7.10f    \n', history.norm_Z_G);  
                                Isconverg = 0;  
                            end  
                        end  
                        if (iter > 200)  
                            Isconverg = 1;  
                        end  
                    end  
                      
                    S1 = abs(U) + abs(U');  
                    C1 = SpectralClustering(S1, k);  
                    res = Clustering8Measure(Tlable, C1);  
                    time = toc;  
                    fprintf(fid, 'runtime: %f\n', time);  
                    ResBest = [ResBest; lamda1(i), lamda2(j), res];  
                end  
                save(fullfile(resultdir, [char(dataname(idata)), '_Per', num2str(del(perMising)), '_result.mat']), "ResBest", "ResStd");  
            end  
            fclose(fid);  
        end  
    end  
end  
  
function [Z0, Z, Q1, M1, G, G0, G1, W, W0, W1, E, E0, Y, tensor_G, tensor_W, tensor_Z, beta, U, Q, M] = initializeParams(X, A, np, numview, N, numclass)  
    numsample = size(X{1}, 2);  
    sx = [numsample, numsample, numview];  
    Z0 = cell(numview, 1);  
    Z = cell(numview, 1);  
    Q1 = cell(numview, 1);  
    M1 = cell(numview, 1);  
    G = cell(numview, 1);  
    G0 = cell(numview, 1);  
    G1 = cell(numview, 1);  
    W = cell(numview, 1);  
    W0 = cell(numview, 1);  
    W1 = cell(numview, 1);  
    E = cell(numview, 1);  
    E0 = cell(numview, 1);  
    Y = cell(numview, 1);  
    tensor_G = zeros(sx);  
    tensor_W = zeros(sx);  
    tensor_Z = zeros(sx);  
    beta = zeros(1, numview);  
    U = zeros(numsample, numsample);  
    Q = zeros(numsample, numsample);  
    M = zeros(numsample, numsample);  
      
    for ii = 1:numview  
        Z0{ii} = eye(np(ii));  
        Z{ii} = eye(numsample, numsample);  
        G{ii} = tensor_G(:, :, ii);  
        W{ii} = tensor_W(:, :, ii);  
        E{ii} = zeros(size(X{ii}, 1), numsample);  
        E0{ii} = zeros(size(X{ii}, 1), np(ii));  
        Y{ii} = zeros(size(X{ii}, 1), np(ii));  
    end  
end  