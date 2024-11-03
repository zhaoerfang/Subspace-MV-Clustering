clear; clc; 
addpath('measure', 'tools', 'gspbox-0.7.0'); 

% dataset config  
resultdir = 'Results';  
datadir = 'datasets/';  

% below is the list of datasets to run

% datanames = {'bbcsport-4view', '3sources-3view', 'BDGP_4view', 'Caltech101-7_6view', 'handwritten-5view', 'NH_3view', 'ORL_3view'};  

datanames = {'bbcsport-4view', '3sources-3view'};  % this is a unit test datasets list

%below is the validation ratio to be used

% del = [0.1, 0.3, 0.5, 0.7]

del = [0.1, 0.3];  % this is a unit test ratio list

%below is the reagularization parameter to be used

% lamda1 = [2e-15, 2e-13, 2e-11, 2e-9, 2e-7, 2e-5, 2e-3, 2e-1, 2e1, 2e3, 2e5, 2e7, 2e9, 2e11, 2e13, 2e15];  
% lamda2 = [2e-15, 2e-13, 2e-11, 2e-9, 2e-7, 2e-5, 2e-3, 2e-1, 2e1, 2e3, 2e5, 2e7, 2e9, 2e11, 2e13, 2e15];  
  
lamda1 = [2e-15, 2e-13];  % this is a unit test ratio list
lamda2 = [2e-15, 2e-13];  

currentDateTime = datestr(now, 'yyyy-mm-dd-HH-MM');  
  
for idata = 1:length(datanames)  % traverse dataset
    datasetName = datanames{idata};  
    datasetDir = fullfile(resultdir, datasetName);  
    if (~exist(datasetDir, 'dir'))  
        mkdir(datasetDir);  
    end  

    for perMising = 1:length(del)  % traverse validation ratios
        ratio = del(perMising);  
        resultSubDir = fullfile(datasetDir, ['result-', currentDateTime, '_Per', num2str(ratio)]);  
        if (~exist(resultSubDir, 'dir'))  
            mkdir(resultSubDir);  
        end  
        
        % config of saving dir
        readmeFile = fullfile(resultSubDir, 'readme.md');  
        fileID = fopen(readmeFile, 'w');  
        fprintf(fileID, 'Running Time: %s\n\n', currentDateTime);  
        fprintf(fileID, 'Results for dataset: %s\n\n', datasetName);  
        fprintf(fileID, 'Hyperparameters:\n');  
        fprintf(fileID, 'del: %s\n', mat2str(ratio));  
        fprintf(fileID, 'lamda1: %s\n', mat2str(lamda1));  
        fprintf(fileID, 'lamda2: %s\n', mat2str(lamda2));  
        fclose(fileID);  
  
        addpath(genpath(resultSubDir));  
        
        % implementation of the algorithm
        result = runAlgorithm(datasetName, ratio, datadir, lamda1, lamda2);
        
        % saving the result
        saveFileName = fullfile(resultSubDir, [char(datasetName), '_Per', num2str(ratio), '_result.mat']);  
        save(saveFileName, 'result');  

        % statistc of the top 10 results
        avg_performance = mean(result(:, 3:end), 2); 
        [sorted_performance, sorted_indices] = sort(avg_performance, 'descend'); 
        num_top_results = min(10, length(sorted_indices));
        top_indices = sorted_indices(1:num_top_results);
        top_results = result(top_indices, :);
  
        fileID = fopen(readmeFile, 'a');  
        fprintf(fileID, '\nTop %d Hyperparameter Combinations and their Performance:\n', num_top_results);  
        fprintf(fileID, 'lamda1\tlamda2\tFscore\tPrecision\tRecall\tnmi\tAR\tEntropy\tACC\tPurity\n');  
        for i = 1:num_top_results  
            fprintf(fileID, '%e\t%e\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n', top_results(i, 1), top_results(i, 2), top_results(i, 3:end));  
        end  
        fclose(fileID);  

    end  
end  