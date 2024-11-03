# Subspace-MV-Clustering
Xudan's postgraduate: Subspace Representation-Based Multi-View Subspace Clustering

## Recent changes

- [2024 Nov 03] 
  - `separate main script and algorithm implementation` now the entry of the whole project is `main.m`, and the algorithm implementation is in `runAlgorithm.m`.
  - `run different datasets` with setting different datasets and validation ratio, the results will be saved in the relevant folder with a brief description of the which.

- [2024 Nov 02] 
  - `fixed passing lambda` passing lambda to `solve_l1l2` should be a scaler, not a vector.
  - `saving results` the exp results can be saved in accordance with the `dataset` and `current time`.
  - `file arranged` the modified script is `run.m`, i.e the main script, but keeping the original one `untitled41.m`

## git相关

### git 追踪
1. 应该只track核心代码文件，如实验脚本、脚本的外部依赖等，如`tools`、`measure`等；
2. 数据集应该不track，如`datasets`、`results`里的`*.mat`文件等，因为它们属于`large file`，应该有专门的存储方式进行管理，可以考虑使用`onedrive`云盘进行管理；
   > 目前已经将所有的`*.mat`文件停止追踪，在`.gitignore`中有所配置。

### git管理
1. 最好使用分支进行管理，如xudan就在`xudan_dev`分支上进行操作，具体实现为:
```
git clone https://github.com/zhaoerfang/Subspace-MV-Clustering.git
git checkout -b xudan_dev
```
2. 好处是：你可以在你的分支上进行算法实现，我可以在我的分支上对你的算法实现进行代码重构和优化；即你可以专注于算法，而暂不考虑代码结构。

### git分支
1. 在克隆仓库时，git会将所有分支和标签克隆到本地仓库，但只会创建默认分支（`master`），而不会为其他远程分支创建本地工作副本；
2. 需要手动创建其他分支，如：`git checkout fze_dev`；
3. 完整的操作流程如下，这样即可从远程的`fze_dev`分支创建`xudan_dev`：
```
git clone https://github.com/zhaoerfang/Subspace-MV-Clustering.git
git checkout fze_dev
git checkout -b xudan_dev
```
4. 如果想推送本地分支到远程，则使用以下命令
```
第一次推送
git push -u origin xudan_dev  
```
```
之后推送
git pull origin xudan_dev  // 先拉取，再推送
git pull origin xudan_dev  
``` 

## how to use
1. 分离程序入口和算法实现，方便代码调试和重构。
  - `main.m`为程序入口，数据集的选择和超参数的设定在这里配置，如下所示
  ```matlab
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
  ```

  - `runAlgorithm.m`为算法实现，如下所示：
```matlab
function result = runAlgorithm(datasetName, ratio, datadir, lamda1, lamda2)

% dataloader
datafile = fullfile(datadir, [char(datasetName), '.mat']);
disp(['Loading data file: ', datafile]);
load(datafile);

datafolds = fullfile(datadir, [char(datasetName), '_Per', num2str(ratio), '.mat']);
disp(['Loading data fold file: ', datafolds]);
load(datafolds);

% algorithm implementation

```

2. 实验结果自动保存和预览
- 现在实验结果会根据`数据集名称`,`时间戳`以及`del ratio`创建子文件夹，**mat结果**和**实验配置及简单的描述**会放在此子文件夹下，如下所示
```shell
  Results
├── 3sources-3view # 数据集名称
│   ├── result-2024-11-03-10-06_Per0.1 # 实验时间戳-del ratio
│   │   ├── 3sources-3view_Per0.1_result.mat # mat结果
│   │   └── readme.md # 超参配置以及简单的top10结果预览
│   └── result-2024-11-03-10-06_Per0.3
│       ├── 3sources-3view_Per0.3_result.mat
│       └── readme.md
└── bbcsport-4view
    ├── result-2024-11-03-10-06_Per0.1
    │   ├── bbcsport-4view_Per0.1_result.mat
    │   └── readme.md
    └── result-2024-11-03-10-06_Per0.3
        ├── bbcsport-4view_Per0.3_result.mat
        └── readme.md
```
- 实验结果中的`readme.md`预览，包括实验时间、数据集、超参设置和top结果。
```md
Running Time: 2024-11-03-10-06

Results for dataset: bbcsport-4view

Hyperparameters:
del: 0.3
lamda1: [2e-15 2e-13]
lamda2: [2e-15 2e-13]

Top 4 Hyperparameter Combinations and their Performance:
lamda1	lamda2	Fscore	Precision	Recall	nmi	AR	Entropy	ACC	Purity
2.000000e-15	2.000000e-15	0.632328	0.585353	0.687500	0.568782	0.501987	0.975847	0.784483	0.784483
2.000000e-15	2.000000e-13	0.632328	0.585353	0.687500	0.568782	0.501987	0.975847	0.784483	0.784483
2.000000e-13	2.000000e-15	0.632328	0.585353	0.687500	0.568782	0.501987	0.975847	0.784483	0.784483
2.000000e-13	2.000000e-13	0.632328	0.585353	0.687500	0.568782	0.501987	0.975847	0.784483	0.784483

```

## road map
此模块为计划列表，如算法更新计划等

- [ ] 分离程序入口和配置模块
- [ ] 整理仓库其他杂项。
- [x] 创建脚本文件，运行多个数据集，并记录log@fze
- [x] 控制台输出重定向到日志@fze

## misc
此模块为杂项模块，如其他建议等内容

1. 比如我新建了一个`exp_log`文件夹，用来完成将每次实验打印的结果存入这个文件夹中，即将`terminal（控制台）`里的输出记录在这个日志文件夹里，这样就可以持久化记录日志，很方便的找到之前的实验记录。