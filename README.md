# Subspace-MV-Clustering
Xudan's postgraduate: Subspace Representation-Based Multi-View Subspace Clustering

## Recent changes

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

## road map
此模块为计划列表，如算法更新计划等

- [ ] 创建脚本文件，运行多个数据集，并记录log@fze
- [x] 控制台输出重定向到日志@fze

## misc
此模块为杂项模块，如其他建议等内容

1. 比如我新建了一个`exp_log`文件夹，用来完成将每次实验打印的结果存入这个文件夹中，即将`terminal（控制台）`里的输出记录在这个日志文件夹里，这样就可以持久化记录日志，很方便的找到之前的实验记录。