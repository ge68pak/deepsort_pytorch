# Deep SORT

下载MOT16 MOT17 放到deepsort_pytorch

注意在MOT17/train 新建3个文件夹DPM,FRCNN,SDP并把对应的序列放入

## Training the RE-ID model

1. dataset Market1501 Preparation 

下载Market1501数据集  运行data/Market_prepare.py (只需在第5行修改路径)

把生成的pytorch文件夹copy到~/deepsort_pytorch/tools 并重命名为data_market1501

2. dataset Mars Preparation 

下载Mars数据集到~/deepsort_pytorch/tools 并重命名为data_mars 

3. 运行tools/train.py（从头开始训练）  或者tools/train.py --resume （迁移学习训练）

注意需要修改

12-17行 选择需要的Net

21-22行 选择需要的数据集路径 market1501/mars

40-41行  如果选择market1501

42-43行 如果选择mars

70-72行  如果迁移学习 修改checkpoint路径

152行  修改打印信息

161行   修改save路径 

188行   修改保存jpg名称

## Generate Detections (bbox->bbox+feature->detections.npy)

先确定所用detector的类型 和 所用Net的类型  

运行tools/generate_detections_pytorch.py 

注意需要修改

6行 进入Extractor进而进入tools/feature_extractor.py 7-12行选择需要的Net

74-75行 选择用CNN/HOG 提取特征

86行 如果用CNN 修改对应Net的checkpoint路径

100/108/120行 如果用HOG 修改维度信息

135行  mot_dir 改为 "../MOT17/train/XXX"或者"../MOT16/train"

136行  output_dir 改为希望 detections.npy保存的位置

## Tracking (detections.npy -> deepsort结果 predict.txt)

先确定所用detector的类型 

修改deep_sort_app.py中125-127行 feature的选取位置

运行evaluate_motchallenge.py

注意需要修改

 13行  mot_dir   "MOT17/train/XXX"或者"MOT16/train"
 
 15行   detection_dir   改为之前保存 detections.npy的位置

18行  output_dir   改为希望predict.txt保存的位置

52行  display可以选择True/False
 
 ## Evaluation (gt.txt + predict.txt -> MOTA,IDs) 

先确定所用detector的类型

运行eval.py

注意需要修改

32行 result_filename 改为之前保存predict.txt的位置

48行  是否保存成一个表格

67-99行  seqs_str  选择对应的detector的类型

104-107行  data_root 选择对应的detector的类型

