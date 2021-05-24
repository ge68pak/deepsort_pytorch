# Deep SORT

下载MOT16 MOT17

## Training the RE-ID model
dataset Market1501 Preparation 参考Person_ReID

把生成的pytorch文件夹copy到~/deepsort_pytorch/tools 并重命名为data_market1501

dataset Mars Preparation 

下载Mars数据集 并重命名为data_mars 修改子文件名称改为train和test

Then you can try train.py to train your own parameter and evaluate it using test.py and evaluate.py.

注意tools/train.py需要修改

import Net + dataset路径 + checkpoint路径 + save路径 + 保存jpg名称

从头开始训练python3 train.py

迁移学习训练python3 train.py --resume

## Generate Detections (bbox->bbox+feature->detections.npy)
deep_sort_app.py 修改create_detections里面 feature的选取位置

tools/generate_detections_pytorch.py 修改mot_dir,output_dir,Extractor(model + XXX.t7)

## Evaluation 

evaluate_motchallenge.py 修改mot_dir,detection_dir,output_dir (detections.npy -> deepsort结果 predict.txt)

eval.py 修改result_filename,seqs_str,data_root (gt.txt + predict.txt -> MOTA,IDs) 

