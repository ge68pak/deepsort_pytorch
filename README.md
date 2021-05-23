# Deep SORT

下载MOT16 MOT17
把yolo_deepsort_pytorch中生成的XXXckpt.t7复制到tools/checkpoint

deep_sort_app.py 修改create_detections里面 feature的选取位置

tools/generate_detections_pytorch.py 修改mot_dir,output_dir,Extractor(model + XXX.t7)

(bbox->bbox+feature->detections.npy)

evaluate_motchallenge.py 修改mot_dir,detection_dir,output_dir (detections.npy -> deepsort结果 predict.txt)

eval.py 修改result_filename,seqs_str,data_root (gt.txt + predict.txt -> MOTA,IDs) 

