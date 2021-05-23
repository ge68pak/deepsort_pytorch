import os
import sys
import getopt
import os.path as osp
import logging
import argparse
from pathlib import Path
import time
import numpy as np
from utils.log import get_logger
from utils.io import write_results
from utils.parser import get_config

import motmetrics as mm
mm.lap.default_solver = 'lap'             # lap: linear assignment problem
from utils.evaluation import Evaluator

def mkdir_if_missing(dir):
    os.makedirs(dir, exist_ok=True)

def main(data_root='', seqs=('',)):
    logger = get_logger()
    logger.setLevel(logging.INFO)
    data_type = 'mot'
    # result_root = os.path.join(Path(data_root), "mot_results")
    # mkdir_if_missing(result_root)

    # run tracking
    accs = []
    for seq in seqs:
        logger.info('start seq: {}'.format(seq))
        result_filename = "./results/test/"+seq+".txt"
        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    # Evaluator.save_summary(summary, os.path.join(result_root, 'summary_global.xlsx'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=False)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    # parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--save_path", type=str, default="./demo/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()

if __name__ == '__main__':

    # seqs_str = '''MOT16-02
    #               MOT16-04
    #               MOT16-05
    #               MOT16-09
    #               MOT16-10
    #               MOT16-11
    #               MOT16-13
    #               '''

    # seqs_str = '''MOT17-02-DPM
    #               MOT17-04-DPM
    #               MOT17-05-DPM
    #               MOT17-09-DPM
    #               MOT17-10-DPM
    #               MOT17-11-DPM
    #               MOT17-13-DPM
    #               '''
    # seqs_str = '''MOT17-02-FRCNN
    #               MOT17-04-FRCNN
    #               MOT17-05-FRCNN
    #               MOT17-09-FRCNN
    #               MOT17-10-FRCNN
    #               MOT17-11-FRCNN
    #               MOT17-13-FRCNN
    #               '''
    seqs_str = '''MOT17-02-SDP
                  MOT17-04-SDP
                  MOT17-05-SDP
                  MOT17-09-SDP
                  MOT17-10-SDP
                  MOT17-11-SDP
                  MOT17-13-SDP
                  '''
    # seqs_str = '''MOT16-02
    #               MOT16-09
    #               '''
    # data_root = 'data/dataset/MOT16/train/'
    # data_root = 'MOT16/train'
    # data_root = 'MOT17/train/DPM'
    # data_root = 'MOT17/train/FRCNN'
    data_root = 'MOT17/train/SDP'


    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root=data_root,
         seqs=seqs)
