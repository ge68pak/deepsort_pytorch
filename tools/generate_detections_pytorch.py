# vim: expandtab:ts=4:sw=4
import os
import errno
import numpy as np
import cv2
from feature_extractor import Extractor







def generate_detections(mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        # detection_file = os.path.join(
        #     detection_dir, sequence, "gt/gt.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            im = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            features = _get_features(rows[:, 2:6].copy(), im)
            # features = _get_features_hog_paper(rows[:, 2:6].copy(), im)   # HOG paper
            # features = _get_features_hog(rows[:, 2:6].copy(), im)       # HOG
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def _get_features(bbox_tlwh, ori_img):
        extractor = Extractor("./checkpoint/64_0_40_ckpt.t7", use_cuda=True)
        im_crops = []
        height, width = ori_img.shape[:2]
        for box in bbox_tlwh:
            x1,y1,x2,y2 = _tlwh_to_xyxy(box,height, width)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = extractor(im_crops)
        else:
            features = np.array([])
        return features

def _get_features_hog_paper(bbox_tlwh, ori_img):
        features = np.zeros((len(bbox_tlwh),512))    # 512 256 128
        height, width = ori_img.shape[:2]
        for i,box in enumerate(bbox_tlwh):
            x1,y1,x2,y2 = _tlwh_to_xyxy(box,height, width)
            im = ori_img[y1:y2,x1:x2]
            if im.shape[0]>0 and im.shape[1]>0:
                image=cv2.resize(im, (64,128))
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # Convert the original image to gray scale
                cell_size = (8, 16)                # 512: (8, 16)  256: (16, 16) 128: (16, 32)
                num_cells_per_block = (2, 2)
                block_size = (num_cells_per_block[0] * cell_size[0],num_cells_per_block[1] * cell_size[1])
                x_cells = gray_image.shape[1] // cell_size[0]
                y_cells = gray_image.shape[0] // cell_size[1]
                h_stride = 2
                v_stride = 2
                block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)
                num_bins = 8
                win_size = (x_cells * cell_size[0] , y_cells * cell_size[1])
                hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
                feature = hog.compute(gray_image)
                features[i]=feature.reshape(512)   # 512 256 128
            else:
                features = np.array([])

        return features

def _tlwh_to_xyxy(bbox_tlwh,height, width):
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),height-1)
        return x1,y1,x2,y2

def main():
    mot_dir="../MOT17/train/SDP"
    output_dir="../resources/detections17_SDP/64CNN"
    detection_dir=None
    generate_detections(mot_dir, output_dir,
                        detection_dir)


if __name__ == "__main__":
    main()
