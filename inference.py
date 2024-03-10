import os
import sys

import math
import cmath
import time
import glob

from argparse import ArgumentParser
from collections import OrderedDict

import cv2
import numpy as np
import torch

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow

from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result, vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo
from mmdet.apis import init_detector, inference_detector
import mmcv
import shutil

try:
    from mmtrack.apis import inference_mot
    from mmtrack.apis import inference_vid
    from mmtrack.apis import init_model as init_tracking_model

    has_mmtrack = True
except (ImportError, ModuleNotFoundError):
    has_mmtrack = False


def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame


def draw_pose_on_img(img, pose_results):
    # palette copied
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    keypoints = 17
    thickness = 2
    color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    for pose in pose_results:
        start_point = (int(pose['bbox'][0]), int(pose['bbox'][1]))
        end_point = (int(pose['bbox'][2]), int(pose['bbox'][3]))
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        img = cv2.putText(img, str(pose['track_id']), (end_point[0] - 2, end_point[1] - 2), font,
                          fontScale, color, thickness, cv2.LINE_AA)

    return img


def vizualize_record(img, flo_res, counter, pose, pose_model):
    # map flow to rgb image
    flo = visualize_flow(flo_res, None)
    img_pos = vis_pose_tracking_result(pose_model, img, pose)
    flo_pos = vis_pose_tracking_result(pose_model, flo, pose)
    img_flo = np.concatenate([img, flo], axis=0)
    img_flo_r = np.concatenate([img_pos, flo_pos], axis=0)

    # concatenate, save and show images
    img_flo = np.concatenate([img, flo], axis=0)
    img_flo = np.concatenate([img_flo, img_flo_r], axis=1)

    cv2.imshow("Tracking", img_flo / 255.0)
    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        return False, flo
    return True, flo


def scale_up(pose_results, from_size, to_size):
    pose_results_2a = []
    keypoints = 17
    for pose in pose_results:
        person_pose = {}
        person_pose['track_id'] = pose['track_id']
        person_pose['bbox'] = [to_size[0] / from_size[0] * pose['bbox'][0], to_size[1] / from_size[1] * pose['bbox'][1],
                               to_size[0] / from_size[0] * pose['bbox'][2], to_size[1] / from_size[1] * pose['bbox'][3],
                               pose['bbox'][4]]
        person_pose['keypoints'] = [[0 for x in range(3)] for y in range(17)]
        for i in range(keypoints):
            person_pose['keypoints'][i][0] = to_size[0] / from_size[0] * pose['keypoints'][i][0]
            person_pose['keypoints'][i][1] = to_size[1] / from_size[1] * pose['keypoints'][i][1]
            person_pose['keypoints'][i][2] = pose['keypoints'][i][2]
        pose_results_2a.append(person_pose)

    return pose_results_2a


def print_pose(pose_results, from_size, to_size):
    pose_results_2a = []
    keypoints = 17
    for pose in pose_results:
        person_pose = {}
        person_pose['track_id'] = pose['track_id']
        person_pose['bbox'] = [to_size[0] / from_size[0] * pose['bbox'][0], to_size[1] / from_size[1] * pose['bbox'][1],
                               to_size[0] / from_size[0] * pose['bbox'][2], to_size[1] / from_size[1] * pose['bbox'][3],
                               pose['bbox'][4]]
        person_pose['keypoints'] = [[0 for x in range(3)] for y in range(17)]
        for i in range(keypoints):
            person_pose['keypoints'][i][0] = to_size[0] / from_size[0] * pose['keypoints'][i][0]
            person_pose['keypoints'][i][1] = to_size[1] / from_size[1] * pose['keypoints'][i][1]
            person_pose['keypoints'][i][2] = pose['keypoints'][i][2]
        pose_results_2a.append(person_pose)

    return pose_results_2a


def process_mmtracking_results(mmtracking_results):
    """Process mmtracking results.

    :param mmtracking_results:
    :return: a list of tracked bounding boxes
    """
    person_results = []
    # 'track_results' is changed to 'track_bboxes'
    # in https://github.com/open-mmlab/mmtracking/pull/300
    if 'track_bboxes' in mmtracking_results:
        tracking_results = mmtracking_results['track_bboxes'][0]
    elif 'track_results' in mmtracking_results:
        tracking_results = mmtracking_results['track_results'][0]

    for track in tracking_results:
        person = {}
        person['track_id'] = int(track[0])
        person['bbox'] = track[1:]
        person['bbox'][2] = person['bbox'][2] - person['bbox'][0]
        person['bbox'][3] = person['bbox'][3] - person['bbox'][1]
        person_results.append(person)

    return person_results


def write_header(fl, vid_name):
    fl.write('video name: ')
    fl.write(vid_name)
    fl.write('\n')
    fl.write('frame_index, vid_time (s), ')
    for i in range(0, 10):
        fl.write('{per}, {per}_x1, {per}_y1, {per}_x2, {per}_y2, {per}_prob, '.format(per='p_' + str(i)))
        for j in range(0, 17):
            fl.write(
                '{kp}_x, {kp}_y, {kp}_prob, {kp}_5x5_u, {kp}_5x5_v, {kp}_7x7_u, {kp}_7x7_v, {kp}_9x9_u, {kp}_9x9_v, {kp}_11x11_u, {kp}_11x11_v, {kp}_13x13_u, {kp}_13x13_v, '.format(
                    kp='kp_' + str(j + 1)))
    fl.write('person_count')
    fl.write('\n')


def write_uv(fl, y, x, sz, flow_result):
    s = (sz - 1) // 2

    if x - s < 0 or y - s < 0 or y + s + 1 > 360 or x + s + 1 > 640:
        fl.write('-1, -1, ')
        return
    u = 0
    v = 0
    for i in range(int(x) - s, int(x) + s + 1):
        for j in range(int(y) - s, int(y) + s + 1):
            u += flow_result[j][i][0]
            v += flow_result[j][i][1]

    u = round(u / (sz * sz), 4)
    v = round(v / (sz * sz), 4)
    fl.write(str(u) + ', ' + str(v) + ', ')


def write_record(fl, frame_index, frame_time_est, pose_results, flow_result):
    fl.write(str(frame_index))
    fl.write(', ')
    fl.write(str(round(frame_time_est, 3)))
    fl.write(', ')
    count = 0
    for pose in pose_results:
        fl.write(str(pose['track_id']))
        fl.write(', ')
        for i in range(0, 5):
            fl.write(str(pose['bbox'][i]))
            fl.write(', ')
        for i in range(0, 17):
            fl.write(str(pose['keypoints'][i][0]))
            fl.write(', ')
            fl.write(str(pose['keypoints'][i][1]))
            fl.write(', ')
            fl.write(str(pose['keypoints'][i][2]))
            fl.write(', ')
            write_uv(fl, pose['keypoints'][i][0], pose['keypoints'][i][1], 5, flow_result)
            write_uv(fl, pose['keypoints'][i][0], pose['keypoints'][i][1], 7, flow_result)
            write_uv(fl, pose['keypoints'][i][0], pose['keypoints'][i][1], 9, flow_result)
            write_uv(fl, pose['keypoints'][i][0], pose['keypoints'][i][1], 11, flow_result)
            write_uv(fl, pose['keypoints'][i][0], pose['keypoints'][i][1], 13, flow_result)
        count += 1
        if count == 10:
            break
        fl.write(str(len(pose_results)))
    fl.write('\n')


def inference(args):
    # f = '{path}/include.txt'.format(path=args.path)
    # print(f)

    if torch.cuda.is_available():
        device = "cuda"
        # parallel between available GPUs
    else:
        device = "cpu"
        # change key names for CPU runtime
    allfiles = os.listdir(args.path)
    vidlist = [fname for fname in allfiles if fname.endswith('.mp4')]
    with torch.no_grad():

        config_file = 'ModelSetup/mmflow/Config/configs/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py'
        checkpoint_file = 'ModelSetup/mmflow//checkpoints/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'

        # init a model
        model = init_model(config_file, checkpoint_file, device=device)

        config_file = 'ModelSetup/mmtrack/Config/configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
        checkpoint_file = 'ModelSetup/mmtrack/checkpoints/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'
        model_track = init_tracking_model(config_file, checkpoint_file, device=device)

        config_file = 'ModelSetup/mmpose/Config/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
        checkpoint_file = 'ModelSetup/mmpose/checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        model_pose = init_pose_model(config_file, checkpoint_file, device=device)

        # build the pose model from a config file and a checkpoint file
        pose_model = init_pose_model(config_file, checkpoint_file, device=device)
        dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            dataset_info = DatasetInfo(dataset_info)
        # optional
        return_heatmap = False
        output_layer_names = None
        video_index = 0
        print('device ' + device)

        for vid_name in vidlist:
            video_path = args.path + '/' + vid_name
            # shutil.copyfile(video_path, local_path)
            print(video_path)

            data_file_name = vid_name + '.csv'
            local_data_file_name = args.path + '/' + data_file_name
            fl = open(local_data_file_name, 'w')
            write_header(fl, vid_name)
            # capture the video and get the first frame
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            start_frame = 0
            end_frame = total_frames

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_valid, frame_0 = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)

            frame_1 = cv2.resize(frame_0, (640, 360), interpolation=cv2.INTER_CUBIC)

            counter = 0
            frame_index = 1

            end = end_frame - start_frame
            while frame_valid and counter < end:
                start = time.time()
                # read the next frame
                frame_valid, frame_2 = cap.read()

                if not frame_valid:
                    break;
                frame_2a = cv2.resize(frame_2, (640, 360), interpolation=cv2.INTER_CUBIC)
                preproc_time = time.time()

                flow_result = inference_model(model, frame_1, frame_2a)
                flow_time = time.time()
                track_result = inference_mot(model_track, frame_2a, frame_id=counter)
                track_time = time.time()
                # model_track.show_result(frame_2a, track_result,
                # score_thr=0.0,
                # show=True,
                # wait_time=int(1000. / 30),
                # out_file='tmp.jpg',
                # backend='cv2')
                boxes = process_mmtracking_results(track_result)

                pose_results, returned_outputs = inference_top_down_pose_model(pose_model, frame_2a,
                                                                               boxes, bbox_thr=None, format='xywh',
                                                                               dataset=dataset,
                                                                               dataset_info=dataset_info,
                                                                               return_heatmap=return_heatmap,
                                                                               outputs=output_layer_names)
                pose_time = time.time()

                # pose_results_2a = scale_up(pose_results, (256, 192), (640,360))

                if not frame_valid:
                    break
                # preprocessing
                # predict the flow

                ret, flo = vizualize_record(frame_1, flow_result, counter, pose_results, pose_model)
                frame_index += 1
                write_record(fl, frame_index, frame_index/fps, pose_results, flow_result)
                record_time = time.time()
                # print(pose_results)
                frame_1 = frame_2a
                counter += 1
                end = time.time()
                if counter % 100 == 0:
                    print('Video {0}, frame {1}, percent complete {2}'.format(video_index, frame_index,
                                                                              round(frame_index / total_frames,
                                                                                    4) * 100))
                # print('Total Time {0}, Preproc {1}, Flow {2}, Track {3}, Pose {4}, Record {5}'.format(round(end - start, 3),
                #    round(preproc_time - start,2), round(flow_time - preproc_time, 2), round(track_time - flow_time, 2), round(pose_time - track_time, 2), round(record_time - track_time), 2))
                # print("Frame {0}".format(frame_index))
            fl.close()
            video_index += 1
            cap.release()
    cv2.destroyAllWindows()


def main():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="./ShakeFive2")
    args = parser.parse_args()

    inference(args)


if __name__ == "__main__":
    main()
