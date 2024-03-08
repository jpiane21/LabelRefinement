from os.path import exists
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from fileloader import load_people
from filedata import FileData

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result, vis_pose_tracking_result)
from mmtrack.apis import inference_mot
from mmtrack.apis import init_model as init_tracking_model

from mmpose.datasets import DatasetInfo
import enum

class Label(enum.Enum):
    STAND = "stand"
    APPROACH = "approach"
    LEAVE = "leave"
    HIGH_FIVE = "high_five"
    HAND_SHAKE = "hand_shake"
    THUMBS_UP = "thumbs_up"
    EXPLAIN_ROUTE = "explain_route"
    HUG = "hug"
    PASS_OBJECT = "pass_object"
    ROCK_PAPER_SCISSORS = "rock_paper_scissors"
    FIST_BUMP = "fist_bump"

    @staticmethod
    def from_str(label):
        if label in ('stand', 'Stand'):
            return Label.STAND
        elif label in ('leave', 'Leave', 'LEAVE'):
            return Label.LEAVE
        elif label in ('approach', 'Approach', 'APPROACH'):
            return Label.APPROACH
        elif label in ('hand_shake', 'Handshake', 'HandShake', 'Hand_Shake', 'handshake', 'HAND_SHAKE'):
            return Label.HAND_SHAKE
        elif label in ('high_five', 'High_Five', 'HIGH_FIVE'):
            return Label.HIGH_FIVE
        elif label in ('thumbs_up', 'Thumbs_Up'):
            return Label.THUMBS_UP
        elif label in ('explain_route', 'Explain_Route'):
            return Label.EXPLAIN_ROUTE
        elif label in ('hug', 'Hug', 'HUG'):
            return Label.HUG
        elif label in ('pass_object', 'Pass_Object'):
            return Label.PASS_OBJECT
        elif label in ('rock_paper_scissors', 'Rock_Paper_Scissors'):
            return Label.ROCK_PAPER_SCISSORS
        elif label in ('fist_bump', 'Fist_Bump', 'FIST_BUMP'):
            return Label.FIST_BUMP
        else:
            raise NotImplementedError

class LabelRange:
    def __init__(self, label, start, end):
        self.label = Label(label)
        self.start = start
        self.end = end

    def copy(self):
        return LabelRange(self.label, self.start, self.end)


class Video:
    def __init__(self, file, xml_file, length, action, annotation_list, label_entries):
        self.file_name = file
        self.xml_file = xml_file
        self.length = length
        self.action = action
        self.annotation_list = annotation_list
        self.label_entries = label_entries

        fd = FileData()
        fd.file_name_full = file + '.csv'
        self.tracked_persons, fd.frame_count = load_people(fd.file_name_full, fd)
        for tp in self.tracked_persons.values():
            #tp.trim_ankles()
            tp.calc_signals()

    def get_boxes_at_frame(self, frame_index):
        boxes = []
        for tp in self.tracked_persons.values():
            index = frame_index - tp.start_frame
            if index > -1:
                boxes.append(tp.bounding_boxes[frame_index])
        return boxes


class VideoLoader:
    def __init__(self, file, device):
        self.video_list = []
        self.curr_frame = 0
        self.curr_img = None
        self.prev_img = None
        self.cap = None
        self.video = None
        self.load_new_video(file)
        self.use_pose = True
        self.colors = [(255, 0, 0), (125, 0, 125), (0, 255, 0), (0, 0, 255), (0, 125, 125), (125, 125, 0)]


        if self.use_pose:
            config_file = 'ModelSetup/mmflow/Config/configs/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py'
            checkpoint_file = 'ModelSetup/mmflow//checkpoints/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'
            self.flow_model = init_model(config_file, checkpoint_file, device=device)

            config_file = 'ModelSetup/mmtrack/Config/configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
            checkpoint_file = 'ModelSetup/mmtrack/checkpoints/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'
            self.model_track = init_tracking_model(config_file, checkpoint_file, device=device)

            config_file = 'ModelSetup/mmpose/Config/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
            checkpoint_file = 'ModelSetup/mmpose/checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
            self.pose_model = init_pose_model(config_file, checkpoint_file, device=device)

            self.dataset = self.pose_model.cfg.data['test']['type']
            self.dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
            self.dataset_info = DatasetInfo(self.dataset_info)
        else:
            self.flow_model = None
            self.model_track = None
            self.pose_model = None
            self.dataset = None
            self.dataset_info = None
            self.dataset_info = None

        self.return_heatmap = False
        self.output_layer_names = None
        self.track_result = None


    def get_scaled_frame_at(self, index):
        if self.curr_frame == index :
            pass
        elif self.curr_frame < index :
            while self.curr_frame < index - 1:
                self.cap.read()
                self.curr_frame += 1
            valid, self.curr_img = self.cap.read()
        else:
            while self.curr_frame > index - 1:
                self.read_prev_image()
            valid, self.curr_img = self.cap.read()
        scaled = cv2.resize(self.curr_img, (800, 450), interpolation=cv2.INTER_CUBIC)
        self.draw_people()
        self.curr_frame = index
        return scaled

    def load_new_video(self, file):
        self.video_list = []
        self.curr_frame = 0
        self.curr_img = None
        self.prev_img = None
        file_no_ext = file.replace(".mp4", "")
        self.cap = cv2.VideoCapture(file)
        length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        action, annotation_list, label_entries = self.__load_label_from_files(file_no_ext, 0, length)
        if label_entries[2].label == Label.STAND:
            le = []
            le.append(label_entries[0])
            le.append(label_entries[1])
            le.append(label_entries[3])
            le.append(label_entries[4])
            le[1].end = label_entries[2].end
            label_entries = le
        self.video = Video(file, file_no_ext, length, action, annotation_list, label_entries)


    def get_img_arr(self, img):
        blue, green, red = cv2.split(img)
        img = cv2.merge((red, green, blue))
        return img

    def get_flow_pose(self):
        if self.use_pose:
            flow_result = inference_model(self.flow_model, self.prev_img, self.curr_img)
            flo = visualize_flow(flow_result, None)
            flo_pos = vis_pose_tracking_result(self.pose_model, flo, self.__get_pose_results())
            return flo_pos
        else:
            return self.curr_img

    def get_rgb_pose(self):
        if self.use_pose:
            img_pos = vis_pose_tracking_result(self.pose_model, self.curr_img, self.__get_pose_results())
            return img_pos
        else:
            return self.curr_img

    def __get_pose_results(self):
        to_remove = []
        if self.use_pose:
            boxes = self.__process_mmtracking_results(self.track_result)
            #boxes2 = self.video.get_boxes_at_frame(self.curr_frame)
            pose_results, returned_outputs = inference_top_down_pose_model(self.pose_model, self.curr_img,
                            boxes, bbox_thr=None, format='xywh', dataset=self.dataset, dataset_info=self.dataset_info,
                            return_heatmap=self.return_heatmap, outputs=self.output_layer_names)



            for i in range(0, len(pose_results)):
                prob = pose_results[i]['bbox'][4]
                if prob < 0.8:
                    to_remove.append(i)

            if len(to_remove) > 0:
                to_remove.sort(reverse=True)
                for ti in to_remove:
                    del pose_results[ti]
            return pose_results
        return self.curr_img

    def __process_mmtracking_results(self, mmtracking_results):
        """Process mmtracking results.

        :param mmtracking_results:
        :return: a list of tracked bounding boxes
        """
        person_results = []
        # 'track_results' is changed to 'track_bboxes'
        # in https://github.com/open-mmlab/mmtracking/pull/300
        if self.use_pose:


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


    def scale_box(self, box):
        return [ x * 2 for x in box ]

    def draw_people(self):
        for tp in self.video.tracked_persons.values():
            i = int(self.curr_frame - tp.start_frame)
            l = len(tp.bounding_boxes)
            if 0 <= i < l:
                box = tp.bounding_boxes[i]
                box = self.scale_box(box)
                start_point = (int(box[0]), int(box[1]))
                start_point_offset = (int(box[0]), int(box[1]) - 20)

                end_point = (int(box[2]), int(box[3]))
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                self.curr_img = cv2.rectangle(self.curr_img, start_point, end_point, self.colors[int(tp.id)], thickness)
                cv2.putText(self.curr_img, 'Person ' + str(int(tp.id) + 1), start_point_offset, font, 1, self.colors[int(tp.id)], 2, cv2.LINE_AA)
        return self.curr_img


    def read_next_image(self):
        self.prev_img = self.curr_img
        if self.curr_frame <= self.video.length:
            valid, self.curr_img = self.cap.read()
            self.curr_frame += 1
            self.draw_people()


            #self.curr_img = cv2.resize(self.curr_img, (640, 360), interpolation=cv2.INTER_CUBIC)

            if self.use_pose:
                self.track_result = inference_mot(self.model_track, self.curr_img, frame_id=int(self.curr_frame))

            return self.curr_img
        return None

    def read_prev_image(self):
        if self.curr_frame > 0:
            next_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.curr_frame = next_frame - 1
            previous_frame = self.curr_frame - 1

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, previous_frame)
            valid, self.curr_img = self.cap.read()
            #self.curr_img = cv2.resize(self.curr_img, (640, 360), interpolation=cv2.INTER_CUBIC)

            self.curr_frame += 1
            return self.curr_img
        return None

    def __load_label_from_files(self, file_path, start, end):
        rev_file = file_path + ".rev"
        #if exists(rev_file):
            #return self.__load_label_as_list(rev_file)
        ann_file =  file_path + ".ann"
        #if exists(ann_file):
            #return self.__load_label_as_list(ann_file)

        action, ann_list, labels_entries = self.__load_label_from_XML(file_path + ".xml", start, end)
        self.__save_ann_list(ann_file, ann_list)
        return action, ann_list, labels_entries


    def __load_label_as_list(self, file_path):
        with open(file_path, 'r') as fp:
            lines = fp.read().splitlines()

        action, labels_entry = self.__get_action_entries(lines)
        return action, lines, labels_entry

    def __get_action_entries(self, ann_list):
        labels_entry = []
        i = 1

        current_label = ''
        start = 0
        end = 0
        for ann in ann_list:
            if ann != 'leave' and ann != 'approach' and ann != 'no_people' and ann != 'stand':
                action = ann
            if current_label != ann:
                if current_label != '':
                    labels_entry.append(LabelRange(current_label, start, end))
                start = i
                current_label = ann
            end = i
            i += 1
        labels_entry.append(LabelRange(current_label, start, end))
        return action, labels_entry

    def __save_ann_list(self, file_path, ann_list):
        with open(file_path, 'w') as fp:
            fp.write('\n'.join(ann_list))


    def __load_label_from_XML(self, file_path, start, end):
        labels_overall = {}
        labels_entry = {}
        labels_end = {}

        xml_file_list = []
        ann_list = []

        mytree = ET.parse(file_path)
        myroot = mytree.getroot()
        video_node = myroot[0]
        num_frames = int(video_node.findall('frames_amount')[0].text)
        frame_list = video_node.findall('frames')[0]
        peopleFound = 0
        start = 0
        end - 0
        for i in range(start, end):
            frame = frame_list[i]
            ts = frame.findall('timestamp')[0].text
            amt = int(frame.findall('skeletons_amount')[0].text)
            if amt > 0:
                skels = frame.findall('skeletons')[0]
                action = 'no_people'
                for j in range(0, amt):
                    id = int(skels[j].findall('id')[0].text)
                    #if action != 'no_people' and action != skels[j].findall('action_name')[0].text:
                    #    print('diff')
                    action = skels[j].findall('action_name')[0].text
                    i += 1
                ann_list.append(action)
                peopleFound += 1
        action = ''
        action, labels_entry = self.__get_action_entries(ann_list)
        return action, ann_list, labels_entry


