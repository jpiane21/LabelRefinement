import math

import numpy
import torch
import torchvision
import cv2
from filedata import FileData
from fileloader import load_people
from torch.utils.data import Dataset, DataLoader
from mmflow.apis import inference_model, init_model
import numpy as np
from random import randint


IMAGE_WIDTH = 640

STAND = 0
APPROACH = 1
HAND_SHAKE = 2
HUG = 3
HIGH_FIVE = 4
FIST_BUMP = 5
LEAVE = 6
UNKNOWN = 7

class SampleVid:
    def __init__(self, file_name, ann, ann_ranges):
        self.file_name = file_name
        self.ann = ann
        self.ann_ranges = ann_ranges
    @staticmethod

    def get_label_at(center_frame, label_str, label_ranges):
        if 0 <= center_frame <= label_ranges[0]:
            return STAND

        for i in range(0, 4):
            if label_ranges[i * 2] <= center_frame <= label_ranges[i * 2 + 1]:
                if i == 0:
                    return APPROACH
                elif i == 1:
                    return SampleVid.getActionOfInterest(label_str)
                elif i == 2:
                    return LEAVE
                elif i == 3:
                    return STAND

        if label_ranges[7] < center_frame and label_ranges[7] != -1:
            return STAND
        if label_ranges[5] < center_frame:
            return LEAVE
        return UNKNOWN

    @staticmethod
    def getActionOfInterest(label_str):
        if label_str == 'HUG':
            return HUG
        elif label_str == 'HIGH_FIVE':
            return HIGH_FIVE
        elif label_str == 'FIST_BUMP':
            return FIST_BUMP
        elif label_str == 'HAND_SHAKE':
            return HAND_SHAKE
        return UNKNOWN


class ShakeFive2Dataset(Dataset):
    def __init__(self, video_list, device, window_size, under_sample):
        self.sampleVids = video_list
        self.labels = []
        self.samples = []
        self.fileData = {}
        self.window_size = window_size
        for v in video_list:
            self.load_samples_from_video(v.file_name, v.ann, v.ann_ranges)

        counts = self.getLabelCounts()
        if under_sample:
            self.undersample(counts)
        self.n_samples = np.shape(self.samples)[0]

        self.y_data = torch.from_numpy(np.asarray(self.labels, dtype=numpy.integer)) # size [n_samples, 1]

    def load_samples_from_video(self, file_name, label_str, label_ranges):

        fd = FileData()
        tracked_persons, fd.frame_count = load_people(file_name + ".csv", fd)
        self.fileData[file_name] = [fd, tracked_persons]
        hw = int((self.window_size - 1)/2)
        hw_e = hw
        if hw == 0:
            hw = 1
            hw_e = 2

        for i in range(hw, fd.frame_count - hw_e):
            self.samples.append([file_name, i])
            self.labels.append(SampleVid.get_label_at(i, label_str, label_ranges))

    def undersample(self, counts):
        #to_value = counts[min(counts, key=counts.get)]
        to_value = max(counts[HAND_SHAKE], max(counts[HUG], max(counts[HIGH_FIVE], counts[FIST_BUMP])))

        reduce = []
        reduce.append([STAND, counts[STAND] - to_value])
        reduce.append([APPROACH, counts[APPROACH] - to_value])
        reduce.append([LEAVE, counts[LEAVE] - to_value])

        while (reduce[0][1] > 0 or reduce[1][1] > 0 or reduce[2][1] > 0):
            i = randint(0, len(self.samples)-1)
            #print("List size is: " + str(len(self.labels)) + " i is: " + str(i))
            for j in range(0, 3):
                if self.labels[i] == reduce[j][0] and reduce[j][1] > 0:
                    del self.labels[i]
                    del self.samples[i]
                    reduce[j][1] -= 1
                    break

    def getLabelCounts(self):
        counts = {STAND:0, APPROACH:0, HAND_SHAKE:0, HUG:0, HIGH_FIVE:0, FIST_BUMP:0, LEAVE:0}
        for l in self.labels:
            counts[l] += 1

        print("Stand: " + str(counts[STAND]))
        print("Approach: " + str(counts[APPROACH]))
        print("Hand Shake: " + str(counts[HAND_SHAKE]))
        print("Hug: " + str(counts[HUG]))
        print("High Five: " + str(counts[HIGH_FIVE]))
        print("Fist Bump: " + str(counts[FIST_BUMP]))
        print("Leave: " + str(counts[LEAVE]))
        return counts


    def __len__(self):
        return self.n_samples


class ShakeFive2DatasetSignals(ShakeFive2Dataset):
    def __init__(self, video_list, device, window_size, under_sample):
        super(ShakeFive2DatasetSignals, self).__init__(video_list, device, window_size, under_sample)

        self.y_min = 80.74
        self.y_max = 429.00
        self.y_range = self.y_max - self.y_min
        self.x_min = 0.00
        self.x_max = 647.86
        self.x_range = self.x_max - self.x_min
        self.s_min = 0.00
        self.s_max = 221.21
        self.s_range = self.s_max - self.s_min
        #self.max_people = int(math.floor(IMAGE_WIDTH / self.window_size))
        self.max_people = 2 #hardcoded limit --

    def getJointsDefaultOrder(self, tp):
        tp.calc_signals()
        joints = []
        joints.append(tp.left_wrist)
        joints.append(tp.left_elbow)
        joints.append(tp.right_wrist)
        joints.append(tp.right_elbow)
        joints.append(tp.left_ankle)
        joints.append(tp.left_knee)
        joints.append(tp.right_ankle)
        joints.append(tp.right_knee)
        joints.append(tp.head)
        return joints

    def getAlternateOrder(self, tp):
        tp.calc_signals()
        joints = []
        joints.append(tp.head)
        joints.append(tp.left_wrist)
        joints.append(tp.right_wrist)
        joints.append(tp.right_ankle)
        joints.append(tp.left_ankle)
        joints.append(tp.left_knee)
        joints.append(tp.right_knee)
        joints.append(tp.left_elbow)
        joints.append(tp.right_elbow)
        return joints

    def getSignals(self, file_name, i):
        fd, tracked_persons = self.fileData[file_name]
        max_people_per_panel = self.max_people
        panel = 0
        tp_dict = {}
        for tp in tracked_persons.values():
            tp_dict[tp] = self.getAlternateOrder(tp)

        hw = int((self.window_size - 1)/2)
        signals = np.zeros((1, 9, self.max_people * (self.window_size + 1)), dtype=float)
        person_count = 0
        offset = 0
        for tp in tracked_persons.values():
            joints = tp_dict[tp]

            for j in range(0, len(joints)):
                offset = 0
                for l in range(i - hw + 2, i + hw + 3):  # signal data in this set is shifted by 2

                    for ss in joints[j].subsigs:
                        if ss.end_frame > l >= ss.start_frame:
                            signals[0][j][offset + (self.window_size * person_count)] = ss.speed_smooth[l - ss.start_frame] #float((ss.speed_smooth[l - ss.start_frame] - self.s_min)/self.s_range)
                           # signals[0][j*3+1][offset + (self.window_size * person_count)] = float((ss.x[l - ss.start_frame] - self.x_min)/self.x_range)
                           # signals[0][j*3+2][offset + (self.window_size * person_count)] = float((ss.y[l - ss.start_frame] - self.y_min)/self.y_range)
                            break
                    offset += 1
            for j in range(0, len(joints)):
                signals[0][j][offset + (self.window_size * person_count)] = 0
            person_count += 1

            if person_count >= max_people_per_panel:
                break
        return signals

    def __getitem__(self, index):

        file_name, i = self.samples[index]
        fd, tracked_persons = self.fileData[file_name]

        signals = self.getSignals(file_name, i)

        return torch.from_numpy(signals).type(torch.FloatTensor), self.y_data[index], [file_name, i, fd.frame_count]

class ShakeFive2DatasetFrames(ShakeFive2Dataset):
    def __init__(self, video_list, device, window_size, under_sample):
        super(ShakeFive2DatasetFrames, self).__init__(video_list, device, window_size, under_sample)

        config_file = 'ModelSetup/mmflow/Config/configs/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py'
        checkpoint_file = 'ModelSetup/mmflow//checkpoints/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'
        self.flow_model = init_model(config_file, checkpoint_file, device=device)

    def getFrames(self, file_name, index):

        cap = cv2.VideoCapture(file_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        valid, prevRGB = cap.read()
        valid, currRGB = cap.read()
        valid, nextRGB = cap.read()
        prevRGB = cv2.resize(prevRGB, (640, 360), interpolation=cv2.INTER_CUBIC)
        currRGB = cv2.resize(currRGB, (640, 360), interpolation=cv2.INTER_CUBIC)
        nextRGB = cv2.resize(nextRGB, (640, 360), interpolation=cv2.INTER_CUBIC)


        prev_flow_result = inference_model(self.flow_model, prevRGB, currRGB)
        next_flow_result = inference_model(self.flow_model, currRGB, nextRGB) #[720, 1280, 2]

        return prev_flow_result, next_flow_result


    def getNextFrame(self, file_name, index):

        cap = cv2.VideoCapture(file_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        valid, currRGB = cap.read()
        valid, nextRGB = cap.read()
        currRGB = cv2.resize(currRGB, (320, 180), interpolation=cv2.INTER_CUBIC)
        nextRGB = cv2.resize(nextRGB, (320, 180), interpolation=cv2.INTER_CUBIC)

        next_flow_result = inference_model(self.flow_model, currRGB, nextRGB) #[720, 1280, 2]
        #next_flow_result_flat = self.flatten(next_flow_result)
        return next_flow_result


    def flatten(self, next_flow_result):
        [rows, cols, channels] = next_flow_result.shape
        next_flow_result_flat = np.zeros((rows, cols, 1), dtype=float)
        for i in range(0, rows):
            for j in range(0, cols):
                next_flow_result_flat[i][j][0] = np.sqrt(next_flow_result[i][j][0]*next_flow_result[i][j][0] + next_flow_result[i][j][1]*next_flow_result[i][j][1])
        return next_flow_result_flat

    def __getitem__(self, index):

        file_name, i = self.samples[index]
        fd, tracked_persons = self.fileData[file_name]

        #prev_flow_result, next_flow_result = self.getFrames(file_name, i)
        next_flow_result = self.getNextFrame(file_name, i)
        #concat_frame = np.concatenate([prev_flow_result, next_flow_result], axis=0)
        x = numpy.transpose(next_flow_result, (2, 0, 1))

        return torch.from_numpy(x).type(torch.FloatTensor), self.y_data[index], [file_name, i, fd.frame_count]

