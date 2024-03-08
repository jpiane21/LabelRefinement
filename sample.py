import cv2
import numpy as np
import pickle
import os
from os import listdir
#from mmflow.apis import inference_model, init_model
from fileloader import load_people
from filedata import FileData
import xml.etree.ElementTree as ET
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt
import time

class Sample:
    ##################################
    # Clip
    #   1, 2 , 3 ,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14.
    #   OF Frames [3, 6], [6, 9], [9, 12]
    ##################################
    def __init__(self, of_frames, signals_2D_pose, label, index, start, end, file):
        self.of_frames = []
        for i in range(0, 3):
            self.of_frames.append(of_frames[i])
            # cv2.imresize()
        img_12 = np.concatenate([self.of_frames[0], self.of_frames[1]], axis=1)
        self.concat_imgs = np.concatenate([img_12, self.of_frames[2]], axis=1)
        self.signals_2D_from_pose = signals_2D_pose
        #self.signals_2D_from_kinect = signals_2D_skel
        self.label = label
        self.index = index
        self.start_frame = start
        self.end_frame = end
        self.file_name = file

def save_sample(s):
    with open("D:\\ShakeFive2Samples\\" + s.label + "\\" + s.label + str(s.index) + ".sam", "wb") as outfile:
        pickle.dump(s, outfile)
    with open("D:\\ShakeFive2Samples\\Sample_" + str(s.index) + ".sam", "wb") as outfile:
        pickle.dump(s, outfile)

def load_samples(path_in):
    samples = []
    folders = listdir(path_in)
    for dir in folders:
        files = listdir(dir)
        for f in files:
            with open(f, "rb") as infile:
                sam = pickle.load(infile)
                samples.append(sam)
    return samples


class SkeletonInfo:
    def __init__(self, id, action, head, left_wrist, right_wrist, left_ankle, right_ankle):
        self.id = id
        self.action = action
        self.head = head
        self.left_wrist = left_wrist
        self.right_wrist = right_wrist
        self.left_ankle = left_ankle
        self.right_ankle = right_ankle

def load_signals_from_xml(file_path, start, end):
    labels_overall = {}
    xml_file_list = []

    mytree = ET.parse(file_path)
    myroot = mytree.getroot()
    video_node = myroot[0]
    num_frames = int(video_node.findall('frames_amount')[0].text)
    frame_list = video_node.findall('frames')[0]
    skels_by_frame = []

    for i in range(start, end+1):
        frame = frame_list[i]
        skels_by_frame.append([])
        ts = frame.findall('timestamp')[0].text
        amt = int(frame.findall('skeletons_amount')[0].text)
        if amt > 0:
            skels = frame.findall('skeletons')[0]
            for j in range(0, amt):
                id = int(skels[j].findall('id')[0].text)
                action = skels[j].findall('action_name')[0].text
                j_amt = int(skels[j].findall('joints_amount')[0].text)
                joints = skels.findall('joints')
                head = make_tuple('(' + joints.findall('joint_3')[0].findall('point3Dd')[0].text + ')')
                left_wrist = make_tuple('(' + joints.findall('joint_6')[0].findall('point3Dd')[0].text + ')')
                right_wrist = make_tuple('(' + joints.findall('joint_10')[0].findall('point3Dd')[0].text + ')')
                left_ankle = make_tuple('(' + joints.findall('joint_14')[0].findall('point3Dd')[0].text + ')')
                right_ankle = make_tuple('(' + joints.findall('joint_18')[0].findall('point3Dd')[0].text + ')')
                skels_by_frame[i].append(SkeletonInfo(id, action, head, left_wrist, right_wrist, left_ankle, right_ankle))
    return skels_by_frame


def load_label_from_XML(file_path, start, end):
    labels_overall = {}
    labels_start = {}
    labels_end = {}

    xml_file_list = []

    mytree = ET.parse(file_path)
    myroot = mytree.getroot()
    video_node = myroot[0]
    num_frames = int(video_node.findall('frames_amount')[0].text)
    frame_list = video_node.findall('frames')[0]
    peopleFound = 0
    for i in range(start, end + 1):
        frame = frame_list[i]
        ts = frame.findall('timestamp')[0].text
        amt = int(frame.findall('skeletons_amount')[0].text)
        if amt > 0:
            skels = frame.findall('skeletons')[0]
            for j in range(0, amt):
                id = int(skels[j].findall('id')[0].text)
                action = skels[j].findall('action_name')[0].text
                if action in labels_overall.keys():
                    labels_overall[action] += 1
                    labels_end[action] = i
                else:
                    labels_overall[action] = 1
                    labels_start[action] = i
                    labels_end[action] = i
            peopleFound += 1

    label = ""
    label_count = 0
    for lab in labels_overall.keys():
        if labels_overall[lab] > label_count:
            label = lab
    if len(label) == 0 and peopleFound == 0:
        label = 'no_people'
    return label, labels_overall, labels_start, labels_end

def handle_gap(tp):
    done = False
    remove = []

    for g in tp.gaps:
        tp.fill_gap(g[0], g[1])
    tp.calc_signals()
    tp.gaps = []
    return True

def plot_signals(people, length, action_start, action_end):
    max = 400
    for tp in people.values():
        low = 0
        while low <2:
            legend_list = []
            legend_string = []
            ls = 'Clip'
            plt.clf()
            for i in range(1, int(length / 14)):
                l1 = plt.vlines(x=i * 14, ymin=0, ymax=max,
                                colors='gray',
                                ls='--',
                                label=ls)
                #legend_list.append(l1)
                #legend_string.append(ls)

            l1 = plt.vlines(x=length - 14, ymin=0, ymax=max,
                            colors='gray',
                            ls='--',
                            label=ls)
            legend_list.append(l1)
            legend_string.append(ls)

            ls = 'Start/End Action'
            l1 = plt.vlines(x=action_start, ymin=0, ymax=max,
                            colors='blue',
                            ls='--',
                            label=ls)
            legend_list.append(l1)
            legend_string.append(ls)
            l1 = plt.vlines(x=action_end, ymin=0, ymax=max,
                            colors='blue',
                            ls='--',
                            label=ls)
            #legend_list.append(l1)
            #legend_string.append(ls)

            X = range(tp.start_frame, tp.end_frame + 1)
            ls = 'Right Wrist'
            l3, = plt.plot(X, tp.right_wrist_speed_smooth, label=ls)
            #l3, = plt.plot(X, tp.right_wrist_accel_smooth, label=ls)
            legend_list.append(l3)
            legend_string.append(ls)
            ls = 'Left Wrist'
            l4, = plt.plot(X, tp.left_wrist_speed_smooth, label=ls)
            #l4, = plt.plot(X, tp.left_wrist_accel_smooth, label=ls)
            legend_list.append(l4)
            legend_string.append(ls)
            ls = 'Right Ankle'
            l5, = plt.plot(X, tp.right_ankle_speed_smooth, label=ls)
            #l5, = plt.plot(X, tp.right_ankle_accel_smooth, label=ls)
            legend_list.append(l5)
            legend_string.append(ls)
            ls = 'Left Ankle'
            l6, = plt.plot(X, tp.left_ankle_speed_smooth, label=ls)
            #l6, = plt.plot(X, tp.left_ankle_accel_smooth, label=ls)
            legend_list.append(l6)
            legend_string.append(ls)
            plt.xlabel('Frame number')
            #plt.ylabel('Acceleration')
            #plt.title('Player acceleration signals, wrists and ankles')
            plt.ylabel('Speed px/33.33ms')

            plt.title('Player speed signals, wrists and ankles')
            plt.legend(legend_list, legend_string)
            if low == 1:
                plt.ylim(0, 75)
            plt.show()
            low += 1

def create_samples_from_video(vid_path, start_index, device):
    done = False
    samples = []
    current_index = 1
    cap = cv2.VideoCapture(vid_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    rgb_frames = []
    fd = FileData()
    fd.file_name_full = vid_path + '.csv'
    fd.tracked_persons, fd.frame_count = load_people(fd.file_name_full, fd)

    waitonce = True
    for tp in fd.tracked_persons.values():
        handle_gap(tp)
        tp.calc_signals()

    xml_file = vid_path.replace(".mp4", ".xml")
    label, labels_overall, labels_start, labels_end = load_label_from_XML(xml_file, 1, length-1)
    s = 'hand_shake'
    if s in labels_overall:
        #plot_signals(fd.tracked_persons, length, labels_start[s], labels_end[s])
        valid, img = cap.read()
        row, col = img.shape[:2]
        bottom = img[row - 2:row, 0:col]
        mean = cv2.mean(bottom)[0]
        bordersize = 30
        boundary = [255, 0, 0]
        activity = [0, 255, 0]
        noactivity = [128, 128, 128]
        i = 1
        while valid:
            border = cv2.copyMakeBorder(img,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value= activity if i in range(labels_start[s], labels_end[s]) else boundary if i in range(labels_start[s] -10, labels_end[s] + 10) else noactivity
            )



            cv2.imshow(s, border / 255.0)
            k = cv2.waitKey(25) & 0xFF
            if k == 27:
                return False
            valid, img = cap.read()
            i+=1
            if waitonce:
                time.sleep(3)
                waitonce = False
        return
    else:
        return

    config_file = 'ModelSetup/mmflow/Config/configs/flownet2/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.py'
    checkpoint_file = 'ModelSetup/mmflow//checkpoints/flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'
    #model = init_model(config_file, checkpoint_file, device=device)

    while not done:
        if length - current_index < 15:
            current_index = length - 15
            done = True
        start = current_index
        end = current_index + 14
        for i in range(0,4):
            cap.read()
            current_index += 1
            cap.read()
            current_index += 1
            valid, frame = cap.read()
            current_index += 1
            rgb_frames.append(frame)

        flow_frames = []

        for i in range(0, 3):
            frame_1 = cv2.resize(rgb_frames[i], (640, 360), interpolation=cv2.INTER_CUBIC)
            frame_2 = cv2.resize(rgb_frames[i+1], (640, 360), interpolation=cv2.INTER_CUBIC)
            #flow_frames.append(inference_model(model, frame_1, frame_2))

        pose_signals = []
        i = 0
        for tp in fd.tracked_persons.values():
            pose_signals.append([])
            pose_signals.append([])
            pose_signals.append([])
            pose_signals.append([])

            for j in range(start, end):
                if tp.is_person_here(j):
                    k = j - tp.start_frame
                    pose_signals[i * 4].append(tp.right_wrist_speed_smooth[k])
                    pose_signals[i * 4 + 1].append(tp.left_wrist_speed_smooth[k])
                    pose_signals[i * 4 + 2].append(tp.left_ankle_speed_smooth[k])
                    pose_signals[i * 4 + 3].append(tp.left_ankle_speed_smooth[k])
                else:
                    pose_signals[i * 4].append(-1)
                    pose_signals[i * 4 + 1].append(-1)
                    pose_signals[i * 4 + 2].append(-1)
                    pose_signals[i * 4 + 3].append(-1)
            i += 1

        while i < 15:
            pose_signals.append([])
            pose_signals.append([])
            pose_signals.append([])
            pose_signals.append([])
            pose_signals[i * 4].append(-1)
            pose_signals[i * 4 + 1].append(-1)
            pose_signals[i * 4 + 2].append(-1)
            pose_signals[i * 4 + 3].append(-1)
            i += 1

        s = Sample(flow_frames, pose_signals, load_label_from_XML(vid_path.replace(".mp4", ".xml"), start, end), start_index, start, end,vid_path)
        samples.append(s)
        start_index += 1
        save_sample(s)
    return samples, start_index


def create_samples(videos_dir, sample_path, dev):
    start_index = 0
    video_count = 1
    sample_list = []
    for file in os.listdir(videos_dir):
        if file.endswith(".mp4"):
            print('video ' + str(video_count) + ': ' + file)
            #samples, start_index = \
            create_samples_from_video(videos_dir + file, start_index, dev)
            #sample_list.extend(samples)
            video_count += 1
