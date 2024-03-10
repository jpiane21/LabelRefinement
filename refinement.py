import torch
import os
from videoloader import VideoLoader
from filedata import FileData
from videoloader import load_people, Label as vlLabel
from autorevise import autorevise
from argparse import ArgumentParser
import cv2

def load_files(dir):
    for file in os.listdir(dir):
        # check only text files
        if file.endswith('.csv'):
            fd = FileData()
            fd.file_name_full = dir + file
            tracked_persons, frame_count = load_people(fd.file_name_full, fd)
            print(fd.file_name_full)



def copy_list(orig):
    list = []
    for lr in orig:
        if lr is None:
            list.append(None)
        else:
            list.append(lr.copy())
    return list

def run_list(action_type, video_list, fl5):

    if torch.cuda.is_available():
        device = "cuda"
        # parallel between available GPUs
    else:
        device = "cpu"
        # change key names for CPU runtime

    newDict = {}
    with open('./videoaction.csv', 'r') as f:
        for line in f:
            splitLine = line.split(',')
            newDict[splitLine[0]] = splitLine[1:]

    vl = VideoLoader(video_list[0], device)
    AR_list = []
    label_list = []
    for file in video_list:
        print(file)
        vl.load_new_video(file)
        label = copy_list(vl.video.label_entries)
        label_list.append(label)
        AR = copy_list(autorevise(file))
        AR_list.append(AR)
        if label[0].label == vlLabel.APPROACH:
            approach = 0
            action = 1
            leave = 2
            stand2 = 3
        elif label[1].label == vlLabel.APPROACH:
            approach = 1
            action = 2
            leave = 3
            stand2 = 4
        else:
            print("ERROR!\n\n\n\\n")
        fl5.write(file + ", " + str(label[action].label).replace("Label.", "") + ", " + str(label[approach].start) + ", " + str(label[approach].end) + ", ")
        fl5.write(str(label[action].start) + ", " + str(label[action].end) + ", " + str(label[leave].start) + ", " + str(label[leave].end) + ", ")
        if len(label) == stand2:
            fl5.write("-1, -1, ")
        else:
            fl5.write(str(label[stand2].start) + ", " + str(label[stand2].end) + ", ")
        fl5.write(str(AR[1].start) + ", " + str(AR[1].end) + ", " + str(AR[2].start) + ", " + str(AR[2].end) + ", " + str(AR[3].start) + ", " + str(AR[3].end) + ", ")
        if AR[stand2] is None:
            fl5.write("-1, -1\n")
        else:
            fl5.write(str(AR[stand2].start) + ", " + str(AR[stand2].end) + "\n")
    fl5.flush()

def load_vids(path):
    video_list = []
    with open("./run_vids.txt", 'r') as fl:
        for line in fl:
            video_list.append(path + line.strip())
    return video_list

def run_all_lists(dir):

    fl5 = open("./autorevise_results.csv", 'w')

    run_list("unknown", load_vids(dir), fl5)
    fl5.close()




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="./ShakeFive2")
    args = parser.parse_args()

    load_files(args.path + '/')

    run_all_lists(args.path + '/')

