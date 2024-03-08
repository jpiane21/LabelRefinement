from autorevisedlabel import *
import tkinter as tk
from tkinter import font as tkFont
import cv2
import torch
from playbackinfo import *
from videoloader import VideoLoader
from PIL import Image, ImageTk
from autorevise import autorevise
import ctypes
MessageBox = ctypes.windll.user32.MessageBoxW
gui = None

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(dev)

root = tk.Tk()
root.geometry('1700x900')
root.title("Annotation Boundary Assessment")

y_loc = 20
fontsize = 20
tk.Label(root, font=('Arial', fontsize), text="Choose the clip where the green border is displayed closest to the start of activity").place(x=25, y=y_loc)
y_loc += 45

fontsize = 16
annotationstr = tk.Label(root, font=("Arial", fontsize), text="Annotation: ")
annotationstr.place(x=30, y=y_loc)
annotation = tk.Label(root, font=("Arial", fontsize), )
annotation.place(x=160, y=y_loc)
clipstr = tk.Label(root, font=("Arial", fontsize), text = "Annotated Segment: ")
clipstr.place(x=1400, y=y_loc)
cliplb = tk.Label(root, font=("Arial", fontsize))
cliplb.place(x=1610, y=y_loc)
y_loc += 40

image_disp1 = tk.Label(root)
image_disp1.place(x=30, y=y_loc)

image_disp2 = tk.Label(root)
image_disp2.place(x=860, y=y_loc)
y_loc += 480
fontsize = 16
framestr = tk.Label(root,font=('Arial', fontsize),  text = "Frame #: ")
framestr.place(x=35, y=y_loc)
framestr = tk.Label(root,font=('Arial', fontsize), text = "Frame #: ")
framestr.place(x=865, y=y_loc)

frame_1 = tk.Label(root,font=('Arial', fontsize))
frame_1.place(x=125, y=y_loc)
frame_2 = tk.Label(root,font=('Arial', fontsize))
frame_2.place(x=955, y=y_loc)

buttonFont = tkFont.Font(family='Arial', size=fontsize)

tk.Button(root, text='Play', command=lambda: play(0), font=buttonFont).place(x=760, y=y_loc)
tk.Button(root, text='Play', command=lambda: play(1), font=buttonFont).place(x=1560, y=y_loc)
y_loc += 40


selectstr = tk.Label(root,font=('Arial', fontsize), text = "Please Select One: ")
selectstr.place(x=30, y=y_loc)
y_loc += 40


ns_text = "Not Sure"
selection = tk.IntVar(root, 1)

tk.Radiobutton(root, text="Left clip is closer to Temporal Boundary", variable=selection, value=0, command=lambda: toggle_closer(0), font=buttonFont).place(x=30, y=y_loc)
tk.Radiobutton(root, text=ns_text, variable=selection, value=-1, command=lambda: toggle_closer(-1), font=buttonFont).place(x=800, y=y_loc)
tk.Radiobutton(root, text="Right clip is closer to Temporal Boundary", variable=selection, value=1, command=lambda: toggle_closer(1), font=buttonFont).place(x=1280, y=y_loc)

y_loc += 60

definitionstr = tk.Label(root,font=('Arial', fontsize),  text = "Definition:")
definitionstr.place(x=50, y=y_loc)
y_loc += 30


fontsize = 12
definition = tk.Text(root, height=5, width=60, font=fontsize)
definition.place(x=60, y=y_loc)
definition.insert(tk.END, "This is very long text.  It's meant to explain the annotation definition.  For example when looking that transistion between")
y_loc = 805

tk.Button(root, text='Next Annotated Segment', command=lambda: next_clip(), font=buttonFont).place(x=1420, y=800)

pbi = PlayBackInfo()


def toggle_closer(val):
    global pbi
    global selection
    ar = pbi.get_curr()[1]

    if val == -1:
        ar.better = "Not Sure"
    elif val == 0: #choose left
        if pbi.orig_on_left:
            ar.better = "Original"
        else:
            ar.better = "Revised"
    else: # choose right
        if pbi.orig_on_left:
            ar.better = "Revised"
        else:
            ar.better = "Original"


def load_next_video():
    global pbi
    global selection
    ar = pbi.get_next()

    get_scaled_clip(ar[1].get_start(False, ar[0]), None, None, 0 if pbi.orig_on_left else 1)
    get_scaled_clip(ar[1].get_start(True, ar[0]), None, None, 1 if pbi.orig_on_left else 0)
    load_image(0)
    load_image(1)
    update_video_info(ar)
    selection.set(2)


def update_video_info(ar):
    global toggle_1
    cliplb['text'] = str(pbi.get_clip_number())
    annotation['text'] = pbi.get_ann_text(ar[0])
    definition.delete(1.0, tk.END)
    definition.insert(tk.END, pbi.get_desc_text(ar[0]))


def get_next(vid):
    global pbi
    pbi.update_clip(vid)
    load_image(vid)

def play(vid):
    global pbi
    if pbi.playing:
        pbi.playing = False
    else:
        pbi.playing = True
    do_play(vid)

def do_play(vid):
    global pbi
    pbi.fps
    if pbi.playing:

        get_next(vid)
        delay = int(1000/pbi.fps)
        if vid == 0:
            image_disp1.after(delay, do_play, vid)
        else:
            image_disp2.after(delay, do_play, vid)


def next_clip():
    global pbi
    if not pbi.record_better():
        selectstr['fg'] = '#f00'
        return
    if pbi.vid_list_index == 59:
        MessageBox(None, 'Done', 'All Finished. Thanks!', 0)
        return

    load_next_video()
    selectstr['fg'] = '#000'


def load_image(vid):
    global image_disp1
    global image_disp2

    global imgtk1
    global imgtk2
    global pbi

    if len(pbi.clip[vid]) > 0 and pbi.curr_disp_indexes[vid] < 25:
        orig_im = pbi.clip[vid][pbi.curr_disp_indexes[vid]]
        orig_im = add_border(orig_im, pbi.curr_disp_indexes[vid])
        img = pbi.get_vl().get_img_arr(orig_im)
        im = Image.fromarray(img)
        if vid == 0:
            imgtk1 = ImageTk.PhotoImage(image=im)
            image_disp1.config(image=imgtk1)
            index = pbi.curr_disp_indexes[vid]
            frame_1['text'] = str(index + pbi.get_start_frame(True))

        else:
            imgtk2 = ImageTk.PhotoImage(image=im)
            image_disp2.config(image=imgtk2)
            index = pbi.curr_disp_indexes[vid]
            frame_2['text'] = str(index+ pbi.get_start_frame(False))

def get_scaled_clip(start, orig, transition, vid):

    global pbi
    if start == 1:
        for i in range(0, 10):
            frame = pbi.clip[vid].append(pbi.get_vl().get_scaled_frame_at(1))
        for i in range(start, start + 15):
            frame = pbi.clip[vid].append(pbi.get_vl().get_scaled_frame_at(i))
        return
    if start - 10 >= 1:
        for i in range(start - 10, start + 15):
            frame = pbi.clip[vid].append(pbi.get_vl().get_scaled_frame_at(i))

def add_border(img, index):

    # making border around image using copyMakeBorder
    border_color = [128, 128, 128]
    if index <= 10:
        border_color = [255, 0, 0]
    elif 10 < index < 15:
        border_color = [0,255, 0]

    borderoutput = cv2.copyMakeBorder(
        img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)

    # showing the image with border
    return borderoutput

def mainGUI():

    global pbi
    pbi.vid_list = RandomizeAndFilter(LoadDatasetAndAutorevise())
    load_next_video()

    root.mainloop()
    print('bye')


def LoadDatasetAndAutorevise():
    dir = "d:\\"
    lines = []
    list = []
    with open(dir + "autorevise_results.csv", 'r') as fp:
        lines = fp.read().splitlines()
    for l in lines:
        values = [x.strip() for x in l.split(',')]
        list.append(AutoRevisedLabel(values))
    return list

def RandomizeAndFilter(diff_list):
    thres = 5
    dict = {Label.APPROACH: [], Label.HUG: [], Label.HIGH_FIVE: [], Label.HAND_SHAKE : [], Label.FIST_BUMP : [], Label.LEAVE : []}

    ar_vid_list = []

    for ar in diff_list:
        if abs(ar.original.Approach.start - ar.revised.Approach.start) >= thres:
            dict[Label.APPROACH].append([abs(ar.original.Approach.start - ar.revised.Approach.start), ar])
        if abs(ar.original.Action.start - ar.revised.Action.start) >= thres:
            dict[ar.revised.Action.label].append([abs(ar.original.Action.start - ar.revised.Action.start), ar])
        if abs(ar.original.Leave.start - ar.revised.Leave.start) >= thres:
            dict[Label.LEAVE].append([abs(ar.original.Leave.start - ar.revised.Leave.start), ar])

    hug = 0
    hs = 0
    hf = 0
    fb = 0
    for lab in dict:
        l = sorted(dict[lab], key=lambda x: x[0], reverse=True)
        if lab == Label.LEAVE:
            for i in range(0, len(l)):
                if l[i][1].action_lab == Label.HUG and hug < 2:
                    ar_vid_list.append([lab, l[i][1]])
                    hug += 1
                elif l[i][1].action_lab == Label.FIST_BUMP and fb < 2:
                    ar_vid_list.append([lab, l[i][1]])
                    fb += 1
                elif l[i][1].action_lab == Label.HAND_SHAKE and hs < 3:
                    ar_vid_list.append([lab, l[i][1]])
                    hs += 1
                elif l[i][1].action_lab == Label.HIGH_FIVE and hf < 3:
                    ar_vid_list.append([lab, l[i][1]])
                    hf += 1
        else:
            for i in range(0, 10):
                ar_vid_list.append([lab, l[i][1]])

    for ar in ar_vid_list:
        ar[1].LoadVideo()
    return ar_vid_list



def main():
    #global gui
    #gui = UI()
    #gui.run()
    mainGUI()
    return


if __name__ == '__main__':
    main()
