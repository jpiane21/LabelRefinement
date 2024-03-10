import torch
import os
from sample import create_samples
from tkinter import *
from tkinter.ttk import *
from videoloader import VideoLoader
from videoloader import LabelRange
from videoloader import Label as vlLabel
from PIL import Image, ImageTk
from numpy import array
import matplotlib
from autorevise import autorevise
import cv2

from generatedata import inference
from fileloader import load_people
from filedata import FileData

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
vl = None
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(dev)
playing = False

root = Tk()
root.geometry('1350x1350')
root.title("Annotation Viewer")

values = {"Image": 1,
          "Pose": 2,
          "Flow/Pose": 3}

video_str = StringVar(root, "./ShakeFive2/Video-24112014-023714.mp4")
video_path_input = Entry(root, textvariable=video_str, width=80)
video_path_input.place(x=110, y=20)

imgtk = None
image_disp = Label(root)
image_disp.place(x=40, y=60)
fps_str = StringVar(root, "5")
fps_input = Entry(root, text=fps_str, width=20)
fps_input.place(x=440, y=800)
image_type = IntVar(root, 1)
signal_type = IntVar(root, 1)
person_type = IntVar(root, 1)
annotation = Label(root)
annotation.place(x=600, y=800)
frame_num = Label(root)
frame_num.place(x=700, y=800)


fig = plt.figure(figsize=(12.5, 4), dpi=100)
plt.ion()
canvas = FigureCanvasTkAgg(fig, master=root)
plot_widget = canvas.get_tk_widget()
plot_widget.place(x=40, y=880)

Label(root, text="y limit").place(x=40, y=850)
y_lim_str = StringVar(root, "40")
y_lim_input = Entry(root, text=y_lim_str, width=20)
y_lim_input.place(x=140, y=850)

rw_bool = BooleanVar(root, True)
lw_bool = BooleanVar(root, True)
ra_bool = BooleanVar(root, True)
la_bool = BooleanVar(root, True)
re_bool = BooleanVar(root, True)
le_bool = BooleanVar(root, True)
rk_bool = BooleanVar(root, True)
lk_bool = BooleanVar(root, True)
graph_values = {"Speed": 1,
          "Accel": 2}

graph_person_values = {"Person 1": 1,
                        "Person 2": 2,
                        "Both": 3}

stand = None
approach = None
action = None
leave = None
stand2 = None
def get_loader():
    global video_str
    file = video_str.get()
    global vl
    vl.load_new_video(file)
    vl.read_next_image()

    global stand
    global approach
    global action
    global leave
    global stand2
    stand, approach, action, leave, stand2, intersections, intersections2, intersections3, enpoints = autorevise(file)
    load_image()


def get_next():
    global vl
    vl.read_next_image()
    load_image()

def get_prev():
    global vl
    vl.read_prev_image()
    load_image()

def play():
    global playing
    if playing:
        playing = False
    else:
        playing = True
    do_play()

def do_play():
    global playing
    global fps_str
    if playing:
        fps = int(fps_str.get())
        get_next()
        delay =  int(1000/fps)
        image_disp.after(delay, do_play)

def load_image():
    global image_disp
    global imgtk
    global annotation
    global image_type

    orig_im = None
    if image_type.get() == 1:
        orig_im = vl.curr_img
    elif image_type.get() == 2:
        orig_im = vl.get_rgb_pose()
    elif image_type.get() == 3:
        orig_im = vl.get_flow_pose()
    orig_im = vl.get_rgb_pose()

    img = vl.get_img_arr(orig_im)
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    image_disp.config(image=imgtk)
    index = int(vl.curr_frame-1)
    if index < len(vl.video.annotation_list):
        ann = vl.video.annotation_list[index]
    else:
        ann = ''
    annotation.config(text=ann)
    frame_num.config(text=str(vl.curr_frame))
    plot_signals()

def plot_signals():
    speed = signal_type.get() == 1
    person = person_type.get()
    label_entries = vl.video.label_entries
    right_wrist = rw_bool.get()
    left_wrist = lw_bool.get()
    right_ankle = ra_bool.get()
    left_ankle = la_bool.get()
    right_elbow = re_bool.get()
    left_elbow = le_bool.get()
    right_knee = rk_bool.get()
    left_knee = lk_bool.get()
    y_lim = int(y_lim_str.get())
    max = y_lim
    i = 1
    plt.clf()
    legend_list = []
    legend_string = []
    ls = ''

    colors = ['blue', 'orange', 'red', 'green', 'gray']
    index = 0

    c = 10
    while c < vl.video.length:
        l1 = plt.vlines(x=c, ymin=0, ymax=max,
                        colors='orange',
                        ls='--',
                        label=ls)
        c += 10

    #ls = 'Observed Action Start/End'
    #l1 = plt.vlines(x=54, ymin=0, ymax=max,
    #                colors='purple',
    #                ls='--',
    #                label=ls)
    #l1 = plt.vlines(x=101, ymin=0, ymax=max,
    #                colors='purple',
    #                ls='--',
    #                label=ls)
    #legend_list.append(l1)
    #legend_string.append(ls)

    ls = 'Annotated Action Start/End'
#    l1 = plt.vlines(x=vl.video.label_entries[2].start, ymin=0, ymax=max, colors='blue', ls='--', label=ls)
#    l1 = plt.vlines(x=vl.video.label_entries[2].end, ymin=0, ymax=max, colors='blue', ls='--', label=ls)
#    legend_list.append(l1)
#    legend_string.append(ls)


    l1 = plt.vlines(x=label_entries[len(label_entries)-1].end + 80, ymin=0, ymax=max,
                    colors='white',
                    ls='--',
                    label=ls)


    for tp in vl.video.tracked_persons.values():

        if i == person or person == 3:
            #legend_list.append(l1)
            #legend_string.append(ls)
            pre = 'Person ' + str(i) + ' - '

            if right_wrist:
                ls = 'Right Wrist'
                do_plot_sig(tp.right_wrist, ls, speed, legend_list, legend_string, pre)
            if left_wrist:
                ls = 'Left Wrist'
                do_plot_sig(tp.left_wrist, ls, speed, legend_list, legend_string, pre)
            if left_ankle:
                ls = 'Left Ankle'
                do_plot_sig(tp.left_ankle, ls, speed, legend_list, legend_string, pre)
            if right_ankle:
                ls = 'Right Ankle'
                do_plot_sig(tp.right_ankle, ls, speed, legend_list, legend_string, pre)
            if left_elbow:
                ls = 'Left Elbow'
                do_plot_sig(tp.left_elbow, ls, speed, legend_list, legend_string, pre)
            if right_elbow:
                ls = 'Right Elbow'
                do_plot_sig(tp.right_elbow, ls, speed, legend_list, legend_string, pre)
            if left_knee:
                ls = 'Left Knee'
                do_plot_sig(tp.left_knee, ls, speed, legend_list, legend_string, pre)
            if right_knee:
                ls = 'Right Knee'
                do_plot_sig(tp.right_knee, ls, speed, legend_list, legend_string, pre)

            vid = vl.video.file_name.replace('./ShakeFive2/', '').replace('.mp4', '')
            if speed:
                plt.title('Person speed signals, wrists and ankles: ' + vid)
                plt.ylabel('Speed px/33.33ms')
            else:
                plt.ylabel('Acceleration px/33.33ms^2')
                plt.title('Person acceleration signals, wrists and ankles')
        i += 1
    plt.xlabel('Frame number')
    plt.legend(legend_list, legend_string)
    plt.ylim(0, y_lim)


def do_plot_sig(joint, ls, speed, legend_list, legend_string, pre):
    count = 1
    #point_to_plot = int(vl.video.length)
    point_to_plot = int(vl.curr_frame)

    for s in joint.subsigs:
        X = range(s.start_frame, point_to_plot + 1)
        if s.start_frame <= point_to_plot:
            length = point_to_plot + 1 - s.start_frame
            if point_to_plot > s.end_frame:
                length = s.end_frame + 1 - s.start_frame
                X = s.x_axis()

            a = array(s.speed_smooth[:length])
            if len(a) < len(X):
                X = range(s.start_frame, s.start_frame + len(a))
            if a.shape[0] == 0:
                print(str(a.shape) + " -- " + str(len(X)))
            ls = ls + '-' + str(count)
            if speed:
                l, = plt.plot(X, s.speed_smooth[:length], label=ls)
            else:
                l, = plt.plot(X, s.accel_smooth[:length], label=ls)
            legend_list.append(l)
            legend_string.append(pre + ls)


def mainGUI():

    y_val = 800
    i = 0
    global image_type
    Label(root, text="Video Path:").place(x=40, y=20)
    Button(root, text='Load', command=lambda: get_loader()).place(x=600, y=18)

    Button(root, text='Back', command=lambda: get_prev()).place(x=40, y=y_val)
    Button(root, text='Play', command=lambda: play()).place(x=125, y=y_val)
    Button(root, text='Forward', command=lambda: get_next()).place(x=210, y=y_val)

    Label(root, text="FPS:").place(x=400, y=800)

    Checkbutton(root, text='Right Wrist', variable=rw_bool, onvalue=True, offvalue=False, command=plot_signals).place(x=450, y=825)
    Checkbutton(root, text='Left Wrist', variable=lw_bool, onvalue=True, offvalue=False, command=plot_signals).place(x=550, y=825)
    Checkbutton(root, text='Right Ankle', variable=ra_bool, onvalue=True, offvalue=False, command=plot_signals).place(x=650, y=825)
    Checkbutton(root, text='Left Ankle', variable=la_bool, onvalue=True, offvalue=False, command=plot_signals).place(x=750, y=825)

    Checkbutton(root, text='Right Elbow', variable=re_bool, onvalue=True, offvalue=False, command=plot_signals).place(x=450, y=850)
    Checkbutton(root, text='Left Elbow', variable=le_bool, onvalue=True, offvalue=False, command=plot_signals).place(x=550, y=850)
    Checkbutton(root, text='Right Knee', variable=rk_bool, onvalue=True, offvalue=False, command=plot_signals).place(x=650, y=850)
    Checkbutton(root, text='Left Knee', variable=lk_bool, onvalue=True, offvalue=False, command=plot_signals).place(x=750, y=850)
    i = 0
    for (text, value) in graph_values.items():
        Radiobutton(root, text=text, variable=signal_type,
                    value=value).place(x=900, y=y_val+i*20)
        i += 1
    i = 0
    for (text, value) in graph_person_values.items():
        Radiobutton(root, text=text, variable=person_type,
                    value=value).place(x=1200, y=y_val + i * 20)
        i += 1

    global video_str
    global vl
    file = video_str.get()
    vl = VideoLoader(file, device)
    get_loader()

    my_button = Button(root, text="X", command=close)
    my_button.place(x=1300, y=20)

    root.mainloop()
    print('bye')

def close():
    root.destroy()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    dir = "./ShakeFive2/"

    mainGUI()






