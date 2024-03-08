import random
import shutil
from videoloader import *

class PlayBackInfo:
    def __init__(self):
        self.clip = [[],[]]
        self.curr_disp_indexes = [0, 0]
        self.playing_video = 0
        self.vid_list_index = -1
        self.orig_on_left = True
        self.fps = 5
        self.playing = False
        self.vid_list = []
        fileName = 'd:\\results.csv'
        if os.path.isfile(fileName):
            cp = self.find_next_file_name()
            shutil.copy(fileName, cp)

        self.fl = open(fileName, 'w')

    def find_next_file_name(self):
        i = 1
        while True:
            t = "d:\\saved_results\\results" + str(i) + ".csv"
            if not os.path.isfile(t):
                return t
            else:
                i += 1

    def record_better(self):
        ar = self.get_curr()
        if len(ar[1].better) > 0:
            self.fl.write(ar[1].vid_file + ", " + str(ar[0]) + ", " + ar[1].better + "\n")
            self.fl.flush()
            return True
        return False

    def close_file(self):
        self.fl.close()

    def get_clip_number(self):
        return self.vid_list_index

    def get_next(self):
        self.curr_disp_indexes = [0, 0]
        self.clip = [[],[]]
        self.vid_list_index += 1
        self.choose_loc()
        return self.get_curr()


    def get_curr(self):
        return self.vid_list[self.vid_list_index]


    def get_vl(self):
        return self.vid_list[self.vid_list_index][1].VL


    def choose_loc(self):
        k = random.randint(0, 1)
        self.orig_on_left = bool(k)


    def update_clip(self, vid):
        if self.curr_disp_indexes[vid] < 25:
            self.curr_disp_indexes[vid] += 1
        else:
            self.playing = False
            self.curr_disp_indexes[vid] = 0

    def get_start_frame(self, left):
        ar = self.vid_list[self.vid_list_index][1]
        lab = self.vid_list[self.vid_list_index][0]
        if self.orig_on_left == left:
            return ar.original.get_label_range(lab).start
        else:
            return ar.revised.get_label_range(lab).start

    def get_desc_text(self, lab):
        if lab == Label.LEAVE:
            return "The Leave activity begins when the people involved in the previous activity\ncomplete the interaction and walk away from each other"
        elif lab == Label.APPROACH:
            return "The Approach activity begins when the people involved in the activity start \nmoving towards each other"
        elif lab == Label.HAND_SHAKE:
            return "The Handshake activity begins when both people involved in the activity have made contact with their hands"
        elif lab == Label.FIST_BUMP:
            return "The Fist Bump activity begins when the people when both people involved\nin the activity have made contact with their fists"
        elif lab == Label.HUG:
            return "The Hug activity begins one of the people involved in the activity has touched the back of the other person"
        elif lab == Label.HIGH_FIVE:
            return "The High Five activity begins when the people when both people involved\nin the activity have made contact with their hands"

    def get_ann_text(self, lab):
        if lab == Label.LEAVE:
            return "Leave"
        elif lab == Label.APPROACH:
            return "Approach"
        elif lab == Label.FIST_BUMP:
            return "Fist Bump"
        elif lab == Label.HAND_SHAKE:
            return "Handshake"
        elif lab == Label.HUG:
            return "Hug"
        elif lab == Label.HIGH_FIVE:
            return "High Five"
