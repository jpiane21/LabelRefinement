from videoloader import VideoLoader
from autorevise import *
import math

class Transition(enum.Enum):
    STAND_APPROACH = "stand-to-approach"
    APPROACH_ACTION = "approach-to-action"
    ACTION_LEAVE = "action-to-leave"

    @staticmethod
    def from_str(label):
        if label in ('stand-to-approach'):
            return Transition.STAND_APPROACH
        elif label in ('approach-to-action'):
            return Transition.APPROACH_ACTION
        elif label in ('action-to-leave'):
            return Transition.ACTION_LEAVE
        else:
            raise NotImplementedError
class LabelGroup:
    def __init__(self, values, action_lab):
        self.Approach = LabelRange(Label.APPROACH, int(values[0]), int(values[1]))
        self.Action = LabelRange(action_lab, int(values[2]), int(values[3]))
        self.Leave = LabelRange(Label.LEAVE, int(values[4]), int(values[5]))

    def get_label_range(self, lab):
        if lab == Label.APPROACH:
            return self.Approach
        if lab == self.Action.label:
            return self.Action
        if lab == Label.LEAVE:
            return self.Leave



class AutoRevisedLabel:
    def __init__(self, values):
        self.action = values[1]
        self.vid_file = values[0]
        self.action_lab = Label.from_str(self.action)
        self.original = LabelGroup(values[2:8], self.action_lab)
        self.revised = LabelGroup(values[8:14], self.action_lab)
        self.VL = None
        self.loaded = False

        self.better = ""

    def set_better(self, b):
        self.better = b

    def LoadVideo(self):
        if not self.loaded:
            self.VL = VideoLoader(self.vid_file, 'CPU')
            self.loaded = True

    def get_start(self, revised, lab):
        group = self.revised if revised else self.original

        if lab == Label.APPROACH:
            return group.Approach.start
        if lab == self.action_lab:
            return group.Action.start
        if lab == Label.LEAVE:
            return group.Leave.start
        return -1

    def get_end(self, revised, lab):
        group = self.revised if revised else self.original

        if lab == Label.APPROACH:
            return group.Approach.end
        if lab == self.action_lab:
            return group.Action.end
        if lab == Label.LEAVE:
            return group.Leave.end
        return -1

    def include(self, transition, threshold):
        start_orig = self.get_start(True, transition)
        start_revised = self.get_start(True, transition)
        if start_orig == -1 or start_revised == -1:
            return False
        elif abs(start_orig - start_revised) >= threshold:
            return True
        else:
            return False


    def get_scaled_clip(self, orig, transition):
        start = self.get_start(orig, transition)
        clip = []

        if start - 10 >= 1:
            for i in range(start - 10, start + 15):
                frame = clip.append(self.VL.get_scaled_frame_at(i))

        return clip

    def get_action_label(self):
        offset = 0
        if self.VL.video.label_entries[0] != Label.STAND:
            offset = -1
        return self.VL.video.label_entries[offset + 2].label
