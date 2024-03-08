

from joint import Joint



class TrackedPerson:

    def __init__(self):
        self.id = -1

        self.start_frame = 0
        self.end_frame = 0

        self.bounding_boxes = []
        self.bboxes_entire_person = []
        self.right_wrist = Joint()
        self.left_wrist = Joint()
        self.right_ankle = Joint()
        self.left_ankle = Joint()
        self.right_elbow = Joint()
        self.left_elbow = Joint()
        self.right_knee = Joint()
        self.left_knee = Joint()
        self.head = Joint()




    def trim_ankles(self):
        self.right_ankle.trim()
        self.left_ankle.trim()

    def calc_signals(self):
        self.right_wrist.calc_sigs()
        self.left_wrist.calc_sigs()
        self.right_ankle.calc_sigs()
        self.left_ankle.calc_sigs()
        self.right_knee.calc_sigs()
        self.left_knee.calc_sigs()
        self.right_elbow.calc_sigs()
        self.left_elbow.calc_sigs()
        self.head.calc_sigs()

    def is_person_here(self, framenum):
        if self.start_frame <= framenum <= self.end_frame:
            return True
        return False

    def is_calced(self):
        return self.right_wrist.calced and self.left_wrist.calced and self.left_ankle.calced and self.right_ankle.calced

