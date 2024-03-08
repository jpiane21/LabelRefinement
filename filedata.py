class FileData:

    def __init__(self):
        self.file_name = ""
        self.file_name_full = ""
        self.tracked_persons = {}
        self.frame_count = 0
        self.frame_ref = {}

    def build_frame_ref(self):
        for i in range(1, self.frame_count):
            self.frame_ref[i] = []
            for tp in self.tracked_persons.values():
                if tp.is_person_here(i):
                    self.frame_ref[i].append(tp)

