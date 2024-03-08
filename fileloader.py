
from trackedperson import TrackedPerson

frame_x2 = 640
frame_y2 = 360

def is_prob(value):
    return value > 0.90


def append_keypoint(s, values, person_headers, offset, i, count):
    s.x.append(float(values[i * person_headers + offset]))
    s.y.append(float(values[i * person_headers + offset + 1]))
    s.prob.append(float(values[i * person_headers + offset + 2]))
    s.u.append(float(values[i * person_headers + offset + 5]))
    s.v.append(float(values[i * person_headers + offset + 6]))
    s.end_frame = count

def get_bounding_box(values, person_headers, offset, i):
    x2 = float(values[i * person_headers + offset + 2])
    y2 = float(values[i * person_headers + offset + 3])
    bbox = [float(values[i * person_headers + offset]),
                 float(values[i * person_headers + offset + 1]), x2, y2]
    result = True
    if x2 >= frame_x2 or y2 >= frame_y2:
        result = False

    return bbox, result

def load_people(file_name, fd):
    f = open(file_name, newline='')
    f.readline()
    f.readline()
    person_count_index = 2272
    person_id_offset = 2
    person_bb_offset = 3
    left_wrist_x_offset = 125
    right_wrist_x_offset = 138
    left_elbow_x_offset = 99
    right_elbow_x_offset = 112

    left_ankle_x_offset = 203
    right_ankle_x_offset = 216
    left_knee_x_offset = 177
    right_knee_x_offset = 190
    head_x_offset = 8
    person_headers = 227

    done = False
    tracked_persons = {}
    count = 2
    split = False
    while not done:
        line = f.readline()
        person = None

        if line is None or len(line) == 0:
            done = True
            continue

        values = line.strip().split(',')
        le = len(values)
        if le == 1:
            return tracked_persons, count
        elif le == 3:
            pc = 0
        elif le == 2273:
            pc = 10
        elif le == 0:
            pc = 0
        else:
            pc = int(values[le-1])
        count += 1
        fd.frame_ref[count] = []

        for i in range(0, pc):
            id = values[i * person_headers + person_id_offset]
            id = id.strip()
            if not (id in tracked_persons.keys()):
                tracked_persons[id] = TrackedPerson()
                person = tracked_persons[id]
                person.start_frame = count
                person.end_frame = count-1
                person.id = id
            person = tracked_persons[id]
            person.end_frame = count
            fd.frame_ref[count].append(person)

            bbox, result = get_bounding_box(values, person_headers, person_bb_offset, i)
            person.bounding_boxes.append(bbox)
            append_keypoint(person.left_wrist.get_create_subsig(count, True), values, person_headers, left_wrist_x_offset, i, count)
            append_keypoint(person.right_wrist.get_create_subsig(count, True), values, person_headers, right_wrist_x_offset, i, count)
            append_keypoint(person.left_ankle.get_create_subsig(count, result), values, person_headers, left_ankle_x_offset, i, count)
            append_keypoint(person.right_ankle.get_create_subsig(count, result), values, person_headers, right_ankle_x_offset, i, count)

            append_keypoint(person.left_elbow.get_create_subsig(count, result), values, person_headers, left_elbow_x_offset, i, count)
            append_keypoint(person.right_elbow.get_create_subsig(count, result), values, person_headers, right_elbow_x_offset, i, count)
            append_keypoint(person.left_knee.get_create_subsig(count, result), values, person_headers, left_knee_x_offset, i, count)
            append_keypoint(person.right_knee.get_create_subsig(count, result), values, person_headers, right_knee_x_offset, i, count)
            append_keypoint(person.head.get_create_subsig(count, result), values, person_headers, head_x_offset, i, count)

    return tracked_persons, count
