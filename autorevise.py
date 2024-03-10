from videoloader import *
import sys
import numpy
import math

stand_threshold = 0.25
move_threshold = 1
sortof_move_threshold = 0.5


def autorevise(video_path):
    vl = VideoLoader(video_path, "cpu")
    stand, approach, action, plat, nearest_valley, limits, leave = stand_to_approach(vl)

    start, end = get_step_gap(vl.video.tracked_persons.values(), action,
                              vl.video.label_entries[len(vl.video.label_entries) - 1].end)

    intersections = []
    intersections2 = []
    intersections3 = []
    if action.label == Label.HAND_SHAKE or action.label == Label.FIST_BUMP:
        approach, action, leave = find_contact(vl.video.tracked_persons.values(), approach, action, leave)
    elif action.label == Label.HIGH_FIVE :
        approach, action, leave = find_arm_spike_contact(vl.video.tracked_persons.values(), approach, action, leave)
    elif action.label == Label.HUG:
        approach, action, leave = find_occlusion_based(vl.video.tracked_persons.values(), approach, action, leave)
    #    approach, action, leave = find_arm_valley_contact(vl.video.tracked_persons.values(), approach, action, leave)
    else:
        approach, action, leave, intersections, intersections2, intersections3 = apporach_to_action(approach, action, leave, vl)
        action, leave = action_to_leave(approach, action, leave, vl, [start, end])

    stand2 = None
    if len(vl.video.label_entries) == 5:
        leave, stand2 = leave_to_stand(leave, vl.video.label_entries[4], vl.video.tracked_persons.values(), vl.video.length -1)
    else:
        stand2 = LabelRange(Label.STAND, vl.video.label_entries[len(vl.video.label_entries)-1].end+1, vl.video.length -1)
        leave, stand2 = leave_to_stand(leave, stand2, vl.video.tracked_persons.values(), vl.video.length -1)

    if stand2 is None:
        s = leave.end
        s = leave.end
    else:
        s = stand2.start
        s = stand2.start

    return stand, approach, action, leave, stand2, intersections, intersections2, intersections3,  [start, end]

def leave_to_stand(leave, stand2, persons, video_end):
    sigs = []
    new_start = video_end
    stand2.end = video_end
    for tp in persons:
        # tp.trim_ankles();
        subsigs = tp.right_ankle.get_subsig_at_range(leave.start, video_end)
        subsigs.extend(tp.left_ankle.get_subsig_at_range(leave.start, video_end))
        for s in subsigs:
            sigs.append(s)

    frame_1 = leave.start
    frame_2 = leave.start
    frame_3 = leave.start
    frame = leave.start
    frame_4 = leave.start
    thres = 0.0
    for s in sigs:
        peaks = find_all_peaks(s, 1)
        results = list(filter(lambda x: x[1] > 2.75, peaks))

        if len(results) > 0:
            index = results[len(results)-1][2]
            frame_4 = max(index + s.start_frame - 1, frame_4)


        frame = max(frame_4, frame)

    leave.end = frame - 1
    stand2.start = frame

    if stand2.end -stand2.start < 3:
        leave.end = stand2.start
        stand2 = None

    return leave, stand2


def find_occlusion_based(persons, approach, action, leave):

    occ_set = []
    occ_s = approach.end
    occ_e = action.end,

    gaps = []
    for tp in persons:
        gaps.extend(find_gaps(tp.right_wrist))
        gaps.extend(find_gaps(tp.left_wrist))

    results = list(filter(lambda x: x[0] < leave.end, gaps))
    results = sorted(results, key=lambda x: x[0])
    l = len(results)
    if l > 0:
        occ_s = results[0][0]
        results = sorted(results, key=lambda x: x[1])
        occ_e = results[l-1][1]

    approach.end = occ_s
    action.start = occ_s + 1
    action.end = occ_e
    leave.start = occ_e + 1
    return approach, action, leave

def find_gaps(joint):
    gaps = []

    for i in range(0, len(joint.subsigs)-1):
        gaps.append([joint.subsigs[i].end_frame, joint.subsigs[i+1].start_frame, joint.subsigs[i+1].start_frame - joint.subsigs[i].end_frame])

    return gaps

def find_arm_valley_contact(persons, approach, action, leave):

    wrists = [[], []]
    sigs = []
    start_frame = 0
    end_frame = 0
    for person in persons:
        #if length < person.end_frame:
            #length = person.end_frame
        wrists[0].append(person.right_wrist)
        wrists[1].append(person.left_wrist)
        sigs.extend(person.right_wrist.subsigs)
        sigs.extend(person.left_wrist.subsigs)
        start_frame = person.start_frame
        end_frame = person.end_frame

    start = -1
    end = -1
    index = 0;
    for person in persons:
        subsigsr = person.right_wrist.get_subsig_at_range(approach.start, leave.end)
        subsigsl = person.left_wrist.get_subsig_at_range(approach.start, leave.end)
        start, end, next_valley, prev_valley = find_occlusion(subsigsr, subsigsl, approach, leave)
        if start != 0 and end != 0:
            approach.end = prev_valley
            action.start = prev_valley + 1
            action.end = find_hand_drop(prev_valley + 1, wrists[index])
            leave.start = action.end + 1
            break
        index += 1

    return approach, action, leave

def find_arm_spike_contact(persons, approach, action, leave):
    wrists = [[], []]
    length = 0
    start_frame = approach.start
    end_frame = leave.end

    #find wrists with max peak
    m = None
    right_person = []
    i = 0
    for person in persons:
        if length < person.end_frame:
            length = person.end_frame
        wrists[0].append(person.right_wrist)
        wrists[1].append(person.left_wrist)
        sigs = []
        sigs.extend(person.right_wrist.subsigs)
        sigs.extend(person.left_wrist.subsigs)

        right = True
        for s in person.right_wrist.subsigs:
            tmax = find_max_peak((find_all_peaks(s)))
            if not tmax is None:
                if m is None or tmax[1] > m[1]:
                    m = tmax
                    right = True

        for s in person.left_wrist.subsigs:
            tmax = find_max_peak((find_all_peaks(s)))
            if not tmax is None:
                if m is None or tmax[1] > m[1]:
                    m = tmax
                    right = False
        right_person.append(right)
    start, end = connected_wrists_expand(m[2], wrists[0] if right else wrists[1], start_frame, end_frame, "d:\\spatial.csv")
    if start == -1 and end == -1  :
        start, end = connected_wrists_expand(m[2], wrists[0] if not right else wrists[1], start_frame, end_frame, "d:\\spatial2.csv")

    if start == -1 and end == -1:
        for person in persons:
            subsigsr = person.right_wrist.get_subsig_at_range(approach.start, leave.end)
            subsigsl = person.left_wrist.get_subsig_at_range(approach.start, leave.end)
            start, end, next_valley, prev_valley = find_occlusion(subsigsr, subsigsl, approach, leave)
            if start != 0 and end != 0:
                break

    approach.end = start - 1
    action.start = start
    action.end = end
    leave.start = end + 1
    return approach, action, leave


def find_hand_drop(contact_section, wrist_set):
    #find y-valley forward from the end of contact
    #find y-peak backwards from valley
    #look for x-gap to increase and y values to drop

    p1 = wrist_set[0].get_subsig_at_range(contact_section, contact_section)[0]
    p2 = wrist_set[1].get_subsig_at_range(contact_section, contact_section)[0]
    p1_peak = find_y_peak(contact_section, p1)
    if p1_peak - p1.start_frame + 1 == len(p1.y):
        return contact_section
    p2_peak = find_y_peak(contact_section, p2)
    if p2_peak - p2.start_frame + 1 == len(p2.y):
        return contact_section
    peak = p1_peak if p1_peak < p2_peak else p2_peak
    return moving_apart(peak+1, p1, p2)


def moving_apart(peak, p1, p2):

    if peak - p1.start_frame + 1 >= len(p1.y) or peak - p2.start_frame + 1 >= len(p2.y):
        return peak - 1

    dist1 = abs(p1.x[peak - p1.start_frame] - p2.x[peak - p2.start_frame])
    dist2 = abs(p1.x[peak + 1 - p1.start_frame] - p2.x[peak + 1 - p2.start_frame])

    if p1.x[peak - p1.start_frame] < p1.x[peak + 1 - p1.start_frame]:  #moved right
        if p2.x[peak - p2.start_frame] < p2.x[peak + 1 - p2.start_frame]:
            return moving_apart(peak + 1, p1, p2)
        else:
            if dist1 < dist2:
                return peak + 1
            else:
                return moving_apart(peak + 1, p1, p2)
    else: #moved left
        if p2.x[peak - p2.start_frame] > p2.x[peak + 1 - p2.start_frame]:
            return moving_apart(peak + 1, p1, p2)
        else:
            if dist1 < dist2:
                return peak + 1
            else:
                return moving_apart(peak + 1, p1, p2)
    return peak
    

def find_y_peak(point, person, y_val_prev=-1):
    if point - person.start_frame < 0 or point - person.start_frame >= len(person.y):
        return point - 1
    y_val = person.y[point - person.start_frame]
    if y_val_prev == -1:
        return find_y_peak(point-1, person, y_val)

    if y_val < y_val_prev:
        return find_y_peak(point - 1, person, y_val)
    return point


def find_y_valley(point, person, y_val_prev = -1):

    if point - person.start_frame < 0 or point - person.start_frame >= len(person.y):
        return point - 1
    y_val = person.y[point - person.start_frame]
    if y_val_prev == -1:
        return find_y_valley(point+1, person, y_val)
    if y_val > y_val_prev:  # in pixel coord
        return find_y_valley(point+1, person, y_val)
    return point

def choose_start_end(start, end, start_2, end_2, s_occ, e_occ):
    if s_occ and start_2 > -1:
        start = start_2
    if e_occ and end_2 > -1:
        end = end_2
    return start, end


def find_max_peak(peaks):
    if len(peaks) == 0:
        return None
    return max(peaks, key=lambda x: x[1])

def find_contact(persons, approach, action, leave):
    wrists = [[], []]
    length = 0
    for person in persons:
        if length < person.end_frame:
            length = person.end_frame
        wrists[0].append(person.right_wrist)
        wrists[1].append(person.left_wrist)
    res = []
    for wrist in wrists:
        l = analyze_wrists(wrist, length, approach.start)
        l = list(filter(lambda x: (x[1] > action.start and x[0] < action.end), l))
        s = sorted(l, key=lambda x: x[1] - x[0], reverse=True)
        if len(s) > 0:
            res.append(s[0])
        else:
            res.append([0, 0])

    index = -1
    if res[0][1] - res[0][0] == res[1][1] - res[1][0]:
        print('weird')
    elif res[0][1] - res[0][0] > res[1][1] - res[1][0]:
        index = 0
    else:
        index = 1

    con = res[index]
    start, end = connected_wrists_close(res[index], wrists[index])
    con[0] = start if start != -1 else con[0]
    con[1] = end if end != -1 else con[1]

    if index > -1:
        approach.end = con[0] - 1
        action.start = con[0]
        action.end = con[1]
        leave.start = con[1] + 1

    return approach, action, leave


def has_contact(point, wrist_set, prev, fie):
    contact_thres = 2.5
    contact_range_thres = 25
    y_diff_thres = 8.0
    p1 = wrist_set[0].get_subsig_at_range(point, point)
    p2 = wrist_set[1].get_subsig_at_range(point, point)

    if len(p1) > 0 and len(p2) > 0:
        fie.write(str(point) + ', ')
        fie.write(str(p1[0].x[point - p1[0].start_frame])  + ", ")
        fie.write(str(p2[0].x[point - p2[0].start_frame])  + ", ")
        fie.write(str(p1[0].y[point - p1[0].start_frame])  + ", ")
        fie.write(str(p2[0].y[point - p2[0].start_frame])  + "\n")

        y_diff = abs(p1[0].y[point - p1[0].start_frame] - p2[0].y[point - p2[0].start_frame])
        curr = abs(p1[0].x[point - p1[0].start_frame] - p2[0].x[point - p2[0].start_frame])
        return abs(prev - curr) < contact_thres and curr < contact_range_thres and y_diff < y_diff_thres, curr, True
    return False, prev, True

def find_contact_point(section, wrist_set, start_frame, end_frame, name):
    fie = open(name, 'w')
    contact = -1
    prev = 99999.0
    while contact == -1:
        for i in range(section[0], section[1]):
            hc, curr, occ = has_contact(i, wrist_set, prev, fie)
            prev = curr
            if hc:
                contact = i
                break
        if section[0] == start_frame and section[1] == end_frame:
            break
        if section[0] > 5:
            section[0] = section[0] - 5
        else:
            section[0] = start_frame
        if section[1] > end_frame:
            section[1] = section[1] + 5
        else:
            section[1] = end_frame

    return contact, fie, prev, curr

def connected_wrists_expand(point, wrist_set, start_frame, end_frame, name):
    start = -1
    prev = 999999.0
    end = -1
    contact = -1
    section = [point - 5, point + 5]
    #find contact
    contact, fie, prev, curr = find_contact_point(section, wrist_set, start_frame, end_frame, name)

    #region of contact
    curr_s = curr
    if contact != -1:
        for i in range(contact, end_frame):
            hc, curr, occ = has_contact(i, wrist_set, prev, fie)
            prev = curr
            if not hc:
                end = i+1
                break
        for i in reversed(range(start_frame, contact+1)):
            hc, curr, occ = has_contact(i, wrist_set, prev, fie)
            prev = curr
            if not hc:
                start = i-1
                break
    else:
        fie.close()
        return -1, -1

    fie.close()
    return start, end


def connected_wrists_close(section, wrist_set):
    contact_thres = 1.5
    start = -1
    prev = 999999.0
    end = -1
    for i in range(section[0], section[1]):

        p1 = wrist_set[0].get_subsig_at_range(i, i)
        p2 = wrist_set[1].get_subsig_at_range(i, i)
        if len(p1) == 0 or len(p2) == 0:
            start = i
            break
        curr = abs(p1[0].x[i - p1[0].start_frame] - p2[0].x[i - p2[0].start_frame])
        if abs(prev - curr) < contact_thres:
            start = i - 1
            break
        prev = curr

    prev = 999999.0
    contact_range_thres = 25

    for i in reversed(range(section[0], section[1] + 1)):

        p1 = wrist_set[0].get_subsig_at_range(i, i)
        p2 = wrist_set[1].get_subsig_at_range(i, i)
        if len(p1) == 0 or len(p2) == 0:
            end = i
            break
        curr = abs(p1[0].x[i - p1[0].start_frame] - p2[0].x[i - p2[0].start_frame])

        if abs(prev - curr) < contact_thres and curr < contact_range_thres:
            end = i + 1
            break
        prev = curr

    return start, end


def connected_wrists2(section, person1, person2):
    start = -1
    for i in range(section[0], section[1]):
        if is_overlapped(person1.bounding_boxes[i-person1.start_frame], person2.bounding_boxes[i-person2.start_frame]):
            return i
    return start

def is_overlapped(box1, box2):
    x1min = min(box1[0], box1[2])
    y1min = min(box1[1], box1[3])
    x1max = max(box1[0], box1[2])
    y1max = max(box1[1], box1[3])

    x2min = min(box2[0], box2[2])
    y2min = min(box2[1], box2[3])
    x2max = max(box2[0], box2[2])
    y2max = max(box2[1], box2[3])

    return (x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max)


def connected_wrists(section, wrist_set):
    contact_thres = 18
    start = -1;
    for i in range(section[0], section[1]):
        p1 = wrist_set[0].get_subsig_at_range(i, i)
        p2 = wrist_set[1].get_subsig_at_range(i, i)
        if len(p1) == 0 or len(p2) == 0:
            return i
        if abs(p1[0].x[i - p1[0].start_frame] - p2[0].x[i - p2[0].start_frame]) < contact_thres:
            return i

    return start

def analyze_wrists(wrist_set, length, start):
    speed_at = []
    moving_together = numpy.zeros(length)
    sortof_moving_together = numpy.zeros(length)
    moving_together_inc_low = numpy.zeros(length)
    for i in range(0, length):
        speed_at.append([])
        if i >= start:
            for wrist in wrist_set:
                sigs = wrist.get_subsig_at_range(i, i)
                for s in sigs:
                    speed_at[i].append(s.speed_smooth[i-s.start_frame])
                    moving_together[i], sortof_moving_together[i], moving_together_inc_low[i] = is_moving_together(speed_at[i])
    segs = find_segments(moving_together_inc_low, 1)
    segs = remove_all_low(segs, moving_together)


    return segs
    #join_segs(mt_segs, find_segments(moving_together_inc_low))
    #, find_segments(sortof_moving_together)

def remove_all_low(segs, moving_together):
    res = []

    for s in segs:
        for i in range(s[0], s[1]):
            if moving_together[i] == 1:
                res.append(s)
                break
    return res
def is_moving_together(speeds):
    if len(speeds) == 0:
        return 0, 0, 0
    a = numpy.average(speeds)
    d = sum([abs(i - a) for i in speeds])
    mt = 1 if move_threshold > d and a > 1.0 else 0
    mtls = 1 if move_threshold > d else 0
    smt = 1 if sortof_move_threshold > d and a > 1.0 else 0
    return mt, smt, mtls

def find_segments(vect, look_for):
    count = 0
    s = 0
    sections = []
    for i in range(0, len(vect)):
        if vect[i] == look_for:
            if count == 0:
                s = i + 1
            count += 1
        else:
            if count != 0:
                if s > 1:
                    sections.append([s, i])
                count = 0
    return sections


def get_unoccluded_person(people, ankle):
    unoccluded_persons = []
    for person in people:
        if ankle and len(person.left_ankle.subsigs) == 1 and len(person.right_ankle.subsigs) == 1:
            unoccluded_persons.append(person)
        elif not ankle and len(person.left_wrist.subsigs) == 1 and len(person.right_wrist.subsigs) == 1:
            unoccluded_persons.append(person)
    return unoccluded_persons


def get_step_gap(people, action, last):
    start = action.start
    end = action.end

    unoccluded_persons = get_unoccluded_person(people, True)

    if len(unoccluded_persons) == 1:
        unoccluded = unoccluded_persons[0]
        left = find_all_peaks(unoccluded.left_ankle.subsigs[0])
        right = find_all_peaks(unoccluded.right_ankle.subsigs[0])

        while is_spike(max(left, key=lambda x: x[1])):
            left.remove(max(left, key=lambda x: x[1]))
        while is_spike(max(right, key=lambda x: x[1])):
            right.remove(max(right, key=lambda x: x[1]))


        left_max = max(left, key=lambda x: x[1])
        right_max = max(right, key=lambda x: x[1])
        real_max = right_max if right_max[1] > left_max[1] else left_max
        thres = real_max[1] * .25

        vect = [0] * last
        bin_sig(vect, unoccluded.right_ankle.subsigs[0], thres)
        bin_sig(vect, unoccluded.left_ankle.subsigs[0], thres)
        gap = choose_section(start, end, vect)

        if gap is not None:
            start = valley_after([left, right], gap[0], [unoccluded.left_ankle, unoccluded.right_ankle], thres)
            end = valley_before([left, right], gap[1], unoccluded, thres)
            if start >= end:
                start = gap[0]
                end = gap[1]


    return start, end


def choose_section(start, end, vect):
    sections = find_segments(vect, 0)
    mi = None
    ms = 0
    for gap in sections:
            if ms < gap[1] - gap[0]:
                ms = gap[1] - gap[0]
                mi = gap
    return mi


def valley_after(ankles, start, person_sigs, thres):
    for i in range(0, len(ankles)):
        for peak in ankles[i]:
            if peak[3] < start and peak[1] > thres:
                next_valley, curr_speed = find_next_valley(person_sigs[i].subsigs[0].speed_smooth, peak[0], peak[1], 1)
                if (next_valley + person_sigs[i].subsigs[0].start_frame) > start:
                    return next_valley + person_sigs[i].subsigs[0].start_frame
    return -1


def valley_before(ankles, end, person, thres):
    frame = person.left_ankle.subsigs[0].start_frame
    for ankle in ankles:
        for peak in ankle:
            if peak[0] + frame > end > peak[2] + frame:
                return peak[2] + frame

    return -1


def bin_sig(arr, subsig, thres):
    m = min(len(arr) - subsig.start_frame, len(subsig.speed_smooth))

    for i in range(m):
        if subsig.speed_smooth[i] > thres:
            arr[i + subsig.start_frame] = 1


def action_to_leave(approach, action, leave, vl, gap):
    occ_start = 0
    occ_end = 0
    for tp in vl.video.tracked_persons.values():
        subsigsr = tp.right_ankle.get_subsig_at_range(action.start, leave.end)
        subsigsl = tp.left_ankle.get_subsig_at_range(action.start, leave.end)

        if len(subsigsr) == 1 and len(subsigsl) == 1:
            last = None
            for s in tp.right_ankle.subsigs:
                if s.start_frame > subsigsr[0].end_frame:
                    if last is None or last.end_frame < s.start_frame:
                        last = s
            if last is not None:
                subsigsr.append(last)
            last = None
            for s in tp.left_ankle.subsigs:
                if s.start_frame > subsigsl[0].end_frame:
                    if last is None or last.end_frame < s.start_frame:
                        last = s
            if last is not None:
                subsigsl.append(last)

        occ_s, occ_e, next_valley, prev_valley = find_occlusion(subsigsr, subsigsl, approach, leave)
        if occ_s > 0:
            if occ_start == 0 or occ_s < occ_start:
                occ_start = occ_s
            if occ_e > occ_end:
                occ_end = occ_e


    person, otherperson = person_least_occ(vl.video.tracked_persons.values())
    if person is not None:
        subsigsr = otherperson.right_ankle.get_subsig_at_range(action.start, leave.end)
        subsigsl = otherperson.left_ankle.get_subsig_at_range(action.start, leave.end)
        occ_start, occ_end, next_valley, prev_valley = find_occlusion(subsigsr, subsigsl, approach, leave)

        lsig, rsig = get_last_ankles(person)
        lp = find_all_peaks(lsig)
        rp = find_all_peaks(rsig)

        # rp, lp, val1, val2, val3 = clear_noise(rp, lp)

        if occ_start > 0:
            remove_except_occ(lp, occ_start, occ_end, lsig.start_frame)
            remove_except_occ(rp, occ_start, occ_end, rsig.start_frame)
        else:
            remove_except_occ(lp, action.start, occ_end, lsig.start_frame)
            remove_except_occ(rp, action.start, occ_end, rsig.start_frame)

        start = leave.end
        if len(lp) > 0:
            start = lp[len(lp)-1][0] + lsig.start_frame
        if len(rp) > 0 and start < (rp[len(rp)-1][0] + rsig.start_frame) and start != leave.end:
            start = rp[len(rp)-1][0] + rsig.start_frame
        if start != leave.end:
            leave.start = start
            action.end = leave.start - 1
        else:
            start = max(lsig.start_frame, rsig.start_frame)
            if action.start < start:
                leave.start = start
                action.end = leave.start - 1

    else:
        end = 0
        for tp in vl.video.tracked_persons.values():
            subsigsr = tp.right_wrist.get_subsig_at_range(action.start, leave.end)
            subsigsl = tp.left_wrist.get_subsig_at_range(action.start, leave.end)
            if len(subsigsr) > 1 or len(subsigsl) > 1:
                occ_start, occ_end, next_valley, prev_valley = find_occlusion(subsigsr, subsigsl, approach, leave)
                if occ_end > 0 and occ_end > end:
                    end = occ_end
                    action.end = occ_end
                    leave.start = occ_end + 1
    return action, leave


def remove_except_occ(peaks, occ_start, occ_end, start_frame):
    while len(peaks) > 0 and (peaks[0][2] + start_frame) < occ_start:
        del peaks[0]

    index = 0
    while len(peaks) > index and (peaks[index][2] + start_frame) < occ_end:
        index += 1

    if index == len(peaks):
        return
    elif index == 0:
        peaks.clear()

    del peaks[index - 1:]


def get_last_ankles(person):
    return person.left_ankle.subsigs[len(person.left_ankle.subsigs) - 1], person.right_ankle.subsigs[
        len(person.right_ankle.subsigs) - 1]


def person_least_occ(tracked_persons):
    person = None
    otherperson = None
    least = sys.maxsize
    for tp in tracked_persons:
        ll = len(tp.left_ankle.subsigs) + len(tp.right_ankle.subsigs)
        if ll < least:
            otherperson = person
            person = tp
            least = ll
        elif ll > least:
            otherperson = tp
        elif ll == least:
            return None, None
    return person, otherperson


def apporach_to_action(approach, action, leave, vl):
    occ = None

    thres = sys.float_info.max
    la_peaks_ave = ra_peaks_ave = 0
    for tp in vl.video.tracked_persons.values():
        la_peaks = []
        ra_peaks = []
        for s in tp.left_wrist.subsigs:
            lp, t = gradual_limits_on_peaks(find_all_peaks(s))
            la_peaks.extend(lp)
        for s in tp.right_wrist.subsigs:
            lp, t = gradual_limits_on_peaks(find_all_peaks(s))
            ra_peaks.extend(lp)

        if (len(tp.right_wrist.subsigs) == 1 and len(tp.left_wrist.subsigs) == 1):
            if (len(la_peaks) > 0):
                la_peaks_ave = sum(map(lambda x: x[1], la_peaks)) / len(la_peaks)
            if (len(ra_peaks) > 0):
                ra_peaks_ave = sum(map(lambda x: x[1], ra_peaks)) / len(ra_peaks)

    right = True
    if (la_peaks_ave > ra_peaks_ave):
        right = False

    for tp in vl.video.tracked_persons.values():
        la_peaks = []
        ra_peaks = []
        for s in tp.left_ankle.subsigs:
            lp, t = gradual_limits_on_peaks(find_all_peaks(s))
            la_peaks.extend(lp)
            thres = min(t, thres)
        for s in tp.right_ankle.subsigs:
            lp, t = gradual_limits_on_peaks(find_all_peaks(s))
            ra_peaks.extend(lp)
            thres = min(t, thres)

    intersections = []
    for tp in vl.video.tracked_persons.values():
        intersections.extend(find_intersections(tp, thres, right))

    intersections.sort()

    intersections3 = []
    for tp in vl.video.tracked_persons.values():
        intersections3.extend(find_intersections(tp, thres, not right))

    intersections3.sort()

    intersections5 = intersections.copy()
    intersections5.extend(intersections3.copy())
    intersections5.sort()

    os = sys.maxsize
    oe = 0

    action_core_begin = sys.maxsize
    action_core_end = 0

    intersections2 = intersections.copy()
    intersections4 = intersections3.copy()
    for tp in vl.video.tracked_persons.values():
        subsigsr = tp.right_wrist.get_subsig_at_range(action.start, leave.end)
        subsigsl = tp.left_wrist.get_subsig_at_range(action.start, leave.end)
        if len(subsigsr) > 1 or len(subsigsl) > 1:
            occ_start, occ_end, next_valley, prev_valley = find_occlusion(subsigsr, subsigsl, approach, leave)
            if occ_start >= 0:
                intersections2 = [i for i in intersections2 if i < prev_valley or i > next_valley]
                intersections4 = [i for i in intersections4 if i < prev_valley or i > next_valley]
            os = min(os, prev_valley)
            oe = max(oe, next_valley)
        else:
            continue

    gap, s, e = find_largest_gap(intersections2)
    if s != 0 and e != 0:
        action.start = s
        action.end = e
    approach.end = action.start - 1
    leave.start = action.end
    return approach, action, leave, intersections2, intersections4, intersections5
    action_core_begin = os
    action_core_end = oe

    if not (oe == 0 or os == sys.maxsize):
        for i in range(0, len(intersections)):
            if intersections[i] < os:
                if i > 0:
                    approach.end = intersections[i - 1] - 1
                    action.start = intersections[i - 1]
            if intersections[i] > oe:
                action.end = intersections[i]
                leave.start = intersections[i] + 1
                return approach, action, leave, intersections2, intersections4, intersections5

        approach.end = os
        action.start = os + 1
        action.end = oe
        leave.start = oe + 1

    else:
        print('No interaction')
        # look for noticable spikes in hand motion
        subsigs = []
        for tp in vl.video.tracked_persons.values():
            subsigs.extend(tp.right_wrist.get_subsig_at_range(action.start, leave.end))
            subsigs.extend(tp.left_wrist.get_subsig_at_range(action.start, leave.end))
        all_peaks = []
        for s in subsigs:
            peaks = find_all_peaks(s)
            lp, thres = gradual_limits_on_peaks(peaks)
            all_peaks.extend(lp)
        all_peaks.sort(key=lambda x: x[3])
        peaks2, thres = gradual_limits_on_peaks(all_peaks)
        if (len(peaks2) != 0):
            peak = peaks2[len(peaks2) - 1]
            for i in range(0, len(intersections)):
                if intersections[i] > peaks2[0][3]:
                    if i > 0:
                        approach.end = intersections[i - 1] - 1
                        action.start = intersections[i - 1]
                if intersections[i] > peak[3]:
                    action.end = intersections[i]
                    leave.start = intersections[i] + 1
                    return approach, action, leave, intersections2, intersections4, intersections5
    return approach, action, leave, intersections2, intersections4, intersections5


def find_largest_gap(intersections):
    l = len(intersections)
    max_gap = 0
    approach = 0
    action = 0
    for i in range(1, l - 1):
        gap = intersections[i] - intersections[i - 1]
        if gap > max_gap:
            max_gap = gap
            approach = intersections[i - 1]
            action = intersections[i]
    return max_gap, approach, action


def find_intersections_side(ankle, wrist, tp, thres):
    intersections = []
    for i in range(tp.start_frame + 1, tp.end_frame):
        sub1 = ankle.get_subsig_at_range(i, i + 1)
        sub2 = wrist.get_subsig_at_range(i, i + 1)
        if len(sub1) != 1 or len(sub2) != 1:
            continue
        s1 = sub1[0]
        s2 = sub2[0]

        x1 = i - s1.start_frame
        x2 = i - s2.start_frame
        if len(s1.speed_smooth) <= x1 + 1 and len(s2.speed_smooth) <= x2 + 1:
            continue
        if s1.speed_smooth[x1] < thres or s1.speed_smooth[x1 + 1] < thres or \
                s2.speed_smooth[x2] < thres or s2.speed_smooth[x2 + 1] < thres:
            continue

        if not (s2.speed_smooth[x2] < s1.speed_smooth[x1] and s2.speed_smooth[x2 + 1] < s1.speed_smooth[x1 + 1]) and \
                not (s2.speed_smooth[x2] > s1.speed_smooth[x1] and s2.speed_smooth[x2 + 1] > s1.speed_smooth[x1 + 1]):
            intersections.append(i)

    return intersections


def find_intersections(tp, thres, right):
    if right:
        intersections = find_intersections_side(tp.right_ankle, tp.right_wrist, tp, thres)
    else:
        intersections = find_intersections_side(tp.left_ankle, tp.left_wrist, tp, thres)

    # intersections.sort()
    return intersections


def find_steps(tp):
    la_peaks = []
    ra_peaks = []
    for s in tp.left_ankle.subsigs:
        lp, thres = gradual_limits_on_peaks(find_all_peaks(s))
        la_peaks.extend(lp)
    for s in tp.right_ankle.subsigs:
        lp, thres = gradual_limits_on_peaks(find_all_peaks(s))
        ra_peaks.extend(lp)

    ##valleys close to zero
    ## valleys are similar in height


def stand_to_approach(vl):
    leave = None
    stand = vl.video.label_entries[0]
    approach = vl.video.label_entries[1]
    action = vl.video.label_entries[2]
    if len(vl.video.label_entries) > 3:
        leave = vl.video.label_entries[3]
    if stand.label != Label.STAND or approach.label != Label.APPROACH:
        if stand.label == Label.APPROACH:
            leave = action
            action = approach
            approach = stand
            stand = LabelRange(Label.STAND, 1, approach.start - 1)
        print('Label Error!')

    sigs = []
    new_start = sys.maxsize
    for tp in vl.video.tracked_persons.values():
        # tp.trim_ankles();
        subsigs = tp.right_ankle.get_subsig_at_range(stand.start, approach.end)
        subsigs.extend(tp.left_ankle.get_subsig_at_range(stand.start, approach.end))
        for s in subsigs:
            sigs.append(s)
            if s.end_frame < approach.end:
                print('found foot occlusion - ending approach early')
                approach.end = s.end_frame
                action.start = approach.end + 1

    frame_1 = approach.end
    frame_2 = approach.end
    frame_3 = approach.end
    frame_4 = approach.end
    for s in sigs:
        if new_start > s.start_frame:
            peaks = find_all_peaks(s)
            if (len(peaks) > 0):
                index = find_nearest_valley(peaks, stand.end)
                frame_1 = min(index, frame_1)

            peaks1 = set_limits_on_peaks(peaks)
            if len(peaks1) > 0:
                index = peaks1[0][2]
                frame_2 = min(index, frame_2)

            peaks2, thres = gradual_limits_on_peaks(peaks)
            if len(peaks2) > 0:
                index = peaks2[0][2]
                frame_3 = min(index, frame_3)

            frame = choose_frame(stand, [frame_1, frame_3], s)

            frame = frame + s.start_frame
            if frame < new_start:
                new_start = frame
            index = escape_plateau(frame_3, s.speed_smooth, 0.25) + s.start_frame
            frame_4 = min(index, frame_4)

    stand.end = new_start
    approach.start = new_start + 1
    s = sigs[0]
    plat = LabelRange(stand.label, stand.start, frame_4 + s.start_frame)
    nearest_valley = LabelRange(stand.label, stand.start, frame_1 + s.start_frame)
    limits = LabelRange(stand.label, stand.start, frame_3 + s.start_frame)
    stand.end = frame_4 + s.start_frame
    approach.start = frame_4 + s.start_frame + 1
    return stand, approach, action, plat, nearest_valley, limits, leave


def find_high(peaks):
    done = False
    if len(peaks) == 0:
        return None

    while not done:
        if (peaks[len(peaks) - 1][1] < peaks[len(peaks) - 2][1]):
            del peaks[len(peaks) - 1]
        else:
            done = True
    return peaks[len(peaks) - 1]


def clear_noise(peaksr, peaksl):
    lr = len(peaksr)
    ll = len(peaksl)
    if lr < 3 or ll < 3:
        return peaksr, peaksl, False, None, None
    if peaksr[lr - 1][1] < peaksr[lr - 2][1] < peaksr[lr - 3][1] \
            or peaksl[ll - 1][1] < peaksl[ll - 2][1] < peaksl[ll - 3][1]:
        return peaksr, peaksl, False, None, None
    peaksr, thres = gradual_limits_on_peaks(peaksr)
    peaksl, thres = gradual_limits_on_peaks(peaksl)

    peakr = find_high(peaksr)
    peakl = find_high(peaksl)
    return peaksr, peaksl, True, peakr, peakl


def find_occlusion(subsigsr, subsigsl, approach, leave):
    next_valley = 0
    prev_valley = 0
    subsigs = []
    subsigs.append(subsigsr)
    subsigs.append(subsigsl)
    occ_start = occ_end = 0
    l = len(subsigsr)
    peaksr = find_all_peaks(subsigsr[0])
    peaksl = find_all_peaks(subsigsl[0])
    peaksr, peaksl, rn, peakr, peakl = clear_noise(peaksr, peaksl)
    if subsigsr[0].end_frame <= leave.end and len(peaksr) > 0:
        occ_start = subsigsr[0].end_frame
        prev_valley = peaksr[len(peaksr) - 1][2]

        prev_valley += subsigsr[0].start_frame - 1

        occ_end = subsigsr[l - 1].start_frame
        # next_valley, curr_speed = find_next_valley(subsigsr[l-1].speed_smooth, 0, subsigsr[l-1].speed_smooth[0], 1)
        peaks = find_all_peaks(subsigsr[l - 1])
        next_valley, curr_speed = find_next_valley(subsigsr[l - 1].speed_smooth, peaks[0][0], peaks[0][1], 1)
        next_valley += subsigsr[l - 1].start_frame - 1

    l = len(subsigsl)
    if subsigsl[0].end_frame <= leave.end and len(peaksl) > 0:
        if occ_start > subsigsl[0].end_frame:
            occ_start = subsigsl[0].end_frame
        temp = peaksl[len(peaksl) - 1][2]
        temp += subsigsl[0].start_frame - 1
        if not rn and temp < prev_valley:
            prev_valley = temp
        if occ_end < subsigsl[l - 1].start_frame:
            occ_end = subsigsl[l - 1].start_frame
        peaks = find_all_peaks(subsigsl[l - 1])
        if len(peaks) > 0:
            temp, curr_speed = find_next_valley(subsigsl[l - 1].speed_smooth, peaks[0][0], peaks[0][1], 1)
            if temp > next_valley:
                next_valley = temp
    if peakr != None and peakl != None:
        if rn and peakr[1] > peakl[1]:
            prev_valley = peakr[2] + subsigsr[0].start_frame - 1
        elif rn and peakr[1] < peakl[1]:
            prev_valley = peakl[2] + subsigsl[0].start_frame - 1
    return occ_start, occ_end, next_valley, prev_valley


def choose_closest(frames, check_val):
    min_index = 0
    i = 0
    min_val = sys.maxsize

    for f in frames:
        d = abs(check_val - f)
        if d < min_val:
            min_val = d
            min_index = i
        i += 1

    return frames[min_index]

def choose_frame(label, frames, s):
    d = [abs(x - (label.end - s.start_frame)) for x in frames]
    closest = d.index(min(d))
    s.visible[frames[closest]]
    return frames[closest]




def set_limits_on_peaks(peaks):
    res = peaks.copy()
    l = int(len(res) * 0.25)
    res.sort(key=lambda x: x[1], reverse=True)
    result = list(filter(lambda x: x in res[0:l], peaks))
    result = list(filter(lambda x: x[0] > 10, result))
    return result


def gradual_limits_on_peaks(peaks, min_thres = 0.0):
    result = peaks.copy()
    thres = min_thres
    ending_length = starting_length = len(result)
    while starting_length * 0.5 <= ending_length and len(result) > 0:
        l = len(result) + 1
        while len(result) < l:
            thres += 0.5
            l = len(result)
            result = list(filter(lambda x: x[1] > thres, result))
        thres += 1.0
        ending_length = l
    result = list(filter(lambda x: x[0] > 10, result))
    return result, thres


def escape_plateau(pos, s, slope_threshold):
    # is slope near zero
    l = len(s)
    slope = 0
    while (abs(slope) < slope_threshold):
        if (l <= pos + 2):
            return pos - 1
        slope = (s[pos + 2] - s[pos]) / 2
        pos = pos + 1
    return pos - 1


def find_furthest_valley(peaks, loc):
    m_abs = 0
    max_index = 0
    for i in range(0, len(peaks)):
        val = abs(loc - peaks[i][2])
        peaks[i].append(val)
        if max_abs < val:
            max_abs = val
            max_index = i
    return peaks[max_index][2]


def find_nearest_valley(peaks, loc):
    min_abs = sys.maxsize
    min_index = 0
    for i in range(0, len(peaks)):
        val = abs(loc - peaks[i][2])
        peaks[i].append(val)
        if min_abs > val:
            min_abs = val
            min_index = i
    return peaks[min_index][2]


def find_change_in_peaks(peaks, s):
    diffs = []
    max = sys.minsize
    max_pos = -1
    min = sys.maxsize
    min_pos = -1

    prev = 0
    for i in range(1, len(peaks)):
        diff = peaks[i][1] - prev
        if diff > -1:
            if diff > max:
                max = diff
                max_pos = peaks[i][0]
            if diff < min:
                min = diff
                min_pos = peaks[i][0]
            prev = peaks[i][1]
        diffs.append(diff)
    return max_pos, diffs

def is_spike(peak):
    thres = 8.0
    s = peak[4]
    pos = peak[0]
    if 0 <= peak[0] <  len(s):
        if abs(s[pos - 1]  - s[pos])  < thres or abs(s[pos + 1]  - s[pos]) < thres:
            return False
    return True

def find_all_peaks(subsig, valley_dir = -1):
    peaks = []
    s = subsig.speed_smooth
    pos = 0
    while True:
        pos, curr_speed = find_next_peak(s, pos + 1)
        if pos == -1:
            break
        valley, speed = find_next_valley(s, pos, curr_speed, valley_dir)
        peaks.append([pos, curr_speed, valley, subsig.start_frame + pos, s])

    return peaks


def find_next_peak(s, pos):
    if len(s) < pos + 5:
        return -1, 0.0
    curr_speed = s[pos]
    if curr_speed >= s[pos + 1]:
        pos, curr_speed = find_next_valley(s, pos, curr_speed, 1)
        if pos == -1:
            return -1, 0.0
    while pos + 1 < len(s) and curr_speed <= s[pos + 1]:
        pos += 1
        curr_speed = s[pos]
    if pos == len(s) - 1:
        return -1, 0.0
    return pos, curr_speed


def find_prev_peak(s, pos, dir):
    if len(s) < pos + 5 * dir:
        return -1, 0.0
    curr_speed = s[pos]
    if curr_speed >= s[pos + 1 * dir]:
        pos, curr_speed = find_next_valley(s, pos, curr_speed, 1 * dir)
        if pos == -1:
            return -1, 0.0
    while pos + 1 < len(s) and curr_speed <= s[pos + 1 * dir]:
        pos += 1 * dir
        curr_speed = s[pos]
    if pos == len(s) - 1:
        return -1, 0.0
    return pos, curr_speed


def find_next_valley(s, pos, curr_speed, dir):
    while pos + 1 * dir < len(s) and pos - 1 * dir >= 0 and curr_speed > s[pos + 1 * dir]:
        pos += 1 * dir
        curr_speed = s[pos]
    if (pos + 1 >= len(s) and pos - 1 < 0):
        return -1, 0.0

    return pos, curr_speed
