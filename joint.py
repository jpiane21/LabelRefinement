import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

from kernel import Kernel
from kernel import Smooth
from kernel import Smooth2
from kernel import *
from kernel import Smooth4
import numpy

thres = 0.0
gap_thres = 3

class SubSig:
    def __init__(self, frame):
        self.start_frame = frame
        self.end_frame = 0
        self.filtered = False

        self.x = []
        self.y = []
        self.u = []
        self.v = []
        self.prob = []

        self.corr = [self.x, self.y, self.u, self.v]

        self.speed_raw = []
        self.accel_raw = []
        self.speed_smooth = []
        self.accel_smooth = []
        self.speed_smooth_sg = []
        self.accel_smooth_sg = []
        self.accel_der = []
        self.calced = False

        self.visible = []

    def x_axis(self):
        return range(self.start_frame, self.end_frame+1)

    def split(self, index):
        orig_index = index

        not_end = False
        for i in range(orig_index, self.size()):
            if self.prob[i] < thres:
                index += 1
            else:
                not_end = True
                break;
        if not_end:
            newsig = SubSig(self.start_frame + index)
            newsig.x.extend(self.x[index:])
            newsig.y.extend(self.y[index:])
            newsig.u.extend(self.u[index:])
            newsig.v.extend(self.v[index:])
            newsig.prob.extend(self.prob[index:])
            newsig.end_frame = self.end_frame
        else:
            newsig = None

        del self.x[orig_index:]
        del self.y[orig_index:]
        del self.u[orig_index:]
        del self.v[orig_index:]
        del self.prob[orig_index:]
        self.end_frame = self.start_frame + orig_index - 1

        return self, newsig, self.end_frame < self.start_frame

    def size(self):
        return self.end_frame - self.start_frame + 1

    def trim(self):
        if True in self.visible:
            start_index = self.visible.index(True)
        else:
            start_index = len(self.visible)
        if start_index > 0:
            del self.x[:start_index]
            del self.y[:start_index]
            del self.u[:start_index]
            del self.v[:start_index]
            del self.prob[:start_index]
            del self.visible[:start_index]
            self.start_frame += start_index

        l = len(self.visible)

        if l > 0:
            end_index = l - self.visible[::-1].index(True) - 1
            if end_index < l-1:
                del self.x[end_index + 1:]
                del self.y[end_index + 1:]
                del self.u[end_index + 1:]
                del self.v[end_index + 1:]
                del self.prob[end_index + 1:]
                del self.visible[end_index + 1:]
        else:
            end_index = 0
        self.end_frame = self.start_frame + end_index

class Joint:
    def __init__(self):
        self.calced = False
        self.gaps = []
        self.subsigs = []

    def get_subsig_at_range(self, start, end):
        sigs = []
        for ss in self.subsigs:
            if start < ss.end_frame and end > ss.start_frame:
                sigs.append(ss)
        return sigs

    def trim(self):
        for ss in self.subsigs:
            ss.trim()

    def get_create_subsig(self, frame, vis):
        if len(self.subsigs) > 0:
            s = self.subsigs[len(self.subsigs)-1]
            if s.end_frame == frame - 1:
                s.visible.append(vis)
                return s
        s = SubSig(frame)
        self.subsigs.append(s)
        s.visible.append(vis)
        return s

    def calc_sigs(self):
        not_done = True
        while not_done:
            not_done = self.filter()
        self.prune_small()
        #self.merge_small_gaps()
        self.calc()

    def filter(self):
        for i in range(0, len(self.subsigs)):
            if not self.subsigs[i].filtered:
                sz = self.subsigs[i].size()
                for j in range(0, sz):
                    if self.subsigs[i].prob[j] < thres:
                        old, new, rem_orig = self.subsigs[i].split(j)
                        old.filtered = True
                        if new is not None:
                            self.subsigs.insert(i+1, new)
                        if rem_orig:
                            del self.subsigs[i]
                            i -= 1
                        return True
                self.subsigs[i].filtered = True
        return False

    def prune_small(self):
        for i in reversed(range(0, len(self.subsigs))):
            size = self.subsigs[i].size()
            if size <= 1:
                del self.subsigs[i]

    def merge_small_gaps(self):
        for i in reversed(range(len(self.subsigs))):
            size = self.subsigs[i].start_frame - self.subsigs[i-1].end_frame - 1
            if size < gap_thres:
                for j in range(0, size):
                    for c in self.subsigs[i - 1].corr:
                        c.append(-1)
                    self.subsigs[i - 1].prob.append(0.0)
                for c in range(0, len(self.subsigs[i].corr)):
                    self.subsigs[i - 1].corr[c].extend(self.subsigs[i].corr[c])
                self.subsigs[i - 1].prob.extend(self.subsigs[i].prob)
                self.subsigs[i - 1].end_frame = self.subsigs[i].end_frame
                del self.subsigs[i]
                for c in self.subsigs[i - 1].corr:
                    self.interpolate(c)

    def calc(self):
        if self.calced:
            return

        for s in self.subsigs:
            self.calc_one_sig(s.x, s.y, s.u, s.v, s.speed_raw, s.accel_raw)
            s.accel_smooth_sg = self.sg_smooth_it(s.accel_raw, Smooth)
            s.accel_smooth = self.smooth_it(s.accel_raw, Smooth)
            s.speed_smooth_sg = self.sg_smooth_it(s.speed_raw, Smooth4)
            s.speed_smooth = self.smooth_it(s.speed_raw, Smooth4)
            s.accel_der = self.der_it(s.speed_raw)
            s.calced = True
        self.calced = True

    def is_joint_here(self, framenum):
        for s in self.subsigs:
            if s.start_frame <= framenum <= s.end_frame:
                return True
        return False

    @staticmethod
    def calc_one_sig(rpx, rpy, rpu, rpv, out_sig_spd_raw, out_sig_acc_raw):
        if len(rpx) == len(rpy):
            for i in range(1, len(rpx)):
                a = rpx[i] - rpx[i - 1]
                b = rpy[i] - rpy[i - 1]
                c = math.sqrt(a * a + b * b)
                out_sig_spd_raw.append(c)
                if i == 1:
                    out_sig_spd_raw.append(c)
            if len(out_sig_spd_raw) > 0:
                out_sig_spd_raw.append(out_sig_spd_raw[len(out_sig_spd_raw)-1])
        else:
            print('ERROR!!!!!')
            assert (False)

        index = 0
        if len(rpu) == len(rpv):
            for i in range(0, len(rpu)):
                a = rpu[i]
                b = rpv[i]
                c = math.sqrt(a * a + b * b)
                out_sig_acc_raw.append(c)
        else:
            print('ERROR!!!!!')
            assert(False)

    def sg_smooth_it(self, sig, K):
        return gaussian_filter1d(sig, 2)


    def smooth_it(self, sig, K):

        if len(sig) > 9:
            n = 7
            ret = numpy.cumsum(sig, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            ret = ret[n - 1:] / n
            val = ret[0]
            ret = numpy.insert(ret, 0, [sig[0], sig[1]])
            ret = numpy.append(ret, [sig[len(ret)], sig[len(ret) + 1]])
            return ret
        else:
            return  sig


    def der_it(self, sig):
        if len(sig) > 2:
            ret = numpy.diff(sig)
            return numpy.insert(ret, 0, 0)
        else:
            return sig


    @staticmethod
    def interpolate(s):
        good = True
        count = 0
        for i in range(0, len(s)):
            r = s[i]
            if s[i] == -1.0:
                if good:
                    good = False
                    count = 1
                else:
                    count += 1
            elif not good:
                # i to i-count
                if i - count > 0:
                    start = s[i - count - 1]
                    end = s[i]
                    step = (end - start) / (count + 1)
                    curr = start + step
                    for j in range(i - count, i):
                        s[j] = round(curr, 4)
                        curr += step
                count = 0
                good = True
