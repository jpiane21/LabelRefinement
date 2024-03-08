import numpy as np

from shakefive2dataset import *
import random
import time
from ConvNet import SignalConvNet
from ResNet import ResNet101
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from copy import deepcopy

class ModelTraining():
    def __init__(self):
        self.Autorev = False
        self.Flow = True
        self.Sig = True
        self.do_train = True

        self.training_vids = []
        self.testing_vids = []
        self.device = "cpu"

        if torch.cuda.is_available():
            self.device = "cuda"

        self.num_epochs = 100
        self.batch_size = 1
        self.learning_rate = 0.001

        self.classes = ('STAND', 'APPROACH', 'HANDSHAKE', 'HUG',
                    'HIGH_FIVE', 'FIST_BUMP', 'LEAVE')
        self.MISSING = 8

        self.model_signal = None
        self.model_OF = None
        if self.Autorev:
            self.PATH_SIG = "./autorevise_model_sig.md"
            self.PATH_OF = "./autorevise_model_of.md"
        else:
            self.PATH_SIG = "./human_annotated_model_sig.md"
            self.PATH_OF = "./human_annotated_model_of.md"

        qual = "_ar_" if self.Autorev else "_or_"
        #self.loadvidlist("./training_set" + qual + ".txt", self.training_vids)
        #self.loadvidlist("./testing_set" + qual + ".txt", self.testing_vids)
        self.createvidlist()
        self.training_dataset_signals = ShakeFive2DatasetSignals(self.training_vids, self.device, 63, True)
        self.testing_dataset_signals = ShakeFive2DatasetSignals(self.testing_vids, self.device, 63, False)
        self.training_loader_signals = torch.utils.data.DataLoader(self.training_dataset_signals, batch_size=self.batch_size, shuffle=True)
        self.testing_loader_signals = torch.utils.data.DataLoader(self.testing_dataset_signals, batch_size=self.batch_size, shuffle=False)

        if self.Flow:
            self.training_dataset_flow = ShakeFive2DatasetFrames(self.training_vids, self.device, 2, True)
            self.testing_dataset_flow = ShakeFive2DatasetFrames(self.testing_vids, self.device, 2, False)
            self.training_loader_flow = torch.utils.data.DataLoader(self.training_dataset_flow, batch_size=self.batch_size, shuffle=True)
            self.testing_loader_flow = torch.utils.data.DataLoader(self.testing_dataset_flow, batch_size=self.batch_size, shuffle=False)
        #self.results_file = None


    def createvidlist(self):

        #with open("./training_list.txt", 'r') as fl:

        dict = {}
        with open("./autorevise_results.csv", 'r') as fl:
            for line in fl:
                vals = line.split(sep=',')
                ann = vals[1].strip()
                if not ann in dict.keys():
                    dict[ann] = []
                offset = 10 if self.Autorev else 2
                dict[ann].append(SampleVid(vals[0].strip(), ann, [eval(i) for i in vals[offset: offset+8]]))

            for li in dict.values():
                temp = random.sample(li, 13)
                self.testing_vids.extend(set(li) - set(temp))
                self.training_vids.extend(temp)

        qual = "_ar_" if self.Autorev else "_or_"

        self.savevidlist("./training_set" + qual + ".txt", self.training_vids)
        self.savevidlist("./testing_set" + qual + ".txt", self.testing_vids)

    def savevidlist(self, path, vidlist):
        with open(path, 'w') as fl:
            for sv in vidlist:
                fl.write(sv.file_name + "," + sv.ann)
                for set in range(0, int(len(sv.ann_ranges)/2)):
                    fl.write("," + str(sv.ann_ranges[set*2]) + "," + str(sv.ann_ranges[set*2 + 1]))
                fl.write("\n")

    def loadvidlist(self,path, vidlist):
        with open(path, 'r') as fl:
            for line in fl:
                vals = line.split(sep=',')
                fn = vals[0].strip()
                ann = vals[1].strip()
                ann_ranges = [eval(i) for i in vals[2: 10]]
                vidlist.append(SampleVid(fn, ann, ann_ranges))

    def train_one_model(self, dsl, crit, opt, model, passnum):
        n_total_steps = len(dsl)
        for epoch in range(self.num_epochs):
            for i, (images, labels, positioning_info) in enumerate(dsl):
                images = images.to(self.device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = crit(outputs, labels)

                # Backward and optimize
                opt.zero_grad()
                loss.backward()
                opt.step()
            print("Flow epoch: " + str(epoch) + " model " + str(passnum))

    def train(self, passnum):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        self.results_file.write(current_time + "\n")

        if self.Sig:
            self.model_signal = SignalConvNet().to(self.device)
            criterion_sig = nn.CrossEntropyLoss()
            optimizer_sig = torch.optim.SGD(self.model_signal.parameters(), lr=self.learning_rate)

            self.train_one_model(self.training_loader_signals, criterion_sig, optimizer_sig, self.model_signal, passnum)
            res_string = 'Finished training Signal'
            print(res_string)
            self.results_file.write(res_string + "\n")

            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
            self.results_file.write(current_time + "\n")
            torch.save(self.model_signal, self.PATH_SIG)
        else:
            self.model_signal = torch.load(self.PATH_SIG)
            self.model_signal.eval()

        if self.Flow:
            self.model_OF = ResNet101(7, 2).to(self.device)
            criterion_of = nn.CrossEntropyLoss()
            optimizer_of = torch.optim.SGD(self.model_OF.parameters(), lr=self.learning_rate)
            self.train_one_model(self.training_loader_flow, criterion_of, optimizer_of, self.model_OF, passnum)
            res_string = 'Finished training flow'
            print(res_string)
            self.results_file.write(res_string + "\n")

            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
            self.results_file.write(current_time + "\n")
            torch.save(self.model_OF, self.PATH_OF)
        else:
            self.model_OF = torch.load(self.PATH_OF)
            self.model_OF.eval()

    def append_results(self, results_mat, file):
        sum = 0.0
        count = 0.0
        str_tIoU = ''
        for row in results_mat:
            count = 0.0
            sum = 0.0
            for col in row:
                count += 1.0
                sum += col
            ave = sum/count
            str_tIoU = str_tIoU + str(ave) + ", "
        str_tIoU = str_tIoU + '\n'
        file.write(str_tIoU)
        file.flush()


    def run(self):
        results_sg = open("./results_sg.csv", 'w')
        results_of = open("./results_of.csv", 'w')
        results_ts = open("./results_ts.csv", 'w')
        for index in range(0, 30):
            self.results_file = open("./model_performance.csv", 'w')
            if self.do_train:
                self.train(index)
            else:
                self.model_signal = torch.load(self.PATH_SIG)
                self.model_signal.eval()
                self.model_OF = torch.load(self.PATH_OF)
                self.model_OF.eval()

            self.results_file.write('*********************** Signal Results ***********************\n')
            files = {}
            if self.Sig:
                self.test_model(files, self.model_signal, self.testing_loader_signals, 'Signals ')
                self.write_files_to_file(files, 'signal_only.txt')
                rev_results = self.smooth_results(files)
                self.append_results(self.calcmAP(rev_results), results_sg)

            else:
                self.load_files_from_file(files, 'signal_only.txt')

            self.results_file.write('*********************** Optical Flow Results ***********************\n')
            files_of = {}
            if self.Flow:
                self.test_model(files_of, self.model_OF, self.testing_loader_flow, 'Optical Flow ')
                self.write_files_to_file(files_of, 'of_only.txt')
                rev_results2 = self.smooth_results(files_of)
                self.append_results(self.calcmAP(rev_results2), results_of)
            else:
                self.load_files_from_file(files_of, 'of_only.txt')

            self.results_file.write('*********************** Combined Results ***********************\n')
            files_merge = self.merge_rev_results(files, files_of)
            rev_results_merge = self.smooth_results(files_merge)

            self.append_results(self.calcmAP(rev_results_merge), results_ts)

            self.results_file.close()
        results_sg.close()
        results_of.close()
        results_ts.close()

    def merge_rev_results(self, files_sig, files_of):
        files_merged = {}
        for file in files_of.keys():
            sig_file = files_sig[file][1]
            of_file = files_of[file][1]
            files_merged[file] = [files_of[file][0], []]
            merged_list = files_merged[file][1]
            for frame in of_file:
                l = list(filter(lambda x: x[0] == frame[0], sig_file))
                if len(l) == 1:
                    sig_frame = l[0]
                    sig_res = sig_frame[2]
                    of_res = frame[2]
                    max_of = max(of_res)
                    max_sig = max(sig_res)
                    if max_sig < 0.5 and max_of > 0.75 and frame[3] != sig_frame[3]:
                        new_res = [0.0, 0.0, 0.0,0.0, 0.0, 0.0,0.0]
                        for i in range(0, 7):
                            new_res[i] = of_res[i]
                        merged_list.append([frame[0], frame[1], new_res, np.array(new_res).argmax()])
                        continue
                    elif max_sig > 0.75 and max_of < 0.5 and frame[3] != sig_frame[3]:
                        new_res = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        for i in range(0, 7):
                            new_res[i] = sig_res[i]
                        merged_list.append([frame[0], frame[1], new_res, np.array(new_res).argmax()])
                        continue
                    else:
                        new_res = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        for i in range(0, 7):
                            new_res[i] = (of_res[i] + sig_res[i]) / 2
                        merged_list.append([frame[0], frame[1], new_res, np.array(new_res).argmax()])
                        continue

                merged_list.append(deepcopy(frame))
        return files_merged

    def load_files_from_file(self, files, file_name):
        with open(file_name, 'r') as fl:
            current_list = None
            for line in fl:
                vals = line.split(sep=',')
                if len(vals) == 2:
                    files[vals[0]] = [int(vals[1]), []]
                    current_list = files[vals[0]][1]
                else:
                    res = vals[2:9]
                    res_float = [float(string) for string in res]

                    current_list.append([int(vals[0]), int(vals[1]), res_float, int(vals[9].strip())])

    def missingValues(self, prev, curr, count, next, valslist):
        for i in range(count, next):
            valslist.append([self.MISSING, None, None, False])
        return next - count


    def smooth(self, valslist):
        # 8, 8, 8, 8, 0, 0, 1, 0

        curr_index = 0
        while(curr_index < len(valslist)-1):
            if valslist[curr_index][0] == self.MISSING:  #First value, always missing
                curr_index += 1
                continue
            #don't smmoth away action of interest -- might be short
            #if valslist[curr_index][0] > 1 and valslist[curr_index][0]< 6:
            #    continue
            prev = valslist[curr_index -1][0]
            next = valslist[curr_index +1][0]
            if valslist[curr_index][0] == prev or valslist[curr_index][0] == next:
                curr_index += 1
                continue
            if prev == self.MISSING or next == self.MISSING:
                valslist[curr_index][0] = self.MISSING
            else:
                valslist[curr_index][0] = next
            curr_index += 1

        curr_index = 0
        prev = self.MISSING
        while(curr_index < len(valslist)):
            if valslist[curr_index][0] == self.MISSING:
                if prev == self.MISSING:
                    curr_index_plus = curr_index
                    while valslist[curr_index_plus][0] == self.MISSING:
                        curr_index_plus += 1
                    prev = valslist[curr_index_plus][0]
                    for i in range(curr_index, curr_index_plus):
                        valslist[i][0] = prev
                else:
                    valslist[curr_index][0] = prev
            else:
                prev = valslist[curr_index][0]
            curr_index += 1


    def test_model(self, files, model, test_loader, model_name):

        results_mat = np.zeros((7,7), dtype=int)

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for i in range(len(self.classes))]
            n_class_samples = [0 for i in range(len(self.classes))]
            for images, labels, positioning_info in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                sm_outputs = F.softmax(outputs, dim=1)
                cpu_sm_outputs = sm_outputs.cpu().detach().numpy()
                _, predicted = torch.max(outputs, 1)

                for k in range (0, len(positioning_info[0])):
                    if not positioning_info[0][k] in files.keys():
                        files[positioning_info[0][k]] = [positioning_info[2][k].item(), []]
                    files[positioning_info[0][k]][1].append([positioning_info[1][k].item(), labels[k].item(), cpu_sm_outputs[k], predicted[k].item()])

                    # max returns (value ,index)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()
                    #print(outputs[k].cpu().detach().numpy())

                    for i in range(len(labels)):
                        label = labels[i]
                        pred = predicted[i]
                        if (label == pred):
                            n_class_correct[label] += 1
                        #else:
                            #print("Actual: " + classes[label] + " Predicted: " + classes[pred] + " Percentage Actual: " + str(outputs[i][label].item()) + " Percentage Predicted: " + str(outputs[i][pred].item()))
                        results_mat[pred][label] += 1
                        n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            acc_txt = model_name + f'Unsmoothed Accuracy of the network: {acc} %'
            print(acc_txt)
            self.results_file.write(acc_txt + "\n")

            for i in range(len(self.classes)):
                acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                acc_txt =  model_name + f'Unsmoothed Accuracy of {self.classes[i]}: {acc} %'
                print(acc_txt)
                self.results_file.write(self.classes[i] + "," + str(acc) + "\n")
        conf_title =  model_name +"Confusion Matrix unsmoothed:"
        print(conf_title)
        print(results_mat)
        self.write_mat(results_mat, conf_title + "\n\n")

    def write_mat(self, results_mat, conf_title):
        self.results_file.write(conf_title)
        for row in results_mat:
            for col in row:
                self.results_file.write("," + str(col))
            self.results_file.write("\n")
        self.results_file.write("\n\n")

    def write_files_to_file(self, files, file_name):
        with open(file_name, 'w') as fl:
            for file in files.keys():
                f = files[file]
                fl.write(f'{file}, {f[0]}\n')
                for r in f[1]:
                    fl.write(f'{r[0]}, {r[1]}, ')
                    for e in r[2]:
                        fl.write(f'{e}, ')
                    fl.write(f'{r[3]}\n')


    def smooth_results(self, files):
        rev_results = {}
        for file in files.keys():
            f = files[file]
            valslist = []
            results = sorted(f[1], key=lambda x: x[0])
            c = 0
            prev = STAND
            for r in results:
                if r[0] > c:
                    c += self.missingValues(prev, r[3], c, r[0], valslist)
                valslist.append([r[3], r[2], r[1], True])
                c += 1
            if f[0] > c+1:
                self.missingValues(prev, STAND, c, f[0], valslist)
            self.smooth(valslist)
            rev_results[file] = valslist
            #Walk through files
            #Create a new list
                #fill in missing values
                #Smooth consecutive values
                #revise predictions
            #report new frame level accuracy


        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(len(self.classes))]
        n_class_samples = [0 for i in range(len(self.classes))]
        results_mat_smooth = np.zeros((7, 7), dtype=int)

        for f in rev_results.values():
            for j in f:
                if j[3]:
                    lab = j[2]
                    predicted = j[0]
                    n_samples += 1
                    n_correct += 1 if predicted == lab else 0
                    outputs = j[1]
                    if (lab == predicted):
                        n_class_correct[lab] += 1
                    #else:
                    #    print("Actual: " + classes[lab] + " Predicted: " + classes[predicted] + " Percentage Actual: " + str(outputs[lab]) + " Percentage Predicted: " + str(outputs[predicted]))
                    results_mat_smooth[predicted][lab] += 1
                    n_class_samples[lab] += 1

        acc = 100.0 * n_correct / n_samples
        acc_txt = f'Smoothed Accuracy of the network: {acc} %'
        print(acc_txt)
        self.results_file.write(acc_txt + "\n")

        for i in range(len(self.classes)):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            acc_txt = f'Smoothed Accuracy of {self.classes[i]}: {acc} %'
            print(acc_txt)
            self.results_file.write(self.classes[i] + "," + str(acc) + "\n")
        conf_title = "Confusion Matrix Smoothed:"
        print(results_mat_smooth)
        self.write_mat(results_mat_smooth, conf_title + "\n\n")
        return rev_results

        #only for shake_five2
    def get_tIOU(self, start_index, end_index, curr, sample_vid):
        ann = sample_vid.ann
        ann_range = sample_vid.ann_ranges
        if curr == STAND:
            if ann_range[0]-1 <= start_index or end_index <= 1:
                if ann_range[7] <= start_index or end_index <= ann_range[6]:
                    return 0.0
                else:
                    index = 6
            else:
                union = float(max(end_index, ann_range[0]-1) - min(start_index, 1))
                intersection = float(min(end_index, ann_range[0]-1) - max(start_index, 1))
                return intersection / union
        elif curr == APPROACH:
            index = 0
        elif curr > 1 and curr < 6 and ann == self.classes[curr]:
            index = 2
        elif curr == 6:
            index = 4
        else:
            return 0.0

        if ann_range[index+1] <= start_index or end_index <= ann_range[index]:
            return 0.0

        union = float(max(end_index, ann_range[index+1])- min(start_index, ann_range[index]))
        intersection = float(min(end_index, ann_range[index+1])- max(start_index, ann_range[index]))

        return intersection/union

    def getConf(self, valslist, start_index, curr_index, curr):
        running_total = 0.0
        count = 0
        for i in range(start_index, curr_index + 1):
            if valslist[i][3]:
                running_total += valslist[i][1][curr]
                count += 1
        return running_total/count

    def addGTCount(self, ranges, gt, curr):
        gt[0:2] += 1
        gt[6] += 1
        gt[curr] += 1


    def calcAP(self, points):
        patr = np.zeros(11, dtype=float)
        m = 0
        patr[0] = 1
        for i in range(1, 11):
            recallnext = (i+1)/10.0
            recall = i/10.0
            res = list(filter(lambda x: x[1] < recallnext and x[1] >= recall, points))
            for r in res:
                patr[i] = max(patr[i], r[0])
        return np.sum(patr)/11.0


    def calcmAP(self, rev_results):
     #mAP
        gt = np.zeros(7)

        predicted_range = []
        for c in range(0, len(self.classes)):
            predicted_range.append([])
        for f in rev_results.keys():
            results = rev_results[f]
            vid = list(filter(lambda x: x.file_name.strip() == f.strip(), self.testing_vids))[0]
            curr = results[0][0]
            start_index = 0
            curr_index = 0
            for p in results:
                if p[0] != curr:
                    predicted_range[curr].append([start_index, curr_index, self.get_tIOU(start_index, curr_index, curr, vid), self.getConf(results, start_index, curr_index, curr)])
                    start_index = curr_index+1
                    curr = p[0]
                curr_index += 1
            predicted_range[curr].append([start_index, curr_index-1, self.get_tIOU(start_index, curr_index-1, curr, vid), self.getConf(results, start_index, curr_index-1, curr)])
            self.addGTCount(vid.ann_ranges, gt, SampleVid.getActionOfInterest(vid.ann))


        testtIoU = 0.9
        results = []
        mat = np.zeros((10,8), dtype=float)
        count = 0
        while testtIoU > 0:
            points = []
            aps = 0.0
            curr = 0
            mat[count] = testtIoU
            for c in range(0, len(self.classes)):
                r = predicted_range[c]
                sorted_range = sorted(r, key=lambda x: x[3], reverse=True)
                cum_TP = 0
                cum_FP = 0
                points.clear()
                for sr in sorted_range:
                    if sr[2] > testtIoU:
                        cum_TP += 1
                    else:
                        cum_FP += 1
                    P = cum_TP /(cum_TP + cum_FP)
                    R = cum_TP / gt[curr]
                    points.append([P, R])
                ap = self.calcAP(points)
                mat[count][c+1] = ap
                aps += ap
                curr += 1
            mAP = aps/len(self.classes)

            print("At tIoU:" + str(testtIoU) + " mAP: " + str(mAP))
            results.append([testtIoU, map])
            testtIoU -= 0.1
            count += 1
        self.write_mat(mat, ',tIoU,Stand,Approach,Handshake,Hug,High Five,Fist Bump, Leave\n')
        return mat

def main():
    mt = ModelTraining()
    mt.run()

if __name__ == '__main__':
    main()
