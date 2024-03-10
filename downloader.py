import os
import gdown
import shutil

mmdet_name = 'faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth'
mmflow_name = 'flownet2css-sd_8x1_sfine_flyingthings3d_subset_chairssdhom_384x448.pth'
mmtrack_name = 'faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'
mmpose_name = 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'

PATH_download = "./download"
print ('Make directory download')
os.makedirs(PATH_download)

url_mmdet = 'https://download.openmmlab.com/mmdetection/v2.0/objects365/faster_rcnn_r50_fpn_16x4_1x_obj365v2/' + mmdet_name
url_mmflow = 'https://download.openmmlab.com/mmflow/flownet2/' + mmflow_name
url_mmtrack = 'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/' + mmtrack_name
url_mmpose = 'https://download.openmmlab.com/mmpose/top_down/hrnet/' + mmpose_name


print ('Downloading files...')
gdown.download(url_mmflow, './ModelSetup/mmflow/checkpoints/' + mmflow_name, quiet=False)
gdown.download(url_mmdet, './ModelSetup/mmdetect/checkpoints/' + mmdet_name, quiet=False)
gdown.download(url_mmtrack, './ModelSetup/mmtrack/checkpoints/' + mmtrack_name, quiet=False)
gdown.download(url_mmpose, './ModelSetup/mmpose/checkpoints/' + mmpose_name, quiet=False)

