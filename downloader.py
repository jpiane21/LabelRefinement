import os
import gdown
import zipfile


PATH_download = "./download"
os.makedirs(PATH_download)
print ('make directory download')


root_folder = "./"

url_mmflow = 'https://drive.google.com/uc?id=1NsjTeNLvcoFU85If6JRykWVxRktwWVlo'
url_shakefive = 'https://drive.google.com/uc?id=1WK_BuhCfS0oL4I-7nowpD3Yyd2JVcnSj'
url_ModelSetup = 'https://drive.google.com/uc?id=1mCEQb8Vli10RpCfQVnk7-vq4st4LCgxX'

print ('Downloading files...')
print ('Downloading mmflow')
gdown.download(url_mmflow,'./download/mmflow.zip',quiet=False)
print ('Downloading shakefive')
gdown.download(url_shakefive,'./download/shakefive.zip',quiet=False)
print ('Downloading ModelSetup')
gdown.download(url_ModelSetup,'./download/ModelSetup.zip',quiet=False)

print ('Unzipping mmflow')
with zipfile.ZipFile(PATH_download + '/mmflow.zip', 'r') as ziphandler:
    ziphandler.extractall(root_folder)
print ('Unzipping shakefive')
with zipfile.ZipFile(PATH_download + '/shakefive.zip', 'r') as ziphandler:
    ziphandler.extractall(root_folder)
print ('Unzipping ModelSetup')
with zipfile.ZipFile(PATH_download + '/ModelSetup.zip', 'r') as ziphandler:
    ziphandler.extractall(root_folder)