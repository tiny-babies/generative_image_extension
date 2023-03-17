import cv2
import numpy as np
import os

folder_path = './dataset'
target_path = './training_data'

dirs = os.listdir(folder_path)
dirs_name_list = []


for dir_item in dirs:
        # modify to full path -> directory
        dir_item = folder_path + "/" + dir_item
        print(dir_item)

        training_folder = os.listdir(dir_item)
        index = 0
        for training_item in training_folder:
            index += 1
            fileName = dir_item + '/' + training_item
            targetName = (dir_item + "/" + training_item).replace(folder_path, target_path)

            # print(fileName)
            
            img = cv2.imread(fileName)
            width = img.shape[1]
            height = img.shape[0]
            flippedimg = cv2.flip(img, 1)

            finalImg = cv2.hconcat([img, flippedimg])
            croppedImg = finalImg[:, width - int(width / 2):width + int(width/2)]
            cv2.imwrite(targetName, croppedImg)

            if(index == 1000): break


# import shutil

# folder_path = './dataset'
# target_path = './sub_dataset'

# dirs = os.listdir(folder_path)
# dirs_name_list = []


# for dir_item in dirs:
#     # modify to full path -> directory
#     dir_item = folder_path + "/" + dir_item
#     print(dir_item)

#     training_folder = os.listdir(dir_item)
#     os.mkdir(dir_item.replace(folder_path, target_path))
#     index = 0
#     for training_item in training_folder:
#         index += 1
#         fileName = dir_item + '/' + training_item
#         targetName = (dir_item + "/" +
#                       training_item).replace(folder_path, target_path)
        
#         shutil.copyfile(fileName, targetName)

#         # # print(fileName)

#         # img = cv2.imread(fileName)
#         # width = img.shape[1]
#         # height = img.shape[0]
#         # flippedimg = cv2.flip(img, 1)

#         # finalImg = cv2.hconcat([img, flippedimg])
#         # croppedImg = finalImg[:, width -
#         #                       int(width / 2):width + int(width/2)]
#         # cv2.imwrite(targetName, croppedImg)

#         if(index == 1000):
#             break
