# This file is used to transform the video into frames and save as ILSVRC2015 VID format
# it contains 80 classes

import os
import imageio
import numpy as np

if not os.path.exists('VIDOR80/Data/train'):
    os.makedirs('VIDOR80/Data/train')
if not os.path.exists('VIDOR80/Data/val'):
    os.makedirs('VIDOR80/Data/val')
if not os.path.exists('VIDOR80/Data/test'):
    os.makedirs('VIDOR80/Data/test')


def transform(mode):
    img_num = 0
    video_num = 0
    if mode == 'train':
        root_dir = 'annotation/training/'
    else:
        root_dir = 'annotation/validation/'
    for preids in os.listdir(root_dir):
        print(preids)

        pre_path = root_dir + preids
        pre_imgs_path = 'VIDOR80/Data/'+mode+'/' + preids
        if not os.path.exists(pre_imgs_path):
            os.makedirs(pre_imgs_path)
        for json_name in os.listdir(pre_path):
            video_num += 1
            if video_num % 100 == 0:
                print(video_num)
            video_id = json_name[:-5]
            video_name = video_id + '.mp4'

            imgs_path = pre_imgs_path + '/' + video_id
            if not os.path.exists(imgs_path):
                os.makedirs(imgs_path)

            # save each frames of the video
            video_path = 'video/' + preids + '/' + video_name
            video = imageio.get_reader(video_path, 'ffmpeg')

            for num, img in enumerate(video):
                img_path = os.path.join(imgs_path, str(num) + '.JPEG')
                imageio.imwrite(img_path, img)
            #img_num += save_img
            #print(video_name + ': ' + str(num) + '       ' + str(img_num / video_num))
            video.close()


def transform_test(mode):
    img_num = 0
    video_num = 0
    root_dir = 'video/test/video/'
    for subset in os.listdir(root_dir):
        print(subset)
        subpath = root_dir + subset
        sub_imgs_path = 'VIDOR80/Data/'+mode+'/' + subset
        if not os.path.exists(sub_imgs_path):
            os.makedirs(sub_imgs_path)
        for vname in os.listdir(subpath):
            video_num += 1
            print(video_num, vname)
            video_id = vname[:-4]

            imgs_path = sub_imgs_path + '/' + video_id
            if not os.path.exists(imgs_path):
                os.makedirs(imgs_path)

            # save each frames of the video
            video_path = os.path.join(subpath, vname)
            video = imageio.get_reader(video_path, 'ffmpeg')

            for num, img in enumerate(video):
                img_path = os.path.join(imgs_path, str(num) + '.JPEG')
                imageio.imwrite(img_path, img)
            #img_num += save_img
            # print(video_name + ': ' + str(num) + '       ' + str(img_num / video_num))
            video.close()


if __name__ == '__main__':
    #transform('train')
    #transform('val')
    transform_test('test')

