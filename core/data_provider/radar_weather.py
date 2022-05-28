__author__ = 'gaozhifeng'

import numpy as np
import logging
import random

import time
import os
from PIL import Image
import logging
import cv2

from core.data_provider import flip_rotate
import random

logger = logging.getLogger(__name__)


class InputHandle:
    def __init__(self, datas, indices, seq_name, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')

        self.img_width = input_param['image_width']

        self.datas = datas
        self.indices = indices
        self.seq_name = seq_name
        self.current_input_length = input_param['seq_length']
        self.minibatch_size = input_param['minibatch_size']

        self.current_position = 0
        self.current_batch_indices = []

        self.time_revolution = 1

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.img_width, self.img_width, 1)).astype(
            self.input_data_type)
        seq_name_batch = []
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length

            names = []
            for j in range(self.current_input_length):
                input_batch[i, j, :, :, :] = self.datas[begin + j * self.time_revolution, :, :, :]
                names.append(self.seq_name[begin + j * self.time_revolution])
            seq_name_batch.append(names)

            # data_slice = self.datas[begin:end, :, :, :]
            # input_batch[i, :self.current_input_length, :, :, :] = data_slice
            # seq_name_batch.append(self.seq_name[begin:end])
        input_batch = input_batch.astype(self.input_data_type) / 255.0
        # todo
        return input_batch, np.asarray(seq_name_batch)

    def get_batch(self):
        input_batch, seq_name_batch = self.batch()

        if random.random() < 0.4:
            input_batch = input_batch[:, ::-1]
        if random.random() < 0.4:
            input_batch = flip_rotate.augment_data(input_batch)

        return input_batch

    # return [input_batch], seq_name_batch

    def get_eval_batch(self):
        input_batch, seq_name_batch = self.batch()
        return input_batch, seq_name_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    Input Data Type: " + str(self.input_data_type))


class DataProcess:
    def __init__(self, input_param):
        self.valid_paths = input_param['valid_data_paths']
        self.train_paths = input_param['train_data_paths']
        self.img_width = input_param['image_width']

        self.input_param = input_param
        # self.time_revolution = input_param['time_revolution']
        self.time_revolution = 1
        self.seq_len = input_param['seq_length'] * self.time_revolution
        # self.small_data = input_param['small_data']
        self.small_data = False
        self.data_diff = 6 * 60

    def load_data(self, path, img_width, train=True):
        print('begin load data' + str(path))
        logger.info('begin load data' + str(path))
        # get all path
        dirlist = os.listdir(path)
        dirlist.sort()
        frames_fname = []
        frames_np = []
        count = 0
        for directory in dirlist:
            dir_path = os.path.join(path, directory)
            filelist = os.listdir(dir_path)
            filelist.sort()
            #if "2017" in directory and 'train' in path:
            #    continue
            # if len(frames_fname) > 1000:
            #     break
            if self.small_data and len(frames_fname) > 1000 and 'test' in path:
                break
            for file in filelist:
                frames_fname.append(file)
                frame_im = Image.open(os.path.join(dir_path, file))
                frame_np = np.array(frame_im)  # numpy array
                # frame_np = frame_np[100:500,100:500]
                #               print(count)
                frames_np.append(frame_np)
                # print(file)
        # is it a begin index of sequence
        indices = []
        for i in range(len(frames_fname) - (self.seq_len - 1)):
            try:
                fname_begin = os.path.splitext(frames_fname[i])[0].split('_')[1]
                fname_end = os.path.splitext(frames_fname[i + self.seq_len - 1])[0].split('_')[1]
                fdate_begin = time.mktime(time.strptime(fname_begin, '%Y%m%d%H%M%S'))
                fdate_end = time.mktime(time.strptime(fname_end, '%Y%m%d%H%M%S'))
                if int(fdate_end - fdate_begin) <= int(self.data_diff * (self.seq_len - 1) + 5 * 60):
                    indices.append(i)
            except:
                print(fname_begin)
                print(fname_end)

            # else:
            #    print(fname_begin)
            #    print(fname_end)
            #    print(fdate_end - fdate_begin)

        frames_np = np.asarray(frames_np)
        frames_np = np.asarray(frames_np)
        data = np.zeros((frames_np.shape[0], img_width, img_width, 1))
        for i in range(len(frames_np)):
            temp = np.uint8(frames_np[i, :, :])
            # self.train_data[i, :, :, 0] = temp
            data[i, :, :, 0] = cv2.resize(temp, (img_width, img_width))
        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")

        # for i in xrange(len(indices)):
        #    print("" + str(i) + ": " + str(frames_fname[indices[i]]) + "    " + str(indices[i]))
        return data, indices, frames_fname

    def get_train_input_handle(self):
        train_data, train_indices, seq_names = self.load_data(self.train_paths, self.img_width, True)
        return InputHandle(train_data, train_indices, seq_names, self.input_param)

    def get_test_input_handle(self, full=False):
        params = self.input_param.copy()
        img_width = self.img_width
        if full:
            params['image_width'] = 384
            img_width = 384
        test_data, test_indices, seq_names = self.load_data(self.valid_paths, img_width, False)
        return InputHandle(test_data, test_indices, seq_names, params)
