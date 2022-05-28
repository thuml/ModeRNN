import numpy as np
import random


class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.img_width = input_param['image_width']
        self.img_height = input_param['image_height']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.length = 0
        self.height = 0
        self.width = 0
        self.load()

    def load(self):
        self.data = np.load(self.paths[0])
        self.length = self.data.shape[1]
        self.height = self.data.shape[2]
        self.width = self.data.shape[3]
        print(self.data.shape)

    def total(self):
        return self.data.shape[0]

    def begin(self, do_shuffle = True):
        self.indices = np.arange(self.total(), dtype="int32")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]

    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]

    def no_batch_left(self):
        if self.current_position >= self.total() - self.current_batch_size:
            return True
        else:
            return False

    def get_batch(self):
        batch = np.zeros(
            (self.current_batch_size, self.length, self.img_height, self.img_width, 3))
        for i in range(self.current_batch_size):
            batch_ind = self.current_batch_indices[i]
            if self.img_height == self.height and self.img_width == self.width:
                # batch[i, :, :, :, 0] = self.data[batch_ind, :, :, :]
                batch[i, :, :, :, 0] = self.data[batch_ind, :, :, :, 0]
                batch[i, :, :, :, 1] = self.data[batch_ind, :, :, :, 1]
                batch[i, :, :, :, 2] = self.data[batch_ind, :, :, :, 2]
            else:
                batch[i, :, :, :, 0] = cv2.resize(self.data[batch_ind], (self.img_height, self.img_width))
                batch[i, :, :, :, 1] = cv2.resize(self.data[batch_ind], (self.img_height, self.img_width))
                batch[i, :, :, :, 2] = cv2.resize(self.data[batch_ind], (self.img_height, self.img_width))

        batch = batch.astype(self.input_data_type) / 255.0
        batch = np.minimum(batch, 1)
        batch = np.maximum(batch, 0)
        return batch
