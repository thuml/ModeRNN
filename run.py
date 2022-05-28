__author__ = 'zhiyu'


import os
import shutil
import argparse
import numpy as np
import torch
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.models.model_adaptive import Model as Adaptive_model
from core.utils import preprocess
import core.trainer as trainer

from tensorboardX import SummaryWriter
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - PredRNN')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cpu:0')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--dataset_name1', type=str, default='mnist_pp')
parser.add_argument('--train_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--train_data_paths1', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths1', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--train_data_paths2', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths2', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--train_data_paths3', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths3', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn')
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_height', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='predrnn')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--pretrained_model1', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

parser.add_argument('--img_ch', type=int, default=1)
parser.add_argument('--patch_consec', type=int, default=0, help='0: down-sample; 1: consecutive within patches')
parser.add_argument('--patch_num', type=int, default=4)
parser.add_argument('--patch_num_old', type=int, default=4)
parser.add_argument('--traj_num', type=int, default=3)
parser.add_argument('--worker', type=int, default=1)


# action-based predrnn
parser.add_argument('--injection_action', type=str, default='concat')
parser.add_argument('--conv_on_input', type=int, default=1, help='conv on input')
parser.add_argument('--res_on_conv', type=int, default=0, help='res on conv')
parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')



args = parser.parse_args()
print(args)
global_step = 0

if torch.cuda.is_available():
    #device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True



def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (args.batch_size,
                            args.total_length - args.input_length - 1,
                            args.img_width // args.patch_size,
                            args.img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag






def train_wrapper(model):
    global global_step
    if args.pretrained_model:
        model.load(args.pretrained_model)

        print("load source successfully")

    if args.pretrained_model1:
        model.load_RNN(args.pretrained_model1)

        print("load source successfully")

    summary_dir = "summary/{}".format(args.save_dir)
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)
    writer = SummaryWriter(log_dir=summary_dir)

    # load data
    train_input_handle1, test_input_handle1 = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths1, args.valid_data_paths1, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=True, injection_action=args.injection_action)

    test_input_handle_adapt = test_input_handle1


    eta = args.sampling_start_value
    fast_weights = None

    for itr in range(1, args.max_iterations + 1):
        if train_input_handle1.no_batch_left():
            train_input_handle1.begin(do_shuffle=True)

        ims_get1 = train_input_handle1.get_batch()
        ims1 = preprocess.reshape_patch(ims_get1, args.patch_size)


        eta, real_input_flag = schedule_sampling(eta, itr)

        trainer.train(model, ims1, real_input_flag, args, itr)

        if itr % args.snapshot_interval == 0:
            model.save(itr)

        if itr % args.test_interval == 0:
            #j = 0
            if "radar" in args.dataset_name:
                trainer.test_single_radar(model, test_input_handle1, args, itr, writer)
            else:
                trainer.test(model, test_input_handle1, args, itr, writer)

        train_input_handle1.next()



def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, is_training=False)
    trainer.test(model, test_input_handle, args, 'test_result')

if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

#gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
#args.n_gpu = len(gpu_list)
print('Initializing models')


if args.is_training == 1:
    model = Model(args, parser)
    train_wrapper(model)
else:
    model = Model_i3d(args)
    test_wrapper(model)

