import os.path
import datetime
import cv2
import numpy as np
from skimage.measure import compare_ssim
from core.utils import preprocess
from core.utils import metrics
from tensorboardX import SummaryWriter
from PIL import Image

def cal_decay(t):
    return 1.0
    # if t <= 10:
    #     return 1.0
    # if t <= 30 and t >= 10:
    #     return 0.5
    # if t >= 30 and t <= 50:
    #     return 1.0
    # if t >= 50 and t <= 70:
    #     return 0.5
    # if t >= 70 and t <= 90:
    #     return 1.0
    # if t >= 90:
    #     return 0.5
    # return (math.cos((3.1415926 / 10) * t)) / 4 + 0.75
    # def cal_decay(t):
    #     a = 0.25 + 0.5 * t / 100.0
    #     b = (math.sin((3.1415926 / 2.5) * t)) / 4
    #     return a + b
    # return (math.cos((3.1415926 / 10) * t)) / 4 + 0.75
    # if t % 20 == 10:
    #     return 1.0
    # else:

def add_palette(array2d, size):
    palette = [0, 0, 0, 0, 237, 237, 0, 237, 237, 0, 237, 237, 0, 237, 237, 0, 237, 237, 0, 237, 237, 0, 237, 237, 0,
               237, 237, 0, 237, 237, 0, 217, 0, 0, 217, 0, 0, 217, 0, 0, 217, 0, 0, 217, 0, 0, 217, 0, 0, 217, 0, 0,
               217, 0, 0, 217, 0, 0, 217, 0, 0, 145, 0, 0, 145, 0, 0, 145, 0, 0, 145, 0, 0, 145, 0, 0, 145, 0, 0, 145,
               0, 0, 145, 0, 0, 145, 0, 0, 145, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255,
               255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 0, 231, 193, 0, 231, 193, 0, 231, 193, 0, 231,
               193, 0, 231, 193, 0, 231, 193, 0, 231, 193, 0, 231, 193, 0, 231, 193, 0, 231, 193, 0, 255, 145, 0, 255,
               145, 0, 255, 145, 0, 255, 145, 0, 255, 145, 0, 255, 145, 0, 255, 145, 0, 255, 145, 0, 255, 145, 0, 255,
               145, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0,
               0, 255, 0, 0, 200, 0, 0, 200, 0, 0, 200, 0, 0, 200, 0, 0, 200, 0, 0, 200, 0, 0, 200, 0, 0, 200, 0, 0,
               200, 0, 0, 200, 0, 0, 170, 0, 0, 170, 0, 0, 170, 0, 0, 170, 0, 0, 170, 0, 0, 170, 0, 0, 170, 0, 0, 170,
               0, 0, 170, 0, 0, 170, 0, 0, 255, 0, 241, 255, 0, 241, 255, 0, 241, 255, 0, 241, 255, 0, 241, 255, 0, 241,
               255, 0, 241, 255, 0, 241, 255, 0, 241, 255, 0, 241, 151, 0, 181, 151, 0, 181, 151, 0, 181, 151, 0, 181,
               151, 0, 181, 151, 0, 181, 151, 0, 181, 151, 0, 181, 151, 0, 181, 151, 0, 181, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245,
               141, 173, 245, 141, 173, 245, 141, 173, 245, 141, 173, 245, 141]

    array2d= array2d.reshape(size, size)
    new_im = Image.fromarray(np.array(array2d))
    new_im.putpalette(palette)
    return new_im


def func(Y_true, Y_pred, shape):
    t1 = np.abs(Y_true[1:, :-1] - Y_true[:-1, :-1])
    t2 = np.abs(Y_true[:-1, 1:] - Y_true[:-1, :-1])
    t3 = np.abs(Y_pred[1:, :-1] - Y_pred[:-1, :-1])
    t4 = np.abs(Y_pred[:-1, 1:] - Y_pred[:-1, :-1])
    N =  (shape[-2] - 1) * (shape[-1] - 1)
    out = np.sum(np.abs((t1+t2)-(t3+t4))) / N
    out = 10*np.log10(255*255/out)
    return out




def train(model, ims, real_input_flag, configs, itr):
    cost = model.train(ims, real_input_flag, itr)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost_r = model.train(ims_rev, real_input_flag, itr)
        cost += cost_r
        cost = cost / 2

    if itr % configs.display_interval == 0:
         print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'itr: '+str(itr))
         print('mse training loss: ' + str(cost))


def test_adapt(model, ims, real_input_flag, configs, itr):
    cost = model.test_adapt(ims, real_input_flag, itr)

    if itr % configs.display_interval == 0:
         print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'itr: '+str(itr))
         print('mse test_adapt loss: ' + str(cost))

def train_multitask(model, ims1, real_input_flag, configs, itr, memory_local, memory_global, writer, w, task_id):
    cost, memory_local, memory_global = model.train_multitask(ims1, real_input_flag, memory_local, memory_global, itr, w, task_id)
    if configs.reverse_input:
        ims_rev1 = np.flip(ims1, axis=1).copy()
        cost_r, memory_local, memory_global = model.train_multitask(ims_rev1, real_input_flag, memory_local, memory_global, itr, w, task_id)
        cost += cost_r
        cost = cost / 2

    if itr % configs.display_interval == 0:
         print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'itr: '+str(itr))
         print('mse training loss: ' + str(cost))
         #writer.add_scalar("train w1: ", w1, itr)
         #writer.add_scalar("train w2: ", w2, itr)
    return memory_local, memory_global, cost

def train_inner_update(model, ims, real_input_flag, configs, itr):
    model.inner_update(ims, real_input_flag, itr)




def test_multitask(model, test_input_handle1, configs, itr, writer, mark):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    test_input_handle1.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir + str(mark), str(itr))
    if not os.path.exists(configs.gen_frm_dir + str(mark)):
        os.mkdir(configs.gen_frm_dir + str(mark))
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    avg_mse1 = 0
    avg_mse2 = 0
    batch_id = 0
    img_mse1, ssim1, psnr1 = [], [], []


    for i in range(configs.total_length - configs.input_length):
        img_mse1.append(0)
        ssim1.append(0)
        psnr1.append(0)


    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    while (test_input_handle1.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims1 = test_input_handle1.get_batch()
        test_dat1 = preprocess.reshape_patch(test_ims1, configs.patch_size)

        img_gen1 = model.test(test_dat1, real_input_flag)

        img_gen1 = preprocess.reshape_patch_back(img_gen1, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_gen_length = img_gen1.shape[1]
        img_out1 = img_gen1[:, -output_length:]
        #print("img_gen_length", img_gen_length, img_gen.shape)

        # MSE per frame
        for i in range(output_length):
            x1 = test_ims1[:, i + configs.input_length, :, :, :]
            gx1 = img_out1[:, i, :, :, :]
            gx1 = np.maximum(gx1, 0)
            gx1 = np.minimum(gx1, 1)
            mse1 = np.square(x1 - gx1).sum()
            img_mse1[i] += mse1
            avg_mse1 += mse1



            real_frm1 = np.uint8(x1 * 255)
            pred_frm1 = np.uint8(gx1 * 255)

            for b in range(configs.batch_size):
                score1, _ = compare_ssim(pred_frm1[b], real_frm1[b], full=True, multichannel=True)
                ssim1[i] += score1

            psnr1[i] += metrics.batch_psnr(pred_frm1, real_frm1)




        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            if not os.path.exists(path):
                os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims1[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(configs.total_length - configs.input_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out1[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)

        test_input_handle1.next()
    #print(label)
    avg_mse1 = avg_mse1 / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse1) + str(avg_mse2))
    writer.add_scalar("Test" + "/mse" + str(mark), avg_mse1, itr)
    for i in range(configs.total_length - configs.input_length):
        print(img_mse1[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim1, dtype=np.float32) / (configs.batch_size * batch_id)
    psnr = np.asarray(psnr1, dtype=np.float32) / batch_id
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])



def test_radar_multitask(model, test_input_handle1, configs, itr, writer, mark):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    test_input_handle1.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir + str(mark), str(itr))
    if not os.path.exists(configs.gen_frm_dir + str(mark)):
        os.mkdir(configs.gen_frm_dir + str(mark))
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    avg_mse1 = 0
    avg_mse2 = 0
    batch_id = 0
    img_mse, ssim, psnr, fmae, csi, sharp, csi_30, csi_40, csi_50 = [], [], [], [], [], [], [], [], []
    gdl = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)
        # if configs.dataset_name == 'radar_weather':
        csi.append(0)
        csi_30.append(0)
        csi_40.append(0)
        csi_50.append(0)


    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    while (test_input_handle1.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims1 = test_input_handle1.get_batch()
        test_dat1 = preprocess.reshape_patch(test_ims1, configs.patch_size)

        img_gen1 = model.test(test_dat1, real_input_flag)

        img_gen1 = preprocess.reshape_patch_back(img_gen1, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_gen_length = img_gen1.shape[1]
        img_out1 = img_gen1[:, -output_length:]
        #print("img_gen_length", img_gen_length, img_gen.shape)

        # MSE per frame
        for i in range(output_length):
            x1 = test_ims1[:, i + configs.input_length, :, :, :]
            gx1 = img_out1[:, i, :, :, :]
            fmae[i] += metrics.batch_mae_frame_float(gx1, x1)
            gx1 = np.maximum(gx1, 0)
            gx1 = np.minimum(gx1, 1)
            mse1 = np.square(x1 - gx1).sum()
            img_mse[i] += mse1
            avg_mse1 += mse1



            real_frm1 = np.uint8(x1 * 255)
            pred_frm1 = np.uint8(gx1 * 255)

            for b in range(configs.batch_size):
                score1, _ = compare_ssim(pred_frm1[b], real_frm1[b], full=True, multichannel=True)
                ssim[i] += score1

            psnr[i] += metrics.batch_psnr(pred_frm1, real_frm1)
            csi[i] += metrics.cal_csi(pred_frm1, real_frm1, 20)
            csi_30[i] += metrics.cal_csi(pred_frm1, real_frm1, 30)
            csi_40[i] += metrics.cal_csi(pred_frm1, real_frm1, 40)
            csi_50[i] += metrics.cal_csi(pred_frm1, real_frm1, 50)




        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            if not os.path.exists(path):
                os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims1[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(configs.total_length - configs.input_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out1[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)

        test_input_handle1.next()
    #print(label)
    avg_mse1 = avg_mse1 / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse1) + str(avg_mse2))
    writer.add_scalar("Test" + "/mse" + str(mark), avg_mse1, itr)
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    fmae = np.asarray(fmae, dtype=np.float32) / batch_id
    # if configs.dataset_name == 'radar_weather':
    csi = np.asarray(csi, dtype=np.float32) / batch_id

    csi_30 = np.asarray(csi_30, dtype=np.float32) / batch_id

    csi_40 = np.asarray(csi_40, dtype=np.float32) / batch_id

    csi_50 = np.asarray(csi_50, dtype=np.float32) / batch_id

    print('fmae per frame: ' + str(np.mean(fmae)))
    for i in range(configs.total_length - configs.input_length):
        print(fmae[i])

    print('csi per frame: ' + str(np.mean(csi)))
    for i in range(configs.total_length - configs.input_length):
        print(csi[i])

    print('csi_30 per frame: ' + str(np.mean(csi_30)))
    for i in range(configs.total_length - configs.input_length):
        print(csi_30[i])

    print('csi_40 per frame: ' + str(np.mean(csi_40)))
    for i in range(configs.total_length - configs.input_length):
        print(csi_40[i])

    print('csi_50 per frame: ' + str(np.mean(csi_50)))
    for i in range(configs.total_length - configs.input_length):
        print(csi_50[i])












def test(model, test_input_handle, configs, itr, writer):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)

        img_gen = model.test(test_dat, real_input_flag)
        #print(img_gen.shape)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_gen_length = img_gen.shape[1]
        img_out = img_gen[:, -output_length:]
        #print("img_gen_length", img_gen_length, img_gen.shape)

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :configs.img_channel]
            gx = img_out[:, i, :, :, :configs.img_channel]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)
            for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                ssim[i] += score

            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            if not os.path.exists(path):
                os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8((1-test_ims[0, i, :, :, :configs.img_channel]) * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(configs.total_length - configs.input_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_gen[0, i, :, :, :configs.img_channel]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8((1-img_pd) * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()
    #print(label)
    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    writer.add_scalar("Test" + "/mse", avg_mse, itr)
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])


def test_single_radar(model, test_input_handle1, configs, itr, writer):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    test_input_handle1.begin(do_shuffle=False)
    mark = 1
    res_path = os.path.join(configs.gen_frm_dir + str(mark), str(itr))
    if not os.path.exists(configs.gen_frm_dir + str(mark)):
        os.mkdir(configs.gen_frm_dir + str(mark))
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    avg_mse1 = 0
    avg_mse2 = 0
    batch_id = 0
    img_mse, ssim, psnr, fmae, csi, sharp, csi_30, csi_40, csi_50 = [], [], [], [], [], [], [], [], []
    gdl = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)
        # if configs.dataset_name == 'radar_weather':
        csi.append(0)
        csi_30.append(0)
        csi_40.append(0)
        csi_50.append(0)


    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    while (test_input_handle1.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims1 = test_input_handle1.get_batch()
        test_dat1 = preprocess.reshape_patch(test_ims1, configs.patch_size)

        img_gen1 = model.test(test_dat1, real_input_flag)

        img_gen1 = preprocess.reshape_patch_back(img_gen1, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_gen_length = img_gen1.shape[1]
        img_out1 = img_gen1[:, -output_length:]
        #print("img_gen_length", img_gen_length, img_gen.shape)

        # MSE per frame
        for i in range(output_length):
            x1 = test_ims1[:, i + configs.input_length, :, :, :]
            gx1 = img_out1[:, i, :, :, :]
            fmae[i] += metrics.batch_mae_frame_float(gx1, x1)
            gx1 = np.maximum(gx1, 0)
            gx1 = np.minimum(gx1, 1)
            mse1 = np.square(x1 - gx1).sum()
            img_mse[i] += mse1
            avg_mse1 += mse1



            real_frm1 = np.uint8(x1 * 255)
            pred_frm1 = np.uint8(gx1 * 255)

            for b in range(configs.batch_size):
                score1, _ = compare_ssim(pred_frm1[b], real_frm1[b], full=True, multichannel=True)
                ssim[i] += score1

            psnr[i] += metrics.batch_psnr(pred_frm1, real_frm1)
            csi[i] += metrics.cal_csi(pred_frm1, real_frm1, 20)
            csi_30[i] += metrics.cal_csi(pred_frm1, real_frm1, 30)
            csi_40[i] += metrics.cal_csi(pred_frm1, real_frm1, 40)
            csi_50[i] += metrics.cal_csi(pred_frm1, real_frm1, 50)




        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            if not os.path.exists(path):
                os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims1[0, i, :, :, :] * 255)
                img_gt = add_palette(img_gt, configs.img_width)
                #cv2.imwrite(file_name, img_gt)
                img_gt.save(file_name)
                #cv2.imwrite(file_name, img_gt)
            for i in range(configs.total_length - configs.input_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out1[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                img_pd = add_palette(img_pd, configs.img_width)
                #cv2.imwrite(file_name, img_pd)
                img_pd.save(file_name)
                #cv2.imwrite(file_name, img_pd)

        test_input_handle1.next()
    #print(label)
    avg_mse1 = avg_mse1 / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse1) + str(avg_mse2))
    writer.add_scalar("Test" + "/mse" + str(mark), avg_mse1, itr)
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    fmae = np.asarray(fmae, dtype=np.float32) / batch_id
    # if configs.dataset_name == 'radar_weather':
    csi = np.asarray(csi, dtype=np.float32) / batch_id

    csi_30 = np.asarray(csi_30, dtype=np.float32) / batch_id

    csi_40 = np.asarray(csi_40, dtype=np.float32) / batch_id

    csi_50 = np.asarray(csi_50, dtype=np.float32) / batch_id

    print('fmae per frame: ' + str(np.mean(fmae)))
    for i in range(configs.total_length - configs.input_length):
        print(fmae[i])

    print('csi per frame: ' + str(np.mean(csi)))
    for i in range(configs.total_length - configs.input_length):
        print(csi[i])

    print('csi_30 per frame: ' + str(np.mean(csi_30)))
    for i in range(configs.total_length - configs.input_length):
        print(csi_30[i])

    print('csi_40 per frame: ' + str(np.mean(csi_40)))
    for i in range(configs.total_length - configs.input_length):
        print(csi_40[i])

    print('csi_50 per frame: ' + str(np.mean(csi_50)))
    for i in range(configs.total_length - configs.input_length):
        print(csi_50[i])





def test_radar(model, test_input_handle1, configs, itr, writer):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    test_input_handle1.begin(do_shuffle=False)
    mark = 1
    res_path = os.path.join(configs.gen_frm_dir + str(mark), str(itr))
    if not os.path.exists(configs.gen_frm_dir + str(mark)):
        os.mkdir(configs.gen_frm_dir + str(mark))
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    avg_mse1 = 0
    avg_mse2 = 0
    batch_id = 0
    img_mse, ssim, psnr, fmae, csi, sharp, csi_30, csi_40, csi_50 = [], [], [], [], [], [], [], [], []
    gdl = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)
        # if configs.dataset_name == 'radar_weather':
        csi.append(0)
        csi_30.append(0)
        csi_40.append(0)
        csi_50.append(0)


    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    while (test_input_handle1.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims1 = test_input_handle1.get_batch()
        test_dat1 = preprocess.reshape_patch(test_ims1, configs.patch_size)

        img_gen1 = model.test(test_dat1, real_input_flag)

        img_gen1 = preprocess.reshape_patch_back(img_gen1, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_gen_length = img_gen1.shape[1]
        img_out1 = img_gen1[:, -output_length:]
        #print("img_gen_length", img_gen_length, img_gen.shape)

        # MSE per frame
        for i in range(output_length):
            x1 = test_ims1[:, i + configs.input_length, :, :, :]
            gx1 = img_out1[:, i, :, :, :]
            fmae[i] += metrics.batch_mae_frame_float(gx1, x1)
            gx1 = np.maximum(gx1, 0)
            gx1 = np.minimum(gx1, 1)
            mse1 = np.square(x1 - gx1).sum()
            img_mse[i] += mse1
            avg_mse1 += mse1



            real_frm1 = np.uint8(x1 * 255)
            pred_frm1 = np.uint8(gx1 * 255)
            #real_frm1 = np.expand_dims(real_frm1.max(axis=-1), axis=-1)

            for b in range(configs.batch_size):
                #print(pred_frm1[b].shape, real_frm1[b].shape)
                score1, _ = compare_ssim(pred_frm1[b], real_frm1[b], full=True, multichannel=True)
                ssim[i] += score1
            
            #print(pred_frm1.shape, real_frm1.shape)
            #pred_frm1 = np.expand_dims(pred_frm1.max(axis=-1), axis=-1)
            #real_frm1 = np.expand_dims(real_frm1.max(axis=-1), axis=-1)

            psnr[i] += metrics.batch_psnr(pred_frm1, real_frm1)
            csi[i] += metrics.cal_csi(pred_frm1, real_frm1, 20)
            csi_30[i] += metrics.cal_csi(pred_frm1, real_frm1, 30)
            csi_40[i] += metrics.cal_csi(pred_frm1, real_frm1, 40)
            csi_50[i] += metrics.cal_csi(pred_frm1, real_frm1, 50)




        # save prediction examples
        
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            if not os.path.exists(path):
                os.mkdir(path)
            for i in range(configs.total_length):
                if configs.dataset_name != 'radar_multi_gz':
                    name = 'gt' + str(i + 1) + '.png'
                    file_name = os.path.join(path, name)
                    img_gt = np.uint8(test_ims1[0, i, :, :, :] * cal_decay(i) * 255)
                    cv2.imwrite(file_name, img_gt)
                
                else:
                    for j in range(configs.img_channel):
                    #for j in range(1):
                        name = str(j) + '_gt_' + str(i + 1) + '.png'
                        file_name = os.path.join(path, name)
                        img_gt = np.uint8(test_ims1[0, i, :, :, j] * cal_decay(i) * 255)
                        cv2.imwrite(file_name, img_gt)
            

            '''
                
            if configs.model_name != 'trajGRU' and configs.model_name != 'srvp':
                for i in range(configs.total_length - 1):
                    if configs.dataset_name != 'radar_multi_gz':
                        name = 'pd' + str(i + 2) + '.png'
                        file_name = os.path.join(path, name)
                        img_pd = img_gen1[0, i, :, :, :]
                        img_pd = np.maximum(img_pd, 0)
                        img_pd = np.minimum(img_pd, 1)
                        img_pd = np.uint8(img_pd * 255)
                        cv2.imwrite(file_name, img_pd)
                    
                    else:
                        for j in range(configs.img_channel):
                        #for j in range(1):
                            name = str(j) + '_pd_' + str(i + 2) + '.png'
                            file_name = os.path.join(path, name)
                            img_pd = img_gen1[0, i, :, :, j]
                            img_pd = np.maximum(img_pd, 0)
                            img_pd = np.minimum(img_pd, 1)
                            img_pd = np.uint8(img_pd * 255)
                            cv2.imwrite(file_name, img_pd)
            '''
                    

        test_input_handle1.next()
    #print(label)
    avg_mse1 = avg_mse1 / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse1) + str(avg_mse2))
    writer.add_scalar("Test" + "/mse" + str(mark), avg_mse1, itr)
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    fmae = np.asarray(fmae, dtype=np.float32) / batch_id
    # if configs.dataset_name == 'radar_weather':
    csi = np.asarray(csi, dtype=np.float32) / batch_id

    csi_30 = np.asarray(csi_30, dtype=np.float32) / batch_id

    csi_40 = np.asarray(csi_40, dtype=np.float32) / batch_id

    csi_50 = np.asarray(csi_50, dtype=np.float32) / batch_id

    print('fmae per frame: ' + str(np.mean(fmae)))
    for i in range(configs.total_length - configs.input_length):
        print(fmae[i])

    print('csi per frame: ' + str(np.mean(csi)))
    for i in range(configs.total_length - configs.input_length):
        print(csi[i])

    print('csi_30 per frame: ' + str(np.mean(csi_30)))
    for i in range(configs.total_length - configs.input_length):
        print(csi_30[i])

    print('csi_40 per frame: ' + str(np.mean(csi_40)))
    for i in range(configs.total_length - configs.input_length):
        print(csi_40[i])

    print('csi_50 per frame: ' + str(np.mean(csi_50)))
    for i in range(configs.total_length - configs.input_length):
        print(csi_50[i])




