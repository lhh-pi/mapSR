from __future__ import print_function
import argparse
import os
from os.path import join
import torch
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_eval_set
import time
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import models
import yaml

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--config', default="")
parser.add_argument('--upscale_factor', type=int, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
# parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')  # for windows
parser.add_argument("--cuda", type=bool, default=True, help="Use cuda?")
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='test')
parser.add_argument('--output', default='Results/', help='Location to save checkpoint models')
parser.add_argument('--model', default="")

args = parser.parse_args()

torch.backends.cudnn.benchmark = True

# 加载配置文件
# args.config = "../configs/test/swinir_x2_test1.yaml"
args.config = "../configs/test/dbpn_x2_test1.yaml"
# args.config = "../configs/test/esrt_x2_test1.yaml"
# args.config = "../configs/test/rcan_x2_test1.yaml"
# args.config = "../configs/test/edsr_x2_test1.yaml"
# args.config = "../configs/test/swinsr_x2_test1.yaml"
# args.config = "../configs/test/swinsrv2_x2_test1.yaml"
# args.config = "../configs/test/swinsrv2_x2_test2.yaml"
# args.config = "../configs/test/swinsrv3_x2_test1.yaml"
# args.config = "../configs/test/swinsrv3_x2_test4.yaml"
# args.config = "../configs/test/swinsrv4_x2_test1.yaml"
# args.config = "../configs/test/swinsrv4_x2_test2.yaml"
# args.config = "../configs/test/swinsrv4_x2_test3.yaml"


# args.config = "../configs/test/swinir_x3_test1.yaml"
# args.config = "../configs/test/dbpn_x3_test1.yaml"
# args.config = "../configs/test/esrt_x3_test1.yaml"
# args.config = "../configs/test/rcan_x3_test1.yaml"
# args.config = "../configs/test/edsr_x3_test1.yaml"

<<<<<<< HEAD
# 开题绘图
args.input_dir = "kaiti"
=======
# args.config = "../configs/test/swinir_L_x2_test1.yaml"

>>>>>>> 0db0559d07919a8eba2cfaa49a2a136191e311a9
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

scale = config['scale']
args.upscale_factor = scale
args.model = config['model']['sd']
gpus_list = range(args.gpus)
print(args)

cuda = args.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Loading datasets')
args.test_dataset = "LRbic" + "x{}".format(args.upscale_factor)
test_set = get_eval_set(os.path.join(args.input_dir, args.test_dataset), args.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)

print('===> Building model')
model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])


def eval():
    model.eval()
    for batch in testing_data_loader:
        with torch.no_grad():
            input, bicubic, name = Variable(batch[0]), Variable(batch[1]), batch[2]
        if cuda:
            input = input.cuda(gpus_list[0])
            # bicubic = bicubic.cuda(gpus_list[0])

        t0 = time.time()

        with torch.no_grad():
            prediction = model(input)

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        save_img(prediction.cpu().data, name[0])
        # save_img(bicubic.cpu().data, name[0])


def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
    # save img
    save_dir = os.path.join(args.output, args.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = save_dir + '/' + img_name
    cv2.imwrite(save_fn, cv2.cvtColor(save_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


# Eval Start!!!!
eval()


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


image_lr_dir = "Results/LRbic" + "x{}".format(args.upscale_factor)
image_gt_dir = "test/GTmod24"
image_lr_filenames = [join(image_lr_dir, x) for x in os.listdir(image_lr_dir) if is_image_file(x)]
image_gt_filenames = [join(image_gt_dir, x) for x in os.listdir(image_gt_dir) if is_image_file(x)]


def my_test():
    avg_psnr = 0
    avg_ssim = 0

    for index in range(len(image_lr_filenames)):
        img_prediction = cv2.imread(image_lr_filenames[index])
        # img_GT = cv2.imread(image_gt_filenames[index])
        image_name = image_lr_filenames[index].split('/')[-1]
        img_GT = cv2.imread(os.path.join('test/GTmod24', image_name))
        # 去除边界
        img_prediction = img_prediction[scale: -scale, scale: -scale]
        img_GT = img_GT[scale: -scale, scale: -scale]

        psnr = peak_signal_noise_ratio(img_prediction, img_GT)
        ssim = structural_similarity(img_prediction, img_GT, channel_axis=2)
        avg_psnr += psnr
        avg_ssim += ssim
        print(image_name, psnr, ssim)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(image_lr_filenames)))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim / len(image_lr_filenames)))


# my_test()
