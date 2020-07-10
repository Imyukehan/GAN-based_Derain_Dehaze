import os
import cv2
import time
import argparse
import datetime
import importlib
import numpy as np
from skimage.measure import compare_psnr, compare_ssim

import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from libs.dataset import PairedImageDataset
from libs.data_logging import make_dir, Logging
from libs.model_initializtion import weights_init_normal


def main(args):
    # create directory to store checkpoints and experiment results
    ckpt_dir = os.path.join(args.exp_dir, 'checkpoints')
    log_dir = os.path.join(args.exp_dir, 'train_log')
    exp_img_dir = os.path.join(args.exp_dir, 'img_exp')
    make_dir(args.exp_dir)
    make_dir(ckpt_dir)
    make_dir(exp_img_dir)
    make_dir(log_dir)

    # set the seed for generating random numbers
    torch.manual_seed(1)

    # specify the device used for computing: GPU ('cuda') or CPU ('cpu')
    if args.gpu_id != [0] and len(args.gpu_id) == 1:
        device = torch.device("cuda:{}".format(args.gpu_id[0]))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define models
    generator = importlib.import_module('models.{}'.format(args.generator)).Generator()
    discriminator = importlib.import_module('models.{}'.format(args.discriminator)).Discriminator()

    if args.init_epoch != 0:
        # load checkpoints
        generator_ckpt = os.path.join(ckpt_dir, 'G_{}.pth'.format(args.init_epoch))
        discriminator_ckpt = os.path.join(ckpt_dir, 'D_{}.pth'.format(args.init_epoch))
        generator.load_state_dict(torch.load(generator_ckpt))
        discriminator.load_state_dict(torch.load(discriminator_ckpt))
    else:
        # initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # if using multiple GPUs, deploy distributed data parallel for models
    if len(args.gpu_id) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=args.gpu_id)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=args.gpu_id)

    # deploy models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # define image pre-processing methods
    tra = transforms.Compose([
        #transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if args.data_split == 0:
        # create dataset objects in case that the dataset is pre-split
        train_dir = os.path.join(args.data_dir, 'train')
        test_dir = os.path.join(args.data_dir, 'test')
        assert os.path.exists(train_dir) and os.path.exists(test_dir), 'Invalid value for the argument \"--data_split\"'
        train_dataset = PairedImageDataset(train_dir, transforms=tra, crop_size=args.crop_size)
        test_dataset = PairedImageDataset(test_dir, transforms=tra, crop_size=args.crop_size)
    else:
        # in case that the dataset is not pre-split,
        # split the dataset using the given ratio and create corresponding dataset objects
        full_dataset = PairedImageDataset(args.data_dir, transforms=tra, crop_size=args.crop_size)
        train_size = int(args.data_split * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # configure data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # training models
    train({'G': generator, 'D': discriminator}, device, train_data_loader, ckpt_dir, log_dir, args)

    # evaluation
    psnr_list, ssim_list = evaluate(generator, device, test_data_loader, exp_img_dir)
    results_log = [{'psnr': psnr_list[i], 'ssim': ssim_list[i]} for i in range(len(psnr_list))]
    save_log(results_log, 'train_results', args.exp_dir, columns=['psnr', 'ssim'])


def save_log(log_data, name, log_dir, columns=[]):
    # create Logging object with specified sheet names
    log = Logging(sheet_names=[name])

    # arrange log data into the tabular Logging object according to the specific columns
    log[name].add_columns(columns)
    for i in log_data:
        for c in columns:
            log[name].add_value(c, i[c])

    # save the log data as an Excel file
    log.save(log_dir, file_name='{}.xlsx'.format(name), order=columns)


def train_gan(input, ground_truth, G, D, device, criterion_GAN, criterion_pixelwise, optimizer_G, optimizer_D):
    # calculate output of image discriminator (PatchGAN)
    patch = (1, input.size(2) // 2 ** 4, input.size(3) // 2 ** 4)

    # adversarial ground truths
    valid = Variable(torch.Tensor(np.ones((input.size(0), *patch))).to(device), requires_grad=False)
    fake = Variable(torch.Tensor(np.zeros((input.size(0), *patch))).to(device), requires_grad=False)

    ##### train generator #####

    optimizer_G.zero_grad()

    output = G(input)

    pred_fake = D(output, input)

    # define GAN loss
    loss_G = criterion_GAN(pred_fake, valid)
    # define pixel-wise loss (L1 pixel-wise loss between translated image and real image)
    loss_pixel = criterion_pixelwise(output, ground_truth)
    # total loss
    loss_total = loss_G + 100 * loss_pixel

    loss_total.backward()
    optimizer_G.step()

    ##### train discriminator #####

    optimizer_D.zero_grad()

    # real loss
    pred_real = D(ground_truth, input)
    loss_real = criterion_GAN(pred_real, valid)
    # fake loss
    pred_fake = D(output.detach(), input)
    loss_fake = criterion_GAN(pred_fake, fake)
    # total loss
    loss_D = 0.5 * (loss_real + loss_fake)

    loss_D.backward()
    optimizer_D.step()

    return {'D': loss_D, 'G': loss_G, 'pix': loss_pixel, 'total': loss_total}



def train(models, device, data_loader, checkpoints_dir, log_dir, args):
    # define loss functions: MSE loss & L1 loss
    criterion_MSE = torch.nn.MSELoss().to(device)
    criterion_L1 = torch.nn.L1Loss().to(device)

    # define optimizers for each model
    optimizers = {m: torch.optim.Adam(models[m].parameters(), lr=args.learn_rate, betas=(0.5, 0.999)) for m in models}

    prev_time = time.time()

    # train each epoch
    for epoch in range(args.init_epoch + 1, args.num_epochs + 1):
        log = []
        for i, batch in enumerate(data_loader):
            # assign variable
            input = Variable(batch["in"]).to(device)   # input image
            ground_truth = Variable(batch["gt"]).to(device)   # ground truth image

            # train model
            losses = train_gan(input, ground_truth, models['G'], models['D'], device, criterion_MSE, criterion_L1,
                               optimizers['G'], optimizers['D'])

            # record training log data
            log.append(dict({'batch': i}, **losses))

            # estimate time left
            batches_done = (epoch-1) * len(data_loader) + i
            batches_left = args.num_epochs * len(data_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # print log
            print("\r[Epoch {}/{}] [Batch {}/{}] [D loss: {:.4f}, G loss: {:.4f}, pixel loss: {:.4f}, total: {:.4f}] Time: {}".format(
                epoch, args.num_epochs, i, len(data_loader), losses['D'], losses['G'], losses['pix'], losses['total'], time_left),
                end='', flush=True)

        # save model checkpoints
        if args.checkpoint != -1 and epoch % args.checkpoint == 0:
            for name in models:
                path = os.path.join(checkpoints_dir,'{}_{}.pth'.format(name, epoch))
                model_params = models[name].module.state_dict() if len(args.gpu_id) > 1 else models[name].state_dict()
                torch.save(model_params, path)

        # save training log
        save_log(log, 'epoch_{}'.format(epoch), log_dir, columns=['batch', 'D', 'G', 'pix', 'total'])

    # save trained model
    for name in models:
        path = os.path.join(args.exp_dir, '{}_epoch={}.pth'.format(name, args.num_epochs))
        model_params = models[name].module.state_dict() if len(args.gpu_id) > 1 else models[name].state_dict()
        torch.save(model_params, path)

    print('\n----- Training Finished -----\n')


def evaluate(model, device, data_loader, exp_img_dir=None):
    psnr_list, ssim_list = [], []
    for i, batch in enumerate(data_loader):
        input = Variable(batch["in"]).to(device)
        ground_truth = Variable(batch["gt"]).to(device)

        with torch.no_grad():
            model.eval()
            output = model(input)

        for index in range(len(output)):
            image_out = output[index].data.permute(1, 2, 0).cpu().numpy()
            image_gt = ground_truth[index].data.permute(1, 2, 0).cpu().numpy()
            image_out = cv2.normalize(src=image_out, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            image_gt = cv2.normalize(src=image_gt, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            psnr = compare_psnr(image_out, image_gt)
            ssim = compare_ssim(image_out, image_gt, multichannel=True)
            print('PSNR: {:.4f} | SSIM: {:.4f}'.format(psnr, ssim))
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            if exp_img_dir is not None:
                exp_image = torch.cat((input[index].data, output[index].data, ground_truth[index].data), -2)
                save_image(exp_image, os.path.join(exp_img_dir, '{}.png'.format(i)), nrow=1, normalize=True)

    return psnr_list, ssim_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rain & Fog Removal Model')
    parser.add_argument('--generator',      type=str,
                        help='specify the module name of the generator model')
    parser.add_argument('--discriminator',  type=str,
                        help='specify the module name of the discriminator model')
    parser.add_argument('--exp_dir',        type=str,
                        help='specify the path to the directory to save the experiment results and logging data')
    parser.add_argument('--data_dir',       type=str,
                        help='specify the path to the directory of the dataset')
    parser.add_argument('--data_split',     type=float,     default=0,
                        help='specify the partition ratio of the training and testing sets (default 0 for pre-split dataset)')
    parser.add_argument('--crop_size',      type=int,       nargs='*',          default=None,
                        help='specify the size of the crop for the input image in training (default None for not cropping')
    parser.add_argument('--num_epochs',     type=int,       default=200,
                        help='define the number of epochs for the training process')
    parser.add_argument('--init_epoch',     type=int,       default=0,
                        help='specify the initial epoch to start from (specify the corresponding checkpoint to load)')
    parser.add_argument("--checkpoint",     type=int,       default=-1,
                        help="interval between model checkpoints (by defalut, no checkpoint will be saved)")
    parser.add_argument('--learn_rate',     type=float,     default=0.0002,
                        help='define the learning rate')
    parser.add_argument('--batch_size',     type=int,       default=1,
                        help='define the batch size')
    parser.add_argument('--num_workers',    type=int,       default=0,
                        help='define the number of workers for the data loaders')
    parser.add_argument('--gpu_id',         type=int,       nargs='*',          default=[0],
                        help='define the id(s) of the GPUs to be used')
    args = parser.parse_args()

    main(args)
