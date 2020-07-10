import os
import argparse
import importlib

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from train import evaluate
from libs.dataset import PairedImageDataset
from libs.data_logging import make_dir, Logging


def main(args):
    # set the seed for generating random numbers
    torch.manual_seed(1)

    # specify the device used for computing: GPU ('cuda') or CPU ('cpu')
    if args.gpu_id != [0] and len(args.gpu_id) == 1:
        device = torch.device("cuda:{}".format(args.gpu_id[0]))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define models
    generator = importlib.import_module('models.{}'.format(args.generator)).Generator()
    #discriminator = importlib.import_module('models.{}'.format(args.discriminator)).Discriminator()

    # Load trained models
    if args.generator_model is not None:
        generator.load_state_dict(torch.load(args.generator_model))
    #if args.discriminator_model is not None:
    #    discriminator.load_state_dict(torch.load(args.discriminator_model))

    # if using multiple GPUs, deploy distributed data parallel for models
    if len(args.gpu_id) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=args.gpu_id)
    #    discriminator = torch.nn.DataParallel(discriminator, device_ids=args.gpu_id)

    # deploy models to device
    generator = generator.to(device)
    #discriminator = discriminator.to(device)

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
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # create directory to store the evaluation results
    evaluation_exp_dir = make_dir(args.exp_dir, allow_repeat=True)

    log = Logging()
    table_heads = ['Data Index', 'PSNR', 'SSIM']

    for name, data_loader in [('training_set', train_data_loader), ('testing_set', test_data_loader)]:
        # create a new sheet in the log file to store results of the corresponding dataset
        log.add_sheet(name, table_heads)

        # evaluation
        if args.exp_img:
            exp_img_dir = make_dir(os.path.join(evaluation_exp_dir, name))
            psnr, ssim = evaluate(generator, device, data_loader, exp_img_dir=exp_img_dir)
        else:
            psnr, ssim = evaluate(generator, device, data_loader)

        assert len(psnr) == len(ssim)
        for i in range(len(psnr)):
            log[name].add_value(table_heads[0], i)
            log[name].add_value(table_heads[1], psnr[i])
            log[name].add_value(table_heads[2], ssim[i])

    log.save(evaluation_exp_dir, file_name='evaluation_log.xlsx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rain & Fog Removal Model (Evaluation)')
    parser.add_argument('--generator',      type=str,
                        help='specify the module name of the generator model')
    parser.add_argument('--discriminator',  type=str,
                        help='specify the module name of the discriminator model')
    parser.add_argument('--generator_model',        type=str,   default=None,
                        help='specify the path to where the trained generator model is saved')
    parser.add_argument('--discriminator_model',    type=str,   default=None,
                        help='specify the path to where the trained discriminator model is saved')
    parser.add_argument('--exp_dir',        type=str,
                        help='specify the path to the directory to save the evaluation results and logging data')
    parser.add_argument('--exp_img', action="store_true", default=False,
                        help='to export images during evaluation')
    parser.add_argument('--data_dir',       type=str,
                        help='specify the path to the directory of the dataset')
    parser.add_argument('--data_split',     type=float,     default=0,
                        help='specify the partition ratio of the training and testing sets (default 0 for pre-split dataset)')
    parser.add_argument('--crop_size',      type=int,       nargs='*',          default=None,
                        help='specify the size of the crop for the input image in training (default None for not cropping')
    parser.add_argument('--batch_size',     type=int,       default=1,
                        help='define the batch size')
    parser.add_argument('--num_workers',    type=int,       default=0,
                        help='define the number of workers for the data loaders')
    parser.add_argument('--gpu_id',         type=int,       nargs='*',          default=[0],
                        help='define the id(s) of the GPUs to be used')
    args = parser.parse_args()

    main(args)
