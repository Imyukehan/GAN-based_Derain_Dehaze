import os
import argparse
import subprocess


def run(G, D, exp_name, data_dir, crop_size, learn_rate, batch_size, args):
    out_stream = open(os.path.join(args.exp_dir, 'train_outputs.log'), 'wb', 0)
    cmd = ['python',                    os.path.join('.', 'CODE', 'train.py'),
           '--generator',               G,
           '--discriminator',           D,
           '--exp_dir',                 os.path.join(args.exp_dir, exp_name),
           '--data_dir',                data_dir,
           '--data_split',              str(args.data_split),
           '--crop_size',               str(crop_size),
           '--num_epochs',              str(args.num_epochs),
           '--checkpoint',              str(args.checkpoint),
           '--learn_rate',              str(learn_rate),
           '--batch_size',              str(batch_size),
           '--num_workers',             str(args.num_workers)]

    if args.gpu_id != [0]:
        cmd.append('--gpu_id')
        for i in args.gpu_id:
            cmd.append(str(i))

    print(' '.join(cmd))
    subprocess.Popen(cmd, stdout=out_stream, shell=False).wait()
    print('OK!')
    return exp_name


def main(args):
    assert os.path.exists(args.exp_dir)
    assert os.path.exists(os.path.join('.', 'CODE', 'train.py'))
    data_dir = os.path.join('.', 'CODE', 'data', args.dataset)
    assert os.path.exists(data_dir)

    for G in args.Gs:
        for D in args.Ds:
            for crop in args.crops:
                for lr in args.lrs:
                    for batch in args.batches:
                        exp_name = '{}_G={}_D={}_crop={}_batch={}_lr={}'.format(args.dataset, G, D, crop, batch, lr)
                        os.mkdir(os.path.join(args.exp_dir, exp_name))
                        run(G, D, exp_name, data_dir, crop, lr, batch, args)
    print('All Finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rain & Fog Removal Model')
    parser.add_argument('--Gs',         type=str,       nargs='*')
    parser.add_argument('--Ds',         type=str,       nargs='*')
    parser.add_argument('--exp_dir',    type=str)
    parser.add_argument('--dataset',    type=str)
    parser.add_argument('--crops',      type=int,       nargs='*',      default=[None])
    parser.add_argument('--lrs',        type=float,     nargs='*',      default=[0.0002])
    parser.add_argument('--batches',    type=int,       nargs='*',      default=[1])
    parser.add_argument('--data_split', type=float,     default=0)
    parser.add_argument('--num_epochs', type=int,       default=200)
    parser.add_argument("--checkpoint", type=int,       default=-1)
    parser.add_argument('--num_workers',type=int,       default=0)
    parser.add_argument('--gpu_id',     type=int,       nargs='*',      default=[0])
    args = parser.parse_args()

    main(args)
