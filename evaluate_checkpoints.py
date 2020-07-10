import os
import argparse
import subprocess

def run(generator_model, discriminator_model, ckpt_exp_dir, args):
    out_stream = open(os.path.join(args.exp_dir, 'evaluate_outputs.log'), 'wb', 0)
    cmd = ['python', os.path.join('.', 'CODE', 'evaluate.py'),
           '--generator', str(args.generator),
           '--discriminator', str(args.discriminator),
           '--generator_model', generator_model,
           '--discriminator_model', discriminator_model,
           '--exp_dir', ckpt_exp_dir,
           '--data_dir', str(args.data_dir),
           '--data_split', str(args.data_split),
           '--batch_size', str(args.batch_size),
           '--num_workers', str(args.num_workers)
           ]

    if args.crop_size is not None:
        cmd.append('--crop_size')
        for i in args.args.crop_size:
            cmd.append(str(i))
    if args.exp_img:
        cmd.append('--exp_img')
    if args.gpu_id != [0]:
        cmd.append('--gpu_id')
        for i in args.gpu_id:
            cmd.append(str(i))

    print(' '.join(cmd))
    subprocess.Popen(cmd, stdout=out_stream, shell=False).wait()
    print('OK!')


def main(args):
    assert os.path.exists(args.exp_dir)
    assert os.path.exists(args.data_dir)
    model_names = {'generator': 'G_{}.pth', 'discriminator': 'D_{}.pth'}

    eval_exp_dir = os.path.join(args.exp_dir, 'evaluation')
    if not os.path.exists(eval_exp_dir):
        os.mkdir(eval_exp_dir)

    assert len(args.range) == 3
    for ckpt_index in range(args.range[0], args.range[1], args.range[2]):
        ckpt_exp_dir = os.path.join(eval_exp_dir, 'checkpoint_{}'.format(ckpt_index))

        ckpt_models_dir = os.path.join(args.exp_dir, 'checkpoints')
        generator_model = os.path.join(ckpt_models_dir, model_names['generator'].format(ckpt_index))
        discriminator_model = os.path.join(ckpt_models_dir, model_names['discriminator'].format(ckpt_index))
        assert os.path.exists(generator_model) and os.path.exists(discriminator_model)

        run(generator_model, discriminator_model, ckpt_exp_dir, args)

    print('ALL FINISHED!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir',        type=str,
                        default='.')
    parser.add_argument('--range',			type=int,       nargs='*',
                        default=(5, 201, 5))
    parser.add_argument('--generator',      type=str,
                        default='pixel2pixel')
    parser.add_argument('--discriminator',  type=str,
                        default='pixel2pixel')
    parser.add_argument('--exp_img',        action="store_true",
                        default=False)
    parser.add_argument('--data_dir',       type=str,
                        default=os.path.join('.', 'CODE', 'data', 'RainCityScapes'))
    parser.add_argument('--data_split',     type=float,
                        default=0)
    parser.add_argument('--crop_size',      type=int,       nargs='*',
                        default=None)
    parser.add_argument('--batch_size',     type=int,
                        default=1)
    parser.add_argument('--num_workers',    type=int,
                        default=0)
    parser.add_argument('--gpu_id',         type=int,       nargs='*',
                        default=[0])
    args = parser.parse_args()

    main(args)