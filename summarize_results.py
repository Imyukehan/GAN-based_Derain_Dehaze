import os
import xlrd
import argparse
import xlsxwriter
import numpy as np

def main(args):
    heads = ['Epoch of Training', 'Average PSNR', 'PSNR Standard Deviation', 'Average SSIM', 'SSIM Standard Deviation']
    results = {'training_set': {head: [] for head in heads}, 'testing_set': {head: [] for head in heads}}

    assert len(args.range) == 3
    for checkpoint_index in range(args.range[0], args.range[1], args.range[2]):
        print(checkpoint_index, end=' - ')
        dir_name = 'checkpoint_%d_1' %checkpoint_index
        read_book = xlrd.open_workbook(os.path.join(args.exp_dir, dir_name, 'evaluation_log.xlsx'))
        for sheet_name in ['training_set', 'testing_set']:
            read_sheet = read_book.sheet_by_name(sheet_name)
            psnr = list(map(float, read_sheet.col_values(1, start_rowx=1)))
            ssim = list(map(float, read_sheet.col_values(2, start_rowx=1)))
            print(sheet_name, len(psnr), len(ssim))

            results[sheet_name]['Epoch of Training'].append(checkpoint_index)
            results[sheet_name]['Average PSNR'].append(float(sum(psnr) / len(psnr)))
            results[sheet_name]['PSNR Standard Deviation'].append(float(np.std(psnr)))
            results[sheet_name]['Average SSIM'].append(float(sum(ssim) / len(ssim)))
            results[sheet_name]['SSIM Standard Deviation'].append(float(np.std(ssim)))


    write_book = xlsxwriter.Workbook('summary.xlsx')

    for sheet_name in results:
        write_sheet = write_book.add_worksheet(sheet_name)
        table = results[sheet_name]
        col = 0
        for head in table:
            write_sheet.write(0, col, head)
            row = 1
            for value in table[head]:
                write_sheet.write(row, col, value)
                row += 1
            col += 1

    write_book.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir',        type=str,
                        default='.')
    parser.add_argument('--range',			type=int,       nargs='*',
                        default=(5, 201, 5))
    args = parser.parse_args()

    main(args)
