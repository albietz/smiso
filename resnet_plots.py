import argparse
import pickle
import math
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import curves


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='curves for dropout training on imdb')
    parser.add_argument('--pdf-file', default=None, help='pdf file to save to')
    args = parser.parse_args()

    pp = PdfPages(args.pdf_file)

    res_dir = '/scratch/clear/abietti/results/ckn/resnet/'
    # template = 'resnet_res_{:.0e}_decay2_resize_crop.pkl'
    template = 'resnet_res_{:.0e}_decay1_color.pkl'

    filter_saga = lambda p: 'saga' in p['name'] and p['lr'] > 0.5

    for (lmbda, legend, step) in [(1e-2, True, 2), (1e-3, False, 2), (1e-4, False, 2)]:
        res = pickle.load(open(os.path.join(res_dir, template.format(lmbda)), 'rb'))
        curves.plot_loss(res, ty='train', log=True, step=step, last=-10,
                         small=True, legend=legend,
                         title='ResNet50, $\mu = 10^{{{}}}$'.format(int(math.log10(lmbda))),
                         filter_fn=filter_saga)

        plt.savefig('/home/thoth/abietti/shared/resnet_mu_{:.0e}.pdf'.format(lmbda), format='pdf', bbox_inches='tight', pad_inches=0)
        pp.savefig()

    pp.close()
