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

    res_dir = '/scratch/clear/abietti/results/ckn/stl10_white_3_11/'
    template = 'ckn_res0v_{:.0e}_decay2_cropresize_mc.pkl'

    filter_saga = lambda p: 'saga' in p['name'] and p['lr'] > 0.5

    for (lmbda, legend, step, last) in [(1e-3, True, 6, -16), (1e-4, False, 4, -16), (1e-5, False, 6, None)]:
        res = pickle.load(open(os.path.join(res_dir, 'accs', template.format(lmbda)), 'rb'))
        curves.plot_loss(res, ty='train', log=True, step=step, last=last,
                         small=True, legend=legend,
                         title='STL-10 ckn, $\mu = 10^{{{}}}$'.format(int(math.log10(lmbda))),
                         filter_fn=filter_saga)

        plt.savefig('/home/thoth/abietti/shared/stl10_ckn_mu_{:.0e}.pdf'.format(lmbda), format='pdf', bbox_inches='tight', pad_inches=0)
        pp.savefig()

    template = 'scattering_res0v_{:.0e}_prox1e-4_decay1_gamma_mc.pkl'

    for (lmbda, legend, step) in [(1e-3, False, 8), (1e-4, False, 7), (1e-5, False, 6)]:
        res = pickle.load(open(os.path.join(res_dir, 'accs_scat', template.format(lmbda)), 'rb'))
        curves.plot_loss(res, ty='train', log=True, step=step, last=None,
                         small=True, legend=legend, ylabel='F - F*',
                         title='STL-10 scattering, $\mu = 10^{{{}}}$'.format(int(math.log10(lmbda))),
                         filter_fn=filter_saga)

        plt.savefig('/home/thoth/abietti/shared/stl10_scat_mu_{:.0e}.pdf'.format(lmbda), format='pdf', bbox_inches='tight', pad_inches=0)
        pp.savefig()

    pp.close()
