import argparse
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import curves


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='curves for dropout training on imdb')
    parser.add_argument('--pdf-file', default=None, help='pdf file to save to')
    parser.add_argument('--pkl-file', default=None, help='pickle file to load')
    args = parser.parse_args()

    plots = pickle.load(open(args.pkl_file, 'rb'))

    pp = PdfPages(args.pdf_file)

    for (lmbda, delta), plot_res in plots:
        curves.plot_loss(plot_res, ty='train', log=True, step=1, last=-40,
                         small=True, legend=delta > 0.2, title='gene dropout, $\delta$ = {:.2f}'.format(delta))
        # plt.savefig('/home/thoth/abietti/shared/gene_dropout_avg_easy_delta_{:.2f}.pdf'.format(delta), format='pdf', bbox_inches='tight', pad_inches=0)
        pp.savefig()

    pp.close()
