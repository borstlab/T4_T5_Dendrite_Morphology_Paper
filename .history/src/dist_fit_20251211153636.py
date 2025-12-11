from fitter import Fitter
import scipy.stats as stats
import numpy as np
import pandas as pd

from .figure_Tools import Subtypes, Subtype_colours


class DistributionFitter:
    def __init__(self, df, DV_col, isExternal, xmin, xmax, n_bins, progress=False):
        """
        Core class to fit distributions to subsets of data and summarize results.
        """
        self.df = df
        self.DV_col = DV_col
        self.isExternal = isExternal
        self.xmin = xmin
        self.xmax = xmax
        self.n_bins = n_bins
        self.progress = progress

        self.nTypes = ['T4', 'T5']
        self.nSubtypes = ['a', 'b', 'c', 'd']
        self.output_dict = {t: {s: [] for s in self.nSubtypes} for t in self.nTypes}

    def fit_distributions(self, distributions=None):
        """
        Fit distributions to all type-subtype combinations.
        """
        if distributions is None:
            distributions = ['lognorm', 'gamma', 'wald', 'expon', 'fisk', 'weibull_min']

        for t in self.nTypes:
            for s in self.nSubtypes:
                if self.progress:
                    print(f"Fitting to {t}{s}")
                data = self.df.loc[(self.df.Subtype == t + s) & (self.df.isExternal == self.isExternal), self.DV_col].values
                f = Fitter(data, xmin=self.xmin, xmax=self.xmax, bins=self.n_bins, distributions=distributions)
                f.fit()
                self.output_dict[t][s] = {'Data': data, 'Fit': f}
        return self.output_dict

    @staticmethod
    def log_lik(data, f, dist_name):
        """
        Compute log-likelihood for given data and distribution.
        """
        dist = getattr(stats, dist_name)
        params = f.fitted_param[dist_name]

        if dist.shapes:
            n_shapes = len(dist.shapes.split(','))
            shapes = params[:n_shapes]
        else:
            shapes = ()
        loc = params[len(shapes)]
        scale = params[len(shapes) + 1]

        logpdf_vals = dist.logpdf(data, *shapes, loc=loc, scale=scale)
        return np.sum(logpdf_vals)

    def bestFit_summary(self):
        """
        Summarize best-fitting distributions based on BIC.
        """
        summary_data = {
            'Type': [],
            'Subtype': [],
            'Best_distribution': [],
            'BIC': [],
            'Log_likelihood': [],
            'Parameters': []
        }

        for t in self.nTypes:
            for s in self.nSubtypes:
                fit_obj = self.output_dict[t][s]['Fit']
                data = self.output_dict[t][s]['Data']

                best_dist = fit_obj.summary(Nbest=1, method='bic', plot=False)['bic'].index[0]
                best_bic = fit_obj.summary(Nbest=1, method='bic', plot=False)['bic'].values[0]
                params = fit_obj.fitted_param[best_dist]
                ll = self.log_lik(data, fit_obj, best_dist)

                summary_data['Type'].append(t)
                summary_data['Subtype'].append(s)
                summary_data['Best_distribution'].append(best_dist)
                summary_data['BIC'].append(best_bic)
                summary_data['Log_likelihood'].append(ll)
                summary_data['Parameters'].append(params)

        return pd.DataFrame(summary_data)

    @staticmethod
    def _plot_pdf(ax, xmin, xmax, num_points, dist_name, params=(), c='r', label=""):
        """
        Plot a single PDF curve on an axis.
        """
        x = np.linspace(xmin, xmax, num_points)

        try:
            dist = getattr(stats, dist_name)
        except AttributeError:
            raise ValueError(f"Distribution '{dist_name}' not found in scipy.stats")

        y = dist.pdf(x, *params)
        ax.plot(x, y, c=c, label=label)

    def plot_density(self, ax, dist_name, scale=1):
        """
        Plot fitted PDFs and pooled data histogram.
        """
        for i, st in enumerate(Subtypes):
            t = st[:2]
            s = st[-1]
            c = fT.Subtype_colours[i]

            params = self.output_dict[t][s]['Fit'].fitted_param[dist_name]
            self._plot_pdf(ax, self.xmin, self.xmax, 1000, dist_name, params, c=c, label=st)

        # Pooled data histogram
        data = self.df.loc[self.df.isExternal == self.isExternal, self.DV_col].values
        counts, bins = np.histogram(data, range=(self.xmin, self.xmax), bins=self.n_bins, density=True)
        ax.bar(x=bins[1:], height=counts, width=(bins[1] - bins[0]) * scale, color='gray', alpha=0.4, label='Observed Pooled data')
