import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import transforms

class MultiSequenceGenerator():
    def generate_sequence(self, generating_function, parameter_list, size):
        self.sequence = generating_function((parameter_list), size=(size, len(parameter_list)))

    def generate_dataframe(self, column_name, parameter_list):
        self.df = pd.DataFrame(self.sequence, columns = [f'{column_name}: {i}' for i in parameter_list])

    def calculate_stats(self):
        self.stats = pd.DataFrame()
        self.stats["mean"] = self.df.mean()
        self.stats["Std.Dev"] = self.df.std()
        self.stats["Var"] = self.df.var()
    
    def __init__(self, generating_function, parameter_list, column_name, size=100):
        self.generate_sequence(generating_function, parameter_list, size)
        self.generate_dataframe(column_name, parameter_list)
        self.calculate_stats()

    def get_dataframe(self):
        return self.df

    def get_stats(self):
        return self.stats

    def displot_simple(self):
        sns.displot(self.df, kde=True, stat='probability')
        plt.show()

    def displot_kde(self):
        # convert the dataframe from wide to long
        dfm = self.df.melt(var_name='Distribution')
        # plot
        sns.displot(kind='kde', data=dfm, col='Distribution', col_wrap=3, x='value', fill=True, facet_kws={'sharey': False, 'sharex': False})
        plt.show()

    def kde_plot(self):
        for col in self.df.columns:
            sns.kdeplot(self.df[col], shade=True, label=col)
        plt.legend(loc='best')
        plt.xlabel('value')
        plt.show()

    def column_by_column_distplot(self):
        for col in self.df.columns:
            sns.distplot(self.df[col], kde=True, label=col)
        plt.xlabel('value')
        plt.legend(loc='best')
        plt.show()

    def draw_plots(self):
        self.displot_simple()
        self.displot_kde()
        self.kde_plot()
        self.column_by_column_distplot()

    def catplot_simple(self):
        sns.catplot(data=self.df)
        plt.ylabel('values')
        plt.show()

    def catplot_box(self):
        sns.catplot(data=self.df, kind="box")
        plt.ylabel('values')
        plt.show()

    def catplot_boxen(self):
        sns.catplot(data=self.df, kind="boxen")
        plt.ylabel('values')
        plt.show()

    def catplot_violin(self):
        sns.catplot(data=self.df, kind="violin", bw_adjust=9.5, cut=0, split=True)
        plt.ylabel('values')
        plt.show()

    def catplot_bar(self):
        sns.catplot(data=self.df, kind="bar", height=4, aspect=1.6)
        plt.ylabel('values')
        plt.show()

    def catplot_violin_none_inner(self):
        sns.catplot(data=self.df, kind="violin", color=".9", inner=None)
        sns.swarmplot(data=self.df, size=3)
        plt.show()

    def catplot_violin2(self):
        sns.catplot(data=self.df, kind="violin")
        plt.show()

    def violinplot_with_inner_quart(self):
        sns.violinplot(data=self.df, split=True, inner="quart")
        plt.figure()
        plt.show()

    def draw_stat_plots(self):
        self.catplot_simple()
        self.catplot_box()
        self.catplot_boxen()
        self.catplot_violin()
        self.catplot_bar()
        self.catplot_violin_none_inner()
        self.catplot_violin2()
        self.violinplot_with_inner_quart()