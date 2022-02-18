import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class BloodGlucoseDiary:

    def __init__(self):
        self.BG = pd.Series(dtype='float64')
        self.QA = pd.Series(dtype='float64')
        self.BI = pd.Series(dtype='float64')
        self.carbs = pd.Series(dtype='float64')
        self.notes = pd.Series(dtype='string')
        self.start_time = None
        self.end_time = None
        self.colors = {'BG':'#e03c2d',
                      'QA':'#e88e33',
                      'BI':'#339de8',
                      'carbs':'#e3d27d',
                      'insulin':'k'
                       }


    def load_from_csv(self, filepath):
        header = pd.read_csv(filepath, nrows=0).columns
        self.end_time = pd.to_datetime(header[2], format=r'%d-%m-%Y %H:%M %Z')

        table = pd.read_csv(filepath, skiprows=1, index_col=2)
        table.index = pd.to_datetime(table.index, format=r'%d-%m-%Y %H:%M')

        historic_bg = table.loc[table['Record Type'] == 0, r'Historic Glucose mmol/L']
        scan_bg = table.loc[table['Record Type'] == 1, 'Scan Glucose mmol/L']
        self.BG = pd.concat((historic_bg, scan_bg))

        self.QA = table.loc[pd.notna(table['Rapid-Acting Insulin (units)']), r'Rapid-Acting Insulin (units)']
        QA_non_numeric = table['Non-numeric Rapid-Acting Insulin'].dropna()
        self.QA = pd.concat((self.QA, QA_non_numeric.replace(1, np.nan)))

        self.BI = table.loc[pd.notna(table['Long-Acting Insulin (units)']), r'Long-Acting Insulin (units)']
        BI_non_numeric = table['Non-numeric Long-Acting Insulin'].dropna()
        self.BI = pd.concat((self.BI, BI_non_numeric.replace(1, np.nan)))

        self.carbs = table.loc[pd.notna(table['Carbohydrates (grams)']), r'Carbohydrates (grams)']
        carbs_non_numeric = table['Non-numeric Food'].dropna()
        self.carbs = pd.concat((self.carbs, carbs_non_numeric.replace(1, np.nan)))

        self.notes = table['Notes'].dropna()

        self.start_time = np.min(np.concatenate((self.BG.index,
                                                self.QA.index,
                                                self.BI.index,
                                                self.carbs.index,
                                                self.notes.index)))

    def plot(self, start_time=None, end_time=None):
        fig, glucose_ax = plt.subplots()
        fig.subplots_adjust(right=0.75)
        insulin_ax = glucose_ax.twinx()
        carbs_ax = glucose_ax.twinx()
        carbs_ax.spines.right.set_position(('axes', 1.2))

        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time

        self.BG.plot(ax=glucose_ax, color=self.colors['BG'])
        insulin_ax.scatter(self.QA.index, self.QA.fillna(-1), color=self.colors['QA'])
        insulin_ax.scatter(self.BI.index, self.BI.fillna(-1), color=self.colors['BI'])
        carbs_ax.scatter(self.carbs.index, self.carbs.fillna(-1), color=self.colors['carbs'])

        glucose_ax.set_xlim((start_time, end_time))
        glucose_ax.set_ylabel('Blood glucose mmol/l', color=self.colors['BG'])
        glucose_ax.tick_params(axis='y', colors=self.colors['BG'])
        insulin_ax.set_ylabel('Insulin u', color=self.colors['insulin'])
        insulin_ax.tick_params(axis='y', colors=self.colors['insulin'])
        carbs_ax.set_ylabel('Carbs g', color=self.colors['carbs'])
        carbs_ax.tick_params(axis='y', colors=self.colors['carbs'])

        return fig, glucose_ax, insulin_ax, carbs_ax


if __name__ == '__main__':
    my_bgdiary = BloodGlucoseDiary()
    my_bgdiary.load_from_csv('JoelWilliams_glucose_1-2-2022.csv')
    my_bgdiary.plot(start_time=pd.to_datetime('2021-07-06'), end_time=pd.to_datetime('2021-07-08'))
    plt.show()
