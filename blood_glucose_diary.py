import numpy as np
import pandas as pd
from numba import jit
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
        self.colors = {'BG': '#147a20',
                       'QA': '#fc7a00',
                       'BI': '#00c1d6',
                       'carbs': '#b38f00',
                       'insulin': 'k'
                       }

    def load_from_csv(self, filepath):
        header = pd.read_csv(filepath, nrows=0).columns
        self.end_time = pd.to_datetime(header[2], format=r'%d-%m-%Y %H:%M %Z')

        table = pd.read_csv(filepath, skiprows=1, index_col=2)
        table.index = pd.to_datetime(table.index, format=r'%d-%m-%Y %H:%M')
        table.index = table.index.tz_localize(self.end_time.tz)

        historic_bg = table.loc[table['Record Type'] == 0, r'Historic Glucose mmol/L']
        scan_bg = table.loc[table['Record Type'] == 1, 'Scan Glucose mmol/L']
        self.BG = pd.concat((historic_bg, scan_bg))
        self.BG = self.BG.sort_index()

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

        glucose_ax.fill_between([self.start_time, self.end_time], 4, 10,
                                linewidth=0, color=self.colors['BG'], alpha=0.2)

        x = self.BG.resample(pd.Timedelta(1, 'h')).mean()
        y = glucose_ax.get_ylim()
        glucose_ax.fill_between(x.index, y[0], y[1],
                                where=(x.index.time >= pd.to_datetime('23:00').time()) |
                                      (x.index.time <= pd.to_datetime('07:00').time()),
                                linewidth=0, color='k', alpha=0.1)
        glucose_ax.set_ylim(y)

        glucose_ax.set_xlim((start_time, end_time))
        glucose_ax.set_ylabel('Blood glucose mmol/l', color=self.colors['BG'])
        glucose_ax.tick_params(axis='y', colors=self.colors['BG'])
        insulin_ax.set_ylabel('Insulin u', color=self.colors['insulin'])
        insulin_ax.tick_params(axis='y', colors=self.colors['insulin'])
        carbs_ax.set_ylabel('Carbs g', color=self.colors['carbs'])
        carbs_ax.tick_params(axis='y', colors=self.colors['carbs'])

        return fig, glucose_ax, insulin_ax, carbs_ax


class BloodGlucosePredictor:

    def __init__(self, blood_glucose_diary):
        self.bgd = blood_glucose_diary
        self.rapid_on_board = None
        self.rapid_activity = None
        self.glucose_rate = self.differentiate(self.bgd.BG)
        self.carb_activity = None

    @staticmethod
    def differentiate(time_series, ts=15):
        resample = time_series.resample(f'{ts}Min')
        time_series = resample.mean()
        time_series.interpolate('index')
        return pd.Series(np.gradient(time_series) / ts, time_series.index)

    @staticmethod
    def resample_sum(variable, sample_rate=1):
        resample = variable.resample(f'{sample_rate}Min')
        return resample.sum()

    def get_rapid_injections(self, sample_rate=1):
        start = pd.Series([0], [self.bgd.start_time])
        end = pd.Series([0], [self.bgd.end_time])
        QA = pd.concat([start, self.bgd.QA, end])
        return self.resample_sum(QA, sample_rate)

    @staticmethod
    @jit(nopython=True)
    def insulin_integration(injections, on_board, concentration, activity, t_steps, a, b, c, d, f, g):
        for i, step_size in enumerate(t_steps):
            on_board[i+1] = on_board[i] + step_size * (-f * on_board[i] + g * injections[i])
            concentration[i+1] = concentration[i] + step_size * (-c * concentration[i] + d * on_board[i])
            activity[i+1] = activity[i] + step_size * (-a * activity[i] + b * concentration[i])

        return on_board, concentration, activity

    def predict_insulin(self, sensitivity, peak_time, unit_conversion=60*0.0005555555):
        a = self._clearance_rate_from_peak(peak_time)
        b = a * sensitivity
        c = d = f = a
        g = unit_conversion

        injections = self.get_rapid_injections()
        inj = np.array(injections)
        on_board = np.zeros_like(inj)
        concentration = np.zeros_like(inj)
        activity = np.zeros_like(inj)

        t_steps = (injections.index[1:] - injections.index[:-1]) / pd.to_timedelta(1, unit='Min')
        t_s = np.array(t_steps)

        on_board, concentration, activity = self.insulin_integration(inj, on_board, concentration,
                                                                     activity, t_s, a, b, c, d, f, g)

        self.rapid_on_board = pd.Series(on_board, injections.index)
        self.rapid_activity = pd.Series(activity, injections.index)

    def get_carbs(self, sample_rate=1):
        start = pd.Series([0], [self.bgd.start_time])
        end = pd.Series([0], [self.bgd.end_time])
        carbs = pd.concat([start, self.bgd.carbs, end])
        return self.resample_sum(carbs, sample_rate)

    def predict_carbs(self, BG_p_10g, clearance_time=3*60, delay=10):
        clearance_time = round(clearance_time)
        carbs = self.get_carbs()
        unit_activity = np.linspace(1, 0, clearance_time - delay)
        unit_activity = unit_activity / sum(unit_activity)
        carb_action_profile = np.concatenate([np.zeros(delay), (BG_p_10g/10) * unit_activity])
        self.carb_activity = pd.Series(np.convolve(carbs, carb_action_profile)[:len(carbs)], carbs.index)
        print()

    @staticmethod
    def _clearance_rate_from_peak(peak_time):
        return 2 / peak_time

    def plot(self, start_time=None, end_time=None):
        fig, glucose_ax, insulin_ax, carbs_ax = self.bgd.plot(start_time, end_time)

        rate_ax = glucose_ax.twinx()
        rate_ax.spines.right.set_position(('axes', 1.4))

        if self.glucose_rate is not None:
            rate_ax.plot(self.glucose_rate, label='Glucose Rate')

        if self.rapid_activity is not None:
            rate_ax.plot(self.rapid_activity, label='Rapid Activity')

        if self.carb_activity is not None:
            rate_ax.plot(self.carb_activity, label='Carb Release')

        net_activity = self.carb_activity + self.rapid_activity
        rate_ax.plot(net_activity, label='Predicted Rate')

        predicted_glucose = net_activity.cumsum()
        predicted_start_value = predicted_glucose[predicted_glucose.index.indexer_at_time('03:00:00')]

        daily_offset = predicted_start_value.reindex(predicted_glucose.index)
        daily_offset.iloc[0] = predicted_glucose.iloc[0]
        daily_offset = daily_offset.ffill()
        predicted_glucose = predicted_glucose + daily_offset
        glucose_ax.plot(predicted_glucose)
        plt.legend()
        plt.tight_layout()

# todo  Same for carbohydrate
# todo  Figure out resampling
# todo  Get change in blood glucose
# todo  New plot function to compare insulin activity, carb release and change in blood glucose


# input is a set of injections
# update on_board with
# https://github.com/LoopKit/Loop/issues/388

# action(k+1) = (k_now - k_previous) * (-a*action(k) + b*concentration(k)) + action(k)
# concentration(k+1) = (k_now - k_previous) * (-c*concentration(k) + d*on_board(k)) + concentration(k)
# on_board(k+1) = (k_now - k_previous) * (-f*on_board(k) + g*injection(k)) + on_board(k)

if __name__ == '__main__':
    my_bgdiary = BloodGlucoseDiary()
    my_bgdiary.load_from_csv('JoelWilliams_glucose_20-9-2022.csv')
    fig = my_bgdiary.plot(start_time=pd.to_datetime('2022-01-29'), end_time=pd.to_datetime('2022-01-31'))[0]

    bgp = BloodGlucosePredictor(my_bgdiary)
    bgp.predict_carbs(3)
    bgp.predict_insulin(-3, 60)
    bgp.plot(start_time=pd.to_datetime('2022-01-29'), end_time=pd.to_datetime('2022-01-31'))
    plt.show()


