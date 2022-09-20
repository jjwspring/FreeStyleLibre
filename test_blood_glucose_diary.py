import unittest
import pandas as pd
import numpy as np
import timeit
from blood_glucose_diary import BloodGlucoseDiary, BloodGlucosePredictor
from matplotlib import pyplot as plt

from scipy import integrate
import pandas as pd
import numpy as np

def integrate_method(self, how='trapz', unit='s'):
    '''Numerically integrate the time series.

    @param how: the method to use (trapz by default)
    @return

    Available methods:
     * trapz - trapezoidal
     * cumtrapz - cumulative trapezoidal
     * simps - Simpson's rule
     * romb - Romberger's rule

    See http://docs.scipy.org/doc/scipy/reference/integrate.html for the method details.
    or the source code
    https://github.com/scipy/scipy/blob/master/scipy/integrate/quadrature.py
    '''
    available_rules = set(['trapz', 'cumtrapz', 'simps', 'romb'])
    if how in available_rules:
        rule = integrate.__getattribute__(how)
    else:
        print('Unsupported integration rule: %s' % (how))
        print('Expecting one of these sample-based integration rules: %s' % (str(list(available_rules))))
        raise AttributeError

    result = rule(self.values, self.index.astype(np.int64) / 10 ** 9)
    # result = rule(self.values)
    return result

pd.Series.integrate = integrate_method


class TestBloodGlucoseDiary(unittest.TestCase):

    def test_load_from_csv(self):
        bgd = BloodGlucoseDiary()
        bgd.load_from_csv('test_data.csv')

        self.assertEqual(bgd.end_time, pd.to_datetime('2022-2-01 17:23 UTC'))
        self.assertEqual(bgd.BG[pd.to_datetime('2021-07-06 23:19')], 4.0)
        self.assertTrue(pd.api.types.is_float_dtype(bgd.BG))
        self.assertEqual(len(bgd.BG), 10)

        self.assertEqual(bgd.QA[pd.to_datetime('2021-07-07 07:30')], 1)
        self.assertTrue(pd.api.types.is_float_dtype(bgd.QA))
        self.assertTrue(pd.isna(bgd.QA[pd.to_datetime('2021-07-13 09:01')]))
        self.assertEqual(len(bgd.QA), 7)

        self.assertEqual(bgd.BI[pd.to_datetime('2021-07-07 08:54')], 15.5)
        self.assertTrue(pd.isna(bgd.BI[pd.to_datetime('2021-09-22 22:21')]))
        self.assertTrue(pd.api.types.is_float_dtype(bgd.BI))
        self.assertEqual(len(bgd.BI), 3)

        self.assertEqual(bgd.carbs[pd.to_datetime('2021-07-07 11:39')], 25)
        self.assertTrue(pd.isna(bgd.carbs[pd.to_datetime('2021-07-07 11:03')]))
        self.assertTrue(pd.api.types.is_float_dtype(bgd.carbs))
        self.assertEqual(len(bgd.carbs), 6)

        self.assertEqual(bgd.notes[pd.to_datetime('2022-01-28 21:33')], 'Exercise')
        self.assertEqual(len(bgd.notes), 2)


class TestBloodGlucosePredictor(unittest.TestCase):

    def setUp(self) -> None:
        self.bgd = BloodGlucoseDiary()
        self.bgd.load_from_csv('test_data.csv')
        self.bgp = BloodGlucosePredictor(self.bgd)

    def test_get_rapid_injections(self):
        insulin = pd.Series([1, 2, np.nan, 4],
                            pd.to_datetime(['2022/01/01 00:01', '2022/01/01 00:01',
                                            '2022/01/01 00:02', '2022/01/01 00:04']))
        self.bgd.QA = insulin
        self.bgd.start_time = pd.to_datetime('2022/01/01 00:00')
        self.bgd.end_time = pd.to_datetime('2022/01/01 00:05')

        injections = self.bgp.get_rapid_injections()
        self.assertEqual(injections['2022/01/01 00:00'], 0)
        self.assertEqual(injections['2022/01/01 00:01'], 3)
        self.assertEqual(injections['2022/01/01 00:02'], 0)
        self.assertEqual(injections['2022/01/01 00:03'], 0)
        self.assertEqual(injections['2022/01/01 00:04'], 4)
        self.assertEqual(injections['2022/01/01 00:05'], 0)

    def test_predict_insulin(self):
        self.bgd = BloodGlucoseDiary()
        self.bgd.load_from_csv('test_insulin_data.csv')
        self.bgp = BloodGlucosePredictor(self.bgd)
        self.bgp.predict_insulin(1, 1*60)

        self.assertEqual(self.bgp.rapid_activity.idxmax(), pd.to_datetime('2021-07-06 01:00:00 UTC'))
        integral = self.bgp.rapid_activity.integrate('trapz', unit='s') / 60
        self.assertAlmostEqual(integral, 1, 4)
        self.bgp.predict_insulin(2, 1*60)
        self.assertEqual(self.bgp.rapid_activity.idxmax(), pd.to_datetime('2021-07-06 01:00:00 UTC'))
        integral = self.bgp.rapid_activity.integrate('trapz', unit='s') / 60
        self.assertAlmostEqual(integral, 2, 4)
        integral = integrate.cumtrapz(self.bgp.rapid_activity, self.bgp.rapid_activity.index) / pd.to_timedelta('1Min')
        integral = pd.Series(np.concatenate([[0], integral]), self.bgp.rapid_activity.index)
        plt.figure(figsize=[10, 6])
        plt.plot(self.bgp.rapid_on_board, label='Insulin on board')
        plt.plot(self.bgp.rapid_activity, label='Insulin activity')
        plt.legend(loc='lower left', bbox_to_anchor=[0, 1.01])
        ax = plt.twinx(plt.gca())
        ax.plot((integral[-1]-integral), label='Effect on BG', color='g')
        plt.legend(loc='lower right', bbox_to_anchor=[1, 1.01])
        plt.tight_layout()
        plt.show()

    def test_differentiate(self):
        time_series = pd.Series([1, 2, 4, 7, 11], index=pd.date_range('2022-01-01 00:00:00 UTC', periods=5, freq='5Min'))
        bgd = BloodGlucoseDiary()
        bgd.BG = time_series
        bgp = BloodGlucosePredictor(bgd)
        bgp.differentiate(time_series)


if __name__ == '__main__':
    unittest.main()
