import unittest
import pandas as pd
from blood_glucose_diary import BloodGlucoseDiary


class TestBloodGlucoseDiary(unittest.TestCase):

    def test_load_from_csv(self):
        bgd = BloodGlucoseDiary()
        bgd.load_from_csv('test_data.csv')

        self.assertEqual(bgd.end_time, pd.to_datetime('2022-2-12 16:39 UTC'))
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


if __name__ == '__main__':
    unittest.main()
