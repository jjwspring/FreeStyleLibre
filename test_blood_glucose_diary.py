import unittest
import pandas
import pandas as pd

from blood_glucose_diary import BloodGlucoseDiary


class TestBloodGlucoseDiary(unittest.TestCase):

    def test_load_from_csv(self):
        diary = BloodGlucoseDiary()
        diary.load_from_csv('test_data.csv')

        self.assertEqual(diary.end_time, pandas.to_datetime('2022-2-12 16:39 UTC'))
        self.assertEqual(diary.BG[pd.to_datetime('2021-07-06 23:19')], 4.0)
        self.assertEqual(len(diary.BG), 10)

        self.assertEqual(diary.QA[pd.to_datetime('2021-07-07 07:30')], 1)
        self.assertTrue(pd.isna(diary.QA[pd.to_datetime('2021-07-13 09:01')]))
        self.assertEqual(len(diary.QA), 7)

        self.assertEqual(diary.BI[pd.to_datetime('2021-07-07 08:54')], 15.5)
        self.assertTrue(pd.isna(diary.BI[pd.to_datetime('2021-09-22 22:21')]))
        self.assertEqual(len(diary.BI), 3)

        self.assertEqual(diary.notes[pd.to_datetime('2022-01-28 21:33')], 'Exercise')
        self.assertEqual(len(diary.notes), 2)


if __name__ == '__main__':
    unittest.main()
