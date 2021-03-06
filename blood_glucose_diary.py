import pandas as pd


class BloodGlucoseDiary:

    def __init__(self):
        self.BG = pd.Series(dtype='float64')
        self.QA = pd.Series(dtype='float64')
        self.BI = pd.Series(dtype='float64')
        self.carbs = pd.Series(dtype='float64')
        self.notes = pd.Series(dtype='string')
        self.start_time = None
        self.end_time = None

    def load_from_csv(self, filepath):
        header = pd.read_csv(filepath, nrows=0).columns
        self.end_time = pd.to_datetime(header[2], format=r'%d-%m-%Y %H:%M %Z')

        table = pd.read_csv(filepath, skiprows=1, index_col=2)
        table.index = pd.to_datetime(table.index, format=r'%d/%m/%Y %H:%M')

        historic_bg = table.loc[table['Record Type'] == 0, r'Historic Glucose mmol/L']
        scan_bg = table.loc[table['Record Type'] == 1, 'Scan Glucose mmol/L']
        self.BG = pd.concat((historic_bg, scan_bg))

        self.QA = table.loc[pd.notna(table['Rapid-Acting Insulin (units)']), r'Rapid-Acting Insulin (units)']
        QA_non_numeric = table['Non-numeric Rapid-Acting Insulin'].dropna()
        self.QA = pd.concat((self.QA, QA_non_numeric.replace(1, pd.NA)))

        self.BI = table.loc[pd.notna(table['Long-Acting Insulin (units)']), r'Long-Acting Insulin (units)']
        BI_non_numeric = table['Non-numeric Long-Acting Insulin'].dropna()
        self.BI = pd.concat((self.BI, BI_non_numeric.replace(1, pd.NA)))

        self.carbs = table.loc[pd.notna(table['Carbohydrates (grams)']), r'Carbohydrates (grams)']
        carbs_non_numeric = table['Non-numeric Food'].dropna()
        self.carbs = pd.concat((self.carbs, carbs_non_numeric.replace(1, pd.NA)))

        self.notes = table['Notes'].dropna()

        print()

if __name__ == '__main__':
    my_bgdiary = BloodGlucoseDiary()
    my_bgdiary.load_from_csv('test_data.csv')

