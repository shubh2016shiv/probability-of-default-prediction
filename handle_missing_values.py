import missingno as msno
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class MissingValues:
    def __init__(self,data):
        self.data = data 

    def showNullPercentage(self):
        #nullity_matrix_fig = plt.figure()
        #nullity_matrix_fig = msno.matrix(self.data.drop('Id',axis=1))
        prec_df = {'Column Name':[],'Percentage of missing Values':[]}
        for col in self.data.columns:
            prct = np.mean(self.data[col].isnull())
            prec_df['Column Name'].append(col)
            prec_df['Percentage of missing Values'].append(prct*100)

        prec_df = pd.DataFrame(prec_df)

        return prec_df

    def fillMissingValues(self,dataframe):

        dataframe['Annual Income'] = dataframe['Annual Income'].fillna(dataframe['Annual Income'].median())
        dataframe['Years in current job'] = dataframe['Years in current job'].fillna("unknown")
        dataframe['Months since last delinquent'] = dataframe['Months since last delinquent'].fillna(dataframe['Months since last delinquent'].mean())
        dataframe['Bankruptcies'] = dataframe['Bankruptcies'].fillna(dataframe['Bankruptcies'].mode()[0])
        dataframe['Credit Score'] = dataframe['Credit Score'].fillna(dataframe['Credit Score'].median())

        return dataframe