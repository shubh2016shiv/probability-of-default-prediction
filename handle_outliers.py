import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px

class Outliers:
    def __init__(self,data):
        self.data = data


    def plotBoxPlot(self,continous_var):
        boxplot_fig = px.box(self.data, y=continous_var)
        return boxplot_fig

    def calculateOutliersInliers(self,continous_var):
        q1 = self.data[continous_var].quantile(0.25)
        q3 = self.data[continous_var].quantile(0.75)
        IQR = q3 - q1
        outliers = self.data[continous_var][(self.data[continous_var]<(q1-1.5*IQR)) | (self.data[continous_var]>(q3+1.5*IQR))]
        inliers = self.data[continous_var][~((self.data[continous_var]<(q1-1.5*IQR)) | (self.data[continous_var]>(q3+1.5*IQR)))]
        return (round(len(outliers)/len(self.data)*100,3),round(len(inliers)/len(self.data)*100,3))     

    def removeOutliers(self,dataframe,continous_variables):
        dataframe['Credit Score'].where(dataframe['Credit Score']<900,900,inplace=True)

        for column in continous_variables:
            if column is not 'Credit Score':
                _,percentage_outliers = self.calculateOutliersInliers(column)
                if percentage_outliers<10 or column=='Current Loan Amount':
                #print("Removing outliers from Column: {}".format(column))
                    q1 = dataframe[column].quantile(0.25)
                    q3 = dataframe[column].quantile(0.75)
                    IQR = q3 - q1
                    outliers = dataframe[column][(dataframe[column]<(q1-1.5*IQR)) | (dataframe[column]>(q3+1.5*IQR))]
                    ids = outliers.index
                    dataframe.drop(ids,inplace=True)

            

        return dataframe 
