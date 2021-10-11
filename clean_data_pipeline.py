from handle_missing_values import MissingValues
from handle_outliers import Outliers

class CleanDataPipeline:
    def __init__(self,data):
        self.data = data 

    def getContinousVars(self):
        continous_variables = []
        for col in self.data.columns:
            if len(self.data[col].value_counts().to_list())>50 and col!='Id':
                continous_variables.append(col)
        
        return continous_variables

    def runCleanPipeline(self):
        missingvalues = MissingValues(self.data)
        outliers = Outliers(self.data)
        filledMissingValues_df = missingvalues.fillMissingValues(self.data)
        removed_outliers_df = outliers.removeOutliers(dataframe=filledMissingValues_df, continous_variables=self.getContinousVars())

        return removed_outliers_df
