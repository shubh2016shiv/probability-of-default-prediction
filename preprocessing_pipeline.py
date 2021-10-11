from sklearn.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder, CountFrequencyEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd

class PreProcessPipeline:
    def __init__(self,data):
        self.data = data
    
    def getContinousVars(self,df):
        continous_variables = []
        for col in df.columns:
            if len(df[col].value_counts().to_list())>50 and col!='Id':
                continous_variables.append(col)
        
        return continous_variables



    def runPreProcessTraining(self):
        transformFunctions = {}

        if 'Id' in self.data.columns:
            self.data.drop(['Id'],axis=1,inplace=True)
        
        years_in_current_job_map = {'10+ years': 10,'9 years': 9,'8 years': 8,'7 years':7,'6 years':6,
                            '5 years': 5, '4 years': 4, '3 years':3, '2 years': 2,'1 year': 1, '< 1 year':0 }

        transformFunctions['years_in_current_job_map'] = years_in_current_job_map

        self.data['Years in current job'] = self.data['Years in current job'].map(years_in_current_job_map)
        median_years_in_current_job = self.data['Years in current job'].median(skipna=True)
        transformFunctions['median_years_in_current_job'] = median_years_in_current_job
        self.data['Years in current job'] = self.data['Years in current job'].agg(lambda x : x.fillna(median_years_in_current_job))  

        term_encoder = OneHotEncoder(variables=['Term'],drop_last=True)
        home_ownership_encoder = OneHotEncoder(variables=['Home Ownership'],drop_last=True)
        purpose_encoder = CountFrequencyEncoder(encoding_method='frequency',variables=['Purpose'])

        pipe = Pipeline([('term_encoder',term_encoder),('home_ownership_encoder',home_ownership_encoder),\
                ('purpose_encoder',purpose_encoder)])

        scaler = StandardScaler()

        X = self.data.drop("Credit Default",axis=1)
        y = self.data["Credit Default"]


        pipe.fit(X, y)

        encoded_df = pipe.transform(X)

        transformFunctions['encodingPipeline'] = pipe

        continous_vars = self.getContinousVars(encoded_df)

        transformFunctions['continous_vars'] = continous_vars

        scaler.fit(encoded_df[continous_vars])

        encoded_df[continous_vars] = scaler.transform(encoded_df[continous_vars])

        transformFunctions['scaler'] = scaler

        processed_df = pd.concat([encoded_df,y],axis=1)
        
        return (processed_df,transformFunctions)


    def runPreProcessTesting(self,transformFunctions):
        if 'Id' in self.data.columns:
            self.data.drop(['Id'],axis=1,inplace=True)

        years_in_current_job_map = transformFunctions['years_in_current_job_map']
        median_years_in_current_job = transformFunctions['median_years_in_current_job'] 
        pipe = transformFunctions['encodingPipeline']
        scaler = transformFunctions['scaler']
        continous_vars = transformFunctions['continous_vars']

        self.data['Years in current job'] = self.data['Years in current job'].map(years_in_current_job_map)
        self.data['Years in current job'] = self.data['Years in current job'].agg(lambda x : x.fillna(median_years_in_current_job))
        
        if 'Credit Default' in self.data:
            X = self.data.drop("Credit Default",axis=1)
            y = self.data["Credit Default"]
            encoded_df = pipe.transform(X)
            #print(encoded_df.dtypes)
            #print(continous_vars)
            encoded_df[continous_vars] = scaler.transform(encoded_df[continous_vars])

            processed_df = pd.concat([encoded_df,y],axis=1)
            return processed_df

        else:
            X = self.data
            encoded_df = pipe.transform(X)
            #print(encoded_df.dtypes)
            #print(continous_vars)
            encoded_df[continous_vars] = scaler.transform(encoded_df[continous_vars])
            return encoded_df
        