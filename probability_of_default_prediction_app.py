import streamlit as st
from PIL import Image
import requests
import os
import pandas as pd
import numpy as np
import missingno as msno
from matplotlib import pyplot as plt
import seaborn as sns
from handle_missing_values import MissingValues
from handle_outliers import Outliers
import plotly.express as px
import plotly.graph_objects as go
from clean_data_pipeline import CleanDataPipeline
from preprocessing_pipeline import PreProcessPipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgbm
import pickle
import os
import base64

sns.set()
global data, cleanedTrainData, cleanedTestData
data = None
cleanedTrainData, cleanedTestData = None, None
isDataPreProcessed = False

st.title("Project - Prediction of Probability of Default")
st.subheader("Classification Model for Loan Default Risk Prediction")
image = Image.open("PD_logo.jpeg")
st.image(image, use_column_width=True)

st.markdown('''
Project Description
------------------------------------------

The project's business objective is to predict whether a loan borrower will default or not.
The data collected contains below columns


1. Borrower ID
2. Home Ownership
3. Annual Income
4. Years in Current Job
5. Tax Lien [A tax lien is a legal claim against the assets of an individual or business that fails to pay taxes owed to the government]
6. Number of open accounts
7. Years of Credit History
8. Maximum open credit
9. Number of Credit Problems
10. Months since last delinquent [When borrower is delinquent, they are past due on their financial obligation(s)]
11. Bankruptcies
12. Purpose
13. Term
14. Current loan amount
15. Current credit balance
16. Monthly debt
17. Credit Score
18. Credit Default [Dependent Variable]

The data contains historical record of 7500 users along with information of past Defaults or non-Defaults. 
Therefore, the dependent variable is "Credit Default". 

This project shows step by step procedure for creating a classifier model to predict the probability of loan default. 
It starts with Exploratory Data Analysis to study the nature of data including missing values and outliers.
It then subsequently use pipelines for handling missing values, handling outliers and transformations in order to clean 
and pre-process the raw data to convert it into Machine Learning usable data for training.

Three types of classification Machine Learning algorithm can be used for training - Logistic Regression, Random Forest and Light Gradient Boost Machines. 

Threshold value is calculated which decides risk of loan default.\n 
\t\t If predicted probability of default > threshold, then, high risk
\t\t If predicted probability of default < threshold, then, low risk
''')

st.markdown("-------")


# Defining Functions
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def getContinousVars(data):
    continous_variables = []
    for col in data.columns:
        if len(data[col].value_counts().to_list()) > 50 and col != 'Id':
            continous_variables.append(col)

    return continous_variables


def plot_rocauc(X, y, classifier, classifier_name):
    fig = plt.figure()

    y_prob = classifier.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    st.write('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # plot the roc curve for the model
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=classifier_name)
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best Discrimination Threshold')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    return fig, thresholds[ix]


def balanceTrainData(X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)

    return (X_res, y_res)


def performRFE(X, y):
    classifier = RandomForestClassifier()
    rfe_cv = RFECV(estimator=classifier, step=1, cv=3, scoring='recall')
    rfe_cv = rfe_cv.fit(X, y)
    optimalNoOfFeatures = rfe_cv.n_features_
    bestFeatures = X.columns[rfe_cv.support_]
    return (optimalNoOfFeatures, bestFeatures)


def performTreeBasedFeatureSelection(X, y):
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=10)
    sel.fit(X, y)
    sel.get_support()
    selected_feat = X.columns[(sel.get_support())]

    return selected_feat


def trainAndEvaluate(model, classifierName, preprocessedTrainData, preprocessedTestData, ifBalancingRequired):
    X_train = preprocessedTrainData.drop(['Credit Default'], axis=1)
    y_train = preprocessedTrainData['Credit Default']
    X_test = preprocessedTestData.drop(['Credit Default'], axis=1)
    y_test = preprocessedTestData['Credit Default']

    if ifBalancingRequired:
        X_train, y_train = balanceTrainData(X_train, y_train)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fig, best_threshold = plot_rocauc(X=X_test, y=y_test, classifier=model, classifier_name=classifierName)
    st.pyplot()

    st.write("Accuracy of the Model: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    if st.checkbox("Use optimal Discimination threshold [{:.4f}]?".format(best_threshold)):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = np.where(y_prob >= best_threshold, 1, 0)
        cm = confusion_matrix(y_test, y_pred)
        fig = sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
        st.pyplot()

        st.write("Recall of the Model: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
        st.write("F1-Score of the Model: {:.2f}%".format(f1_score(y_test, y_pred) * 100))
    else:
        cm = confusion_matrix(y_test, y_pred)
        fig = sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
        st.pyplot()

        st.write("Recall of the Model: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
        st.write("F1-Score of the Model: {:.2f}%".format(f1_score(y_test, y_pred) * 100))


# Load Data

st.header("Loan Credit Data")


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Click here to save {file_label} in your local computer</a>'
    return href


if st.button(label="Download Data"):
    try:
        id = "1Hm1xsqb_n0c6U8wK0tkfBsP4oQexInEM"
        download_file_from_google_drive(id, './train.csv')
        st.success(
            "'train.csv' successfully downloaded from google drive. Save it locally and upload it.")
        st.markdown(get_binary_file_downloader_html('./train.csv', 'training data'), unsafe_allow_html=True)

    except Exception as e:
        st.error(e.message)
upload_file = st.file_uploader("Upload the same downloaded data", type='csv')

if upload_file is not None:
    if upload_file.name == 'train.csv':
        data = pd.read_csv(upload_file)
        st.write(data)
        st.success("Credit Loan data successfully uploaded")
        st.info("Dataset Information and basic statistics:")
        st.write("Shape of Data: {}".format(data.shape))
        st.write(data.drop('Id', axis=1).describe().T)
    else:
        st.warning("Please upload 'train.csv' file only downloaded from above")
else:
    st.warning("Please upload Valid file!")

# Exploratory Data Analysis
st.write("-----")
st.header("Exploratory Data Analysis")
st.info("Select type of EDA from Sidebar in left")
st.sidebar.subheader("EDA")
eda_type = st.sidebar.selectbox("Select EDA Type", [
    'Missing Values', 'Outliers', 'Univariate Analysis', 'Multivariate Analysis'])
if eda_type == 'Missing Values' and data is not None:
    st.subheader("Missing Values")
    nullValues = MissingValues(data=data)
    prec_df = nullValues.showNullPercentage()
    fig = plt.figure()
    msno.matrix(data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    st.write(prec_df)
if eda_type == 'Outliers' and data is not None:
    st.subheader("Outliers")
    outliers = Outliers(data)
    continous_var = st.selectbox(
        "Continous Variable Column", getContinousVars(data=data))
    fig = px.box(data, y=continous_var)
    fig.add_trace(go.Box(
        y=data[continous_var],
        name="Suspected Outliers",
        boxpoints='suspectedoutliers',  # only suspected outliers
        marker=dict(
            color='rgb(8,81,156)',
            outliercolor='rgba(219, 64, 82, 0.6)',
            line=dict(
                outliercolor='rgba(219, 64, 82, 0.6)',
                outlierwidth=5)),
        line_color='rgb(8,81,156)'
    ))
    st.plotly_chart(fig)
    perc_inliers, perc_outliers = outliers.calculateOutliersInliers(continous_var)
    st.info("Column: '{}' has {}% outliers and {}% inliers".format(continous_var, perc_inliers, perc_outliers))
if eda_type == "Univariate Analysis" and data is not None:
    st.subheader("Univariate Analysis")
    col = st.selectbox('Select Column', data.drop('Id', axis=1).columns)
    categorical_vars = [col for col in data.drop(['Id', 'Number of Open Accounts', 'Tax Liens'], axis=1).columns if
                        col not in getContinousVars(data)]
    if col in categorical_vars:
        val_counts = dict(data[col].value_counts())
        val = val_counts.keys()
        counts = val_counts.values()
        fig = px.pie(names=val, values=counts)
        st.plotly_chart(fig)
    elif col in getContinousVars(data) or col in ['Number of Open Accounts', 'Tax Liens']:
        fig = px.histogram(data, x=col)
        st.plotly_chart(fig)
if eda_type == 'Multivariate Analysis' and data is not None:
    st.subheader("Multivariate Analysis")
    columns = st.multiselect("Select two or more columns", data.drop('Id', axis=1).columns)
    categorical_vars = [col for col in data.drop(['Id', 'Number of Open Accounts'], axis=1).columns if
                        col not in getContinousVars(data)]
    numerical_vars = getContinousVars(data)
    # print(numerical_vars)
    if len(columns) == 2:
        if columns[0] in numerical_vars and columns[1] in numerical_vars:
            st.write("Scatter Plot")
            fig = px.scatter(data, x=columns[0], y=columns[1])
            st.plotly_chart(fig)
        if (columns[0] in numerical_vars and columns[1] in categorical_vars) or (
                columns[1] in numerical_vars and columns[0] in categorical_vars):
            if columns[0] in categorical_vars:
                fig = px.bar(data, x=columns[0], y=columns[1])
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig)

            if columns[1] in categorical_vars:
                fig = px.bar(data, x=columns[1], y=columns[0])
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig)
        if columns[0] in categorical_vars and columns[1] in categorical_vars:
            st.error("Two or More Categorical Columns are invalid!")

    elif len(columns) > 2 and set(columns) <= set(numerical_vars):
        st.write("Correlation Matrix")
        fig = sns.heatmap(data[columns].corr(), cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
                          annot=True, annot_kws={"size": 8}, square=True)
        st.pyplot()

    else:
        if len(columns) == 1:
            st.error("Select more columns")
        elif len(columns) == 0:
            st.info("Select valid combination of columns for Multi-Variate Analysis")
        else:
            st.error("Invalid combination of Columns for EDA")

# Pre-Processing
st.write("-----")
st.header("Clean Data/Pre-Processing")

st.info("Click on each check box to initiate pre-processing pipelines step by step ")

st.subheader("Select Data Split for Training and Testing")
train_val_split = st.slider(
    label="Percentage of Training / Testing Data", min_value=50, max_value=100, value=80)
if int(train_val_split) < 70:
    st.warning("Recommended Split percentage should 80/20 or 70/30")

st.info("{}% will be cleaned/pre-processed as Training Data and {}% as Testing Data".format(train_val_split,
                                                                                            100 - train_val_split))
if data is not None:
    X = data.drop('Credit Default', axis=1)
    y = data['Credit Default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_val_split / 100, random_state=100,
                                                        stratify=y)

if st.checkbox("Initiate data clean pipeline") and data is not None:
    st.info("Starting clean pipeline..")

    # Clean Training Data
    train_dataset = pd.concat([X_train, y_train], axis=1)
    cleanPipeline = CleanDataPipeline(train_dataset)
    st.success("Training Data after handling Missing Values and Outliers")
    cleanedTrainData = cleanPipeline.runCleanPipeline()
    st.write(cleanedTrainData)
    st.write("Train Data Shape: " + str(cleanedTrainData.shape))
    # Clean Validation Data
    test_dataset = pd.concat([X_test, y_test], axis=1)
    cleanPipeline = CleanDataPipeline(test_dataset)
    st.success("Testing Data after handling Missing Values and Outliers")
    cleanedTestData = cleanPipeline.runCleanPipeline()
    st.write(cleanedTestData)
    st.write("Test Data Shape: " + str(cleanedTestData.shape))

    if st.checkbox("Initiate Pre-Processing Pipeline"):

        st.info("Starting Pre-Processing pipeline for Feature Engineering..")
        train_dataset = pd.concat([X_train, y_train], axis=1)
        cleanPipeline = CleanDataPipeline(train_dataset)
        cleanedTrainData = cleanPipeline.runCleanPipeline()
        preprocessPipeline = PreProcessPipeline(cleanedTrainData)
        preprocessedTrainData, transformFunctions = preprocessPipeline.runPreProcessTraining()
        st.success("Training Data after Feature Engineering")
        st.write(preprocessedTrainData)
        st.write("Test Data Shape: " + str(preprocessedTrainData.shape))

        test_dataset = pd.concat([X_test, y_test], axis=1)
        cleanPipeline = CleanDataPipeline(test_dataset)
        cleanedTestData = cleanPipeline.runCleanPipeline()
        preprocessPipeline = PreProcessPipeline(cleanedTestData)
        preprocessedTestData = preprocessPipeline.runPreProcessTesting(transformFunctions)
        st.success("Test Data after Feature Engineering")
        st.write(preprocessedTestData)
        st.write("Test Data Shape: " + str(preprocessedTestData.shape))

        if st.checkbox("Perform Feature Selection"):
            X_train_processed = preprocessedTrainData.drop(['Credit Default'], axis=1)
            y_train_processed = preprocessedTrainData['Credit Default']
            X_test_processed = preprocessedTestData.drop(['Credit Default'], axis=1)
            y_test_processed = preprocessedTestData['Credit Default']

            feature_selection_method = st.radio(label="Method of Feature Selection", options=(
                "Tree Based Feature Selection [Recommended]", "Recursive Feature Elimination [Time consuming]"))

            if feature_selection_method == "Recursive Feature Elimination [Time consuming]":
                st.warning(
                    "RFE takes sometime to automatically find optimal best features. Elimination is done recursively using 'Recall' KPI with cross-validation")
                optimalNoOfFeatures_rfe, bestFeatures_rfe = performRFE(X_train_processed, y_train_processed)
                # print(optimalNoOfFeatures_rfe)
                st.write('Optimal number of features :' + str(optimalNoOfFeatures_rfe))
                st.write("Best Optimal Columns:")
                st.write(bestFeatures_rfe)
                st.info("Training data after Feature Selection")
                preprocessedTrainData = pd.concat([X_train_processed[bestFeatures_rfe], y_train_processed], axis=1)
                st.write(preprocessedTrainData)
                st.info("Test data after Feature Selection")
                preprocessedTestData = pd.concat([X_test_processed[bestFeatures_rfe], y_test_processed], axis=1)
                isDataPreProcessed = True
                st.write(preprocessedTestData)


            elif feature_selection_method == "Tree Based Feature Selection [Recommended]":
                bestFeatures_treeBased = performTreeBasedFeatureSelection(X_train_processed, y_train_processed)
                st.write("Best Optimal Columns:")
                st.write(bestFeatures_treeBased)
                st.info("Training data after Feature Selection")
                preprocessedTrainData = pd.concat([X_train_processed[bestFeatures_treeBased], y_train_processed],
                                                  axis=1)
                st.write(preprocessedTrainData)
                st.info("Test data after Feature Selection")

                preprocessedTestData = pd.concat([X_test_processed[bestFeatures_treeBased], y_test_processed], axis=1)
                isDataPreProcessed = True
                st.write(preprocessedTestData)

            st.success(
                "Data Pre-Processing Completed! Now, this processed data can be trained using ML algorithms or prediction for Probability of Default can be performed directly using already trained Gradient Boosting Machine")
else:
    st.error("Download the data, Upload it and run pre-processing pipelines step by step!")

st.sidebar.subheader("Classification Models")
model = st.sidebar.selectbox("Choose Classification Model", (
    'Logistic Regression', 'Random Forest', 'Light-Gradient Boosting Machine'))

st.write("-----")
st.header("ML Model Playground: " + model)
st.info(
    "Select Machine Learning Algorithms from Sidebar to train data, perform the hyper tuning optimization and evaluate the prediction for given hyper-parameters of each Ml Model")
if isDataPreProcessed:
    if model == "Logistic Regression":
        st.subheader("Select Hyper Parameters")
        penalty = st.radio(label="Penality", options=[
            'L1 Regularization', 'L2 Regularization', 'elasticnet', 'none'])
        # C = st.slider(label="C", min_value=0.001, max_value=1.0, value=0.001)
        C = st.number_input(label="C", min_value=0.001, max_value=1.0, step=0.05, value=0.1)
        if penalty == 'L1 Regularization':
            penality_value = 'l1'
            solver = 'liblinear'
        elif penalty == 'L2 Regularization' or 'none':
            penality_value = 'l2'
            solver = 'lbfgs'
        elif penalty == 'elasticnet':
            solver = 'saga'

        st.info(
            "As Training Data is Im-Balanced, Do you want to balance the Data by sythetically generating new dataset for training?")
        ifBalancingRequired = st.checkbox(
            label="Balance Training set using SMOTE [Synthetic Minority Oversampling Technique]")

        if st.checkbox("Train and Evaluate Model"):
            logitRegrssion = LogisticRegression(penalty=penality_value, C=C, random_state=100, max_iter=1000,
                                                solver=solver)

            trainAndEvaluate(logitRegrssion, 'Logistic Regression', preprocessedTrainData, preprocessedTestData,
                             ifBalancingRequired)

    elif model == "Random Forest":
        st.subheader("Select Hyper Parameters")
        n_estimators = st.slider(label="n_estimators", min_value=100,
                                 max_value=1000, value=200, step=20)
        max_depth = st.number_input(label="max_depth", min_value=2, max_value=20, value=6)
        max_features = st.radio(label="max_features", options=['auto', 'sqrt'])
        criterion = st.radio(label="criterion", options=['gini', "entropy"])

        st.info(
            "As Training Data is Im-Balanced, Do you want to balance the Data by sythetically generating new dataset for training?")
        ifBalancingRequired = st.checkbox(
            label="Balance Training set using SMOTE [Synthetic Minority Oversampling Technique]")
        st.info("Start Training and Evaluate the model with Discrimation Threshold of 0.5")
        if st.checkbox("Train and Evaluate Model"):
            randomForestModel = RandomForestClassifier(random_state=100, n_estimators=n_estimators, criterion=criterion, \
                                                       max_depth=max_depth, max_features=max_features)

            trainAndEvaluate(randomForestModel, 'Random Forest', preprocessedTrainData, preprocessedTestData,
                             ifBalancingRequired)

    elif model == 'Light-Gradient Boosting Machine':
        learning_rate = st.number_input(label='Learning Rate', min_value=0.01, max_value=2.0, value=0.1, step=0.01)
        num_leaves = st.slider(label='Number of Leaves', min_value=30, max_value=200, value=31, step=5)
        max_depth = st.slider(label='Maximum Depth', min_value=10, max_value=200, value=10, step=10)
        lambda_l1 = st.number_input(label='L1 Regularization', min_value=0.01, max_value=2.0, value=0.1, step=0.01)
        lambda_l2 = st.number_input(label='L2 Regularization', min_value=0.01, max_value=2.0, value=0.1, step=0.01)

        if st.checkbox("Train and Evaluate Model"):
            X_train = preprocessedTrainData.drop(['Credit Default'], axis=1)
            y_train = preprocessedTrainData['Credit Default'].values
            X_test = preprocessedTestData.drop(['Credit Default'], axis=1)
            y_test = preprocessedTestData['Credit Default'].values

            X_train.columns = X_train.columns.str.replace(' ', '_')
            X_test.columns = X_test.columns.str.replace(' ', '_')

            d_train = lgbm.Dataset(X_train, label=y_train)
            v_train = lgbm.Dataset(X_test, label=y_test)

            params = {'objective': "binary", 'boosting': 'gbdt', \
                      'learning_rate': learning_rate, 'num_leaves': num_leaves, 'seed': 100, \
                      'max_depth': max_depth, 'lambda_l1': lambda_l1, 'lambda_l2': lambda_l2,
                      'metric': ['binary_logloss', 'auc', 'accuracy'], 'early_stopping_rounds': 10}

            lightGradientBoosting = lgbm.train(params=params, train_set=d_train, valid_sets=v_train)

            y_pred = lightGradientBoosting.predict(X_test)
            y_pred = np.where(y_pred > 0.5, 1, 0)
            st.write("Accuracy: " + str(accuracy_score(y_test, y_pred)))
else:
    st.warning(
        "Data not pre-processed yet for training data using any Machine Learning algorithm or for performing any predictions")

st.write("-----")
st.header("Prediction")
st.info(
    "Final Prediction is done using pre-trained Light Gradient Boosting Machine (LGBM) with Focal Loss optimized using HyperOpt. Click on below to see code for training LGBM")
st.write("[Google Colab Link](https://colab.research.google.com/drive/1pYorF4HeI8OVPKLql8rYdYH_NJqV9P52?usp=sharing)")
home_ownership = st.selectbox(label='Home Ownership', options=['Own Home', 'Home Mortgage', 'Rent', 'Have Mortgage'])
annual_income = st.number_input(label="Annual Income", min_value=164597.0, max_value=10149344.0, step=10000.0,
                                value=1473887.0)
years_in_current_job = st.selectbox(label="Years in current job",
                                    options=['10+ years', '9 years', '8 years', '7 years', '6 years', '5 years',
                                             '4 years', '3 years', '2 years', '1 year', '< 1 year', 'unknown'])

tax_liens = st.number_input(label="Tax Liens", min_value=0, max_value=7, step=1, value=1)
number_of_openAccounts = st.number_input(label="Number of Open Accounts", min_value=1, max_value=13, step=1)
years_of_credit_history = st.number_input(label='Years of Credit History', min_value=4, max_value=60, step=1, value=15)
maximum_open_credit = st.number_input("Maximum Open Credit", min_value=0.0, max_value=1304726170.0, step=10000.0,
                                      value=329912.0)
number_of_credit_problems = st.number_input("Number of Credit Problems", min_value=0, max_value=0, step=1)
bankruptcies = st.number_input("Bankruptcies", min_value=0, max_value=4, step=1, value=0)
purpose = st.selectbox("Purpose", options=['debt consolidation', 'other', 'home improvements', 'take a trip',
                                           'buy a car', 'small business', 'business loan', 'wedding',
                                           'educational expenses', 'buy house', 'medical bills', 'moving',
                                           'major purchase', 'vacation', 'renewable energy'])
months_since_last_delinquent = st.number_input(label='Months since last delinquent', min_value=0, max_value=41, step=5)
term = st.selectbox("Term", options=['Short Term', 'Long Term'])
current_loan_amount = st.number_input(label='Current Loan Amount', min_value=10000, max_value=99999999, step=10000,
                                      value=291918)
current_credit_balance = st.number_input(label='Current Credit Balance', min_value=0, max_value=6500000, step=100000,
                                         value=243333)
monthly_debt = st.number_input(label='Monthly Debt', min_value=0, max_value=150000, step=1000, value=17195)
credit_score = st.number_input(label='Credit Score', min_value=500, max_value=900, step=10, value=700)

if st.button("Predict"):
    if not isDataPreProcessed:
        st.error(
            "Prediction Denied! Complete the pre-processing pipelines to enable the prediction pipeline and then click 'Predict'")
    else:
        valuesDict = {'Home Ownership': [home_ownership], 'Annual Income': [annual_income],
                      'Years in current job': [years_in_current_job],
                      'Tax Liens': [tax_liens], 'Number of Open Accounts': [number_of_openAccounts],
                      'Years of Credit History': [years_of_credit_history],
                      'Maximum Open Credit': [maximum_open_credit],
                      'Number of Credit Problems': [number_of_credit_problems],
                      'Months since last delinquent': [months_since_last_delinquent], 'Bankruptcies': [bankruptcies],
                      'Purpose': [purpose], 'Term': [term],
                      'Current Loan Amount': [current_loan_amount], 'Current Credit Balance': [current_credit_balance],
                      'Monthly Debt': [monthly_debt],
                      'Credit Score': [credit_score]}

        important_columns = ['Annual Income', 'Number of Open Accounts', 'Years of Credit History',
                             'Maximum Open Credit', \
                             'Current Loan Amount', 'Current Credit Balance', 'Monthly Debt', 'Credit Score']

        model = pickle.load(open('./lgbm_model.sav', 'rb'))

        values_df = pd.DataFrame(valuesDict)

        if transformFunctions is not None:
            preprocessPipeline = PreProcessPipeline(values_df)
            valuesPreProcessed_df = preprocessPipeline.runPreProcessTesting(transformFunctions=transformFunctions)
            valuesPreProcessed_df = valuesPreProcessed_df[important_columns]
            prediction = model.predict(valuesPreProcessed_df)
            threshold_cutOff = 0.468883
            probability = 1. / (1. + np.exp(-prediction))
            st.write("Probability of Default: {:.4f}".format(probability[0]))
            st.write("Threshold Cut off: {}".format(threshold_cutOff))
            if probability > threshold_cutOff:
                st.write("Result: {}".format("Likely to be defaulted"))
            else:
                st.write("Result: {}".format("Likely to be NOT defaulted"))

        else:
            st.error('Prediction not allowed till pre-processing is done. Please complete the pre-processing first.')
