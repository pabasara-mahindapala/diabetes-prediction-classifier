########################################################################################################################
############################################## EE 7206 : Machine Learning ##############################################
############################ CLASSIFIER MODEL FOR DIABETES PREDICTION BASED ON MEDICAL DATA ############################
#################################################### Group Number 13 ###################################################
######################################### MAHINDAPALA D. P. P. - EG/2016/2916 ##########################################
##################################### THALPAWILA T. W. K. M. B. K. - EG/2016/2997 ######################################
########################################################################################################################

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Reading the dataset as a dataframe
diabetesDF = pd.read_csv('diabetes.csv')

# Information about the dataset
print("Information about the dataset:")
diabetesDF.info()
print("\n")

# Proportion of patients having diabetes out of the dataset
print("Percentage of patients having diabetes: ", (diabetesDF.Outcome.value_counts()[
                                                       1] / diabetesDF.Outcome.count()) * 100, "%")
print("\n")

# Missing values in Glucose, BloodPressure, SkinThickness, Insulin and BMI are replaced by the median
# First zeros are replaced by NaN, then NaN values are replaced by the median
diabetesDF.Glucose.replace(0, np.nan, inplace=True)
diabetesDF.Glucose.replace(np.nan, diabetesDF['Glucose'].median(), inplace=True)
diabetesDF.BloodPressure.replace(0, np.nan, inplace=True)
diabetesDF.BloodPressure.replace(np.nan, diabetesDF['BloodPressure'].median(), inplace=True)
diabetesDF.SkinThickness.replace(0, np.nan, inplace=True)
diabetesDF.SkinThickness.replace(np.nan, diabetesDF['SkinThickness'].median(), inplace=True)
diabetesDF.Insulin.replace(0, np.nan, inplace=True)
diabetesDF.Insulin.replace(np.nan, diabetesDF['Insulin'].median(), inplace=True)
diabetesDF.BMI.replace(0, np.nan, inplace=True)
diabetesDF.BMI.replace(np.nan, diabetesDF['BMI'].median(), inplace=True)

# Generate the feature-outcome distribution (Green - No Diabetes, Red - Diabetes)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
            'Age']

ROWS, COLS = 2, 4
fig, ax = plt.subplots(ROWS, COLS, figsize=(18, 8))
row, col = 0, 0
for i, feature in enumerate(features):
    if col == COLS - 1:
        row += 1
    col = i % COLS
    diabetesDF[diabetesDF.Outcome == 0][feature].hist(bins=35, color='green', alpha=0.5, ax=ax[row, col]).set_title(
        feature)
    diabetesDF[diabetesDF.Outcome == 1][feature].hist(bins=35, color='red', alpha=0.7, ax=ax[row, col])

# Add the legend
plt.legend(['No Diabetes', 'Diabetes'])
fig.subplots_adjust(hspace=0.3)

# Split the dataset into training set and testing set
dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:]

# Split the training and testing sets as labels and features
trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome', 1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome', 1))

# Standardize the training data to have a mean of 0 and a standard deviation of 1
mean1 = np.mean(trainData, axis=0)
std1 = np.std(trainData, axis=0)

trainData = (trainData - mean1) / std1

# Standardize the testing data to have a mean of 0 and a standard deviation of 1
mean2 = np.mean(testData, axis=0)
std2 = np.std(testData, axis=0)

testData = (testData - mean2) / std2

# # Using GridSearchCV to find the best hyperparameters for LR
# param_grid = {'C': [0.001, 0.01, 1, 10, 20]}
# grid = GridSearchCV(LogisticRegression(solver='liblinear'),
#                     param_grid, cv=5, scoring='f1')
# grid.fit(trainData, trainLabel)
# print(grid.best_estimator_)
print("Selected best hyperparameters for Logistic Regression: C=0.001, solver=liblinear")
print("\n")

# Logistic Regression classifier
clfLR = LogisticRegression(C=0.001, solver='liblinear')

# Train the model using the training sets
clfLR.fit(trainData, trainLabel)

# Predict the results using the testing data
predictedDataLR = clfLR.predict(testData)

# Performance measures of Logistic Regression
print("Accuracy of Logistic Regression:", metrics.accuracy_score(testLabel, predictedDataLR) * 100, "%")
print("Precision of Logistic Regression:", metrics.precision_score(testLabel, predictedDataLR) * 100, "%")
print("Recall of Logistic Regression:", metrics.recall_score(testLabel, predictedDataLR) * 100, "%")
print("\n")

# Plot the confusion matrix of Logistic Regression
metrics.plot_confusion_matrix(clfLR, testData, testLabel)

# # Using GridSearchCV to find the best hyperparameters for SVM
# param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
# grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
# grid.fit(trainData, trainLabel)
# print(grid.best_estimator_)
print("Selected best hyperparameters for SVM: kernel=sigmoid, C=10 and gamma=0.001")
print("\n")

# SVM classifier
clfSVM = svm.SVC(C=10, kernel="sigmoid", gamma=0.001)

# Train the model using the training sets
clfSVM.fit(trainData, trainLabel)

# Predict the results using the testing data
predictedDataSVM = clfSVM.predict(testData)

# Performance measures of SVM
print("Accuracy of SVM:", metrics.accuracy_score(testLabel, predictedDataSVM) * 100, "%")
print("Precision of SVM:", metrics.precision_score(testLabel, predictedDataSVM) * 100, "%")
print("Recall of SVM:", metrics.recall_score(testLabel, predictedDataSVM) * 100, "%")
print("\n")

# Plot the confusion matrix of SVM
metrics.plot_confusion_matrix(clfSVM, testData, testLabel)

# Plot the precision-Recall Curves
plt.figure()
LR_p, LR_r, _ = metrics.precision_recall_curve(testLabel, predictedDataLR)
SVM_p, SVM_r, _ = metrics.precision_recall_curve(testLabel, predictedDataSVM)
plt.plot(LR_p, LR_r, 'b', marker="+", label="Logistic Regression")
plt.plot(SVM_p, SVM_r, 'r', marker="*", label="SVM")
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title("Precision-Recall Curves")
plt.legend()

# Show the generated plots
plt.show()
