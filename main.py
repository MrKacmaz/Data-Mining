from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)

data1 = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")
data1 = pd.DataFrame(data1)
data1.info()
print('Number of records and features in the dataset >> ' + str(data1.shape))

duplicate_rows = data1[data1.duplicated()]
print("Number of duplicate rows >> ", duplicate_rows.shape)

data1 = data1.drop_duplicates()
duplicate_rows = data1[data1.duplicated()]
print("Number of duplicate rows >> ", duplicate_rows.shape)

print("Null values >> \n" + str(data1.isnull() .sum()))


print(data1.shape)



# No Outliers observed in 'age'
sns.boxplot(x=data1['age'])

# No outliers observed in sex data
sns.boxplot(x=data1['sex'])

# No outliers in 'cp'
sns.boxplot(x=data1['cp'])

# Some outliers are observed in 'trtbps'. They will be removed later
sns.boxplot(x=data1['trtbps'])

# Some outliers are observed in 'chol'. They will be removed later
sns.boxplot(x=data1['chol'])
sns.boxplot(x=data1['fbs'])
sns.boxplot(x=data1['restecg'])

# Outliers present in thalachh
sns.boxplot(x=data1['thalachh'])
sns.boxplot(x=data1['exng'])

# Outliers are present in 'OldPeak'
sns.boxplot(x=data1['oldpeak'])
sns.boxplot(x=data1['slp'])

# Outliers are present in 'caa'
sns.boxplot(x=data1['caa'])
sns.boxplot(x=data1['thall'])


# Find the InterQuartile Range
Q1 = data1.quantile(0.25)
Q3 = data1.quantile(0.75)
IQR = Q3-Q1
print('\t InterQuartile Range \t')
print(IQR)
# Remove the outliers using IQR
data2 = data1[~((data1 < (Q1-1.5*IQR)) | (data1 > (Q3+1.5*IQR))).any(axis=1)]
data2.shape


# Removing outliers using Z-score
z = np.abs(stats.zscore(data1))
data3 = data1[(z < 3).all(axis=1)]
data3.shape


# Finding the correlation between variables
pearsonCorr = data3.corr(method='pearson')
spearmanCorr = data3.corr(method='spearman')
fig = plt.subplots(figsize=(14, 8))
sns.heatmap(pearsonCorr, vmin=-1, vmax=1,
            cmap="Greens", annot=True, linewidth=0.1)
plt.title("Pearson Correlation")


# Generating mask for upper triangle
maskP = np.triu(np.ones_like(pearsonCorr, dtype=bool))

# Adjust mask and correlation
maskP = maskP[1:, :-1]
pCorr = pearsonCorr.iloc[1:, :-1].copy()

# Setting up a diverging palette
cmap = sns.diverging_palette(0, 200, 150, 50, as_cmap=True)
fig = plt.subplots(figsize=(14, 8))
sns.heatmap(pCorr, vmin=-1, vmax=1, cmap=cmap,
            annot=True, linewidth=0.3, mask=maskP)
plt.title("Pearson Correlation")



fig = plt.subplots(figsize=(14, 8))
sns.heatmap(spearmanCorr, vmin=-1, vmax=1,
            cmap="Blues", annot=True, linewidth=0.1)
plt.title("Spearman Correlation")


# From this we observe that the minimum correlation between output and other features in fbs,trtbps and chol
x = data3.drop("output", axis=1)
y = data3["output"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# Building classification models
names = ['Age', 'Sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
         'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
#   ****************Logistic Regression*****************
logReg = LogisticRegression(random_state=0, solver='liblinear')
logReg.fit(x_train, y_train)
# Check accuracy of Logistic Regression
y_pred_logReg = logReg.predict(x_test)
# Model Accuracy
print("Accuracy of logistic regression classifier >> ",
      metrics.accuracy_score(y_test, y_pred_logReg))
# Removing the features with low correlation and checking effect on accuracy of model
x_train1 = x_train.drop("fbs", axis=1)
x_train1 = x_train1.drop("trtbps", axis=1)
x_train1 = x_train1.drop("chol", axis=1)
x_train1 = x_train1.drop("restecg", axis=1)
x_test1 = x_test.drop("fbs", axis=1)
x_test1 = x_test1.drop("trtbps", axis=1)
x_test1 = x_test1.drop("chol", axis=1)
x_test1 = x_test1.drop("restecg", axis=1)
logReg1 = LogisticRegression(
    random_state=0, solver='liblinear').fit(x_train1, y_train)
y_pred_logReg1 = logReg1.predict(x_test1)
print("nAccuracy of logistic regression classifier after removing features >> ",
      metrics.accuracy_score(y_test, y_pred_logReg1))


decTree = DecisionTreeClassifier(max_depth=6, random_state=0)
decTree.fit(x_train, y_train)
y_pred_decTree = decTree.predict(x_test)
print("Accuracy of Decision Trees >> ",
      metrics.accuracy_score(y_test, y_pred_decTree))

# Remove features which have low correlation with output (fbs, trtbps, chol)
x_train_dt = x_train.drop("fbs", axis=1)
x_train_dt = x_train_dt.drop("trtbps", axis=1)
x_train_dt = x_train_dt.drop("chol", axis=1)
x_train_dt = x_train_dt.drop("age", axis=1)
x_train_dt = x_train_dt.drop("sex", axis=1)
x_test_dt = x_test.drop("fbs", axis=1)
x_test_dt = x_test_dt.drop("trtbps", axis=1)
x_test_dt = x_test_dt.drop("chol", axis=1)
x_test_dt = x_test_dt.drop("age", axis=1)
x_test_dt = x_test_dt.drop("sex", axis=1)

decTree1 = DecisionTreeClassifier(max_depth=6, random_state=0)
decTree1.fit(x_train_dt, y_train)
y_pred_dt1 = decTree1.predict(x_test_dt)
print("Accuracy of decision Tree after removing features >> ",
      metrics.accuracy_score(y_test, y_pred_dt1))


rf = RandomForestClassifier(n_estimators=500)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print("Accuracy of Random Forest Classifier >> ",
      metrics.accuracy_score(y_test, y_pred_rf))

# Find the score of each feature in model and drop the features with low scores
f_imp = rf.feature_importances_
for i, v in enumerate(f_imp):
    print('Feature: %s, Score: %.5f' % (names[i], v))


knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
y_pred_knc = knc.predict(x_test)
print("Accuracy of K-Neighbours classifier >> ",
      metrics.accuracy_score(y_test, y_pred_knc))


print("Logistic Regression Classifier >> ",
      metrics.accuracy_score(y_test, y_pred_logReg1))
print("Decision Tree >> ", metrics.accuracy_score(y_test, y_pred_dt1))
print("Random Forest Classifier >> ", metrics.accuracy_score(y_test, y_pred_rf))
print("K Neighbours Classifier >> ", metrics.accuracy_score(y_test, y_pred_knc))
