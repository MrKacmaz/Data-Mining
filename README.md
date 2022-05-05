# Build a Prediction Model for Heart Disease

_Alperen KAÇMAZ_

_Computer Engineering Department_
_Akdeniz University_
**Antalya,Turkey**
__alperen703.akm@gmail.com__
**20170808033@ogr.akdeniz.edu.tr**

#### **Abstract — This document has been prepared for the prediction of heart attacks. Models were created and compared on the datasets used.**

## I. INTRODUCTION

This document is a template of the codes I have written. An electronic copy can be downloaded from the public repository I created on my github account or it can be viewed on [jupyter notebook viewer](https://nbviewer.org/github/MrKacmaz/Data-Mining/blob/master/main.ipynb).

## II. GETTING STARTED

### A. Preparation

We start by first downloading the CSV file shared with us. This file contains classified data on heart attacks. We read them with the relevant libraries[1] and make our dataset ready.

### B. Distortions and Arrangements

In this section, it was checked whether the data we obtained was wrong, or empty and whether there were any rows that disrupted the layout.

Before providing data to a model, it is important to clean up the data and handle nulls, outliers, and duplicate data records.

After using the necessary functions[2] (figure 2.2.1), our dataset is ready to be processed.

![fig-2.2.1](https://user-images.githubusercontent.com/57367737/166969587-e918d118-adeb-43b1-b545-7f9d896d92ec.png)

### C. Detecting Outliers using IQR (Inter Quantitative Interval)

By making an extra adjustment in the edited data sets, we look at the differences between the existing data.

In the opposite case, the data set must be rearranged and started from the beginning.

However, since the data set we have has been edited, it is sufficient to check (figure 2.3.1). In IQR, data points above the upper limit and below the lower limit are considered outliers, and the following formula is applied.

$$upper limit = (Q_3 + (1.5 \ IQR))$$

$$lower limit = (Q_1 – (1.5 \ IQR))$$

figure 2.3.1
![fig-2.3.1](https://user-images.githubusercontent.com/57367737/166969703-8c5dea97-71dc-4ab5-8b1b-5ed4a892996b.png)

### D. Correlation

In this section, the correlation of the variables specified in the data headers with each other is examined. After removing the outliers from the data, we will find the correlation between all the features. Two types of correlations will be used here. (figure 2.4.1)

- Pearson Correlation[3]
- Spearman Correlation [4]

![fig-2 4 1](https://user-images.githubusercontent.com/57367737/166969803-b8743309-393c-4102-aff2-e099b74ef90d.png)

From the heatmap, the same correlation values are repeated twice. To remove this we will mask the top half of the heatmap and show only the bottom half. The same will be done for the Spearman Correlation (figure 2.4.2 – figure 2.4.3).

![fig-2 4 2](https://user-images.githubusercontent.com/57367737/166969872-9fd419c0-8368-412f-af1a-cc4ea68af0df.png)

![fig-2 4 3](https://user-images.githubusercontent.com/57367737/166969928-19d2c3c5-19e5-4773-b3b0-3fe710bcfe20.png)

## III. CLASSIFICATION

Before applying any classification algorithms, we will separate our dataset into training data and test data.

I have used 70% of the data for training and the remaining 30% will be used for testing.

We will implement four classification algorithms;

- Logistic Regression Classifier[5]
- Decision Trees Classifier[6]
- Random Forest Classifier[7]
- K Nearest Neighbours Classifier[8]

### A. Logistic Regression Classifier

After creating the Logistic Regression Classifier, we measured the efficiency of the classifier and the efficiency of the model we created. Afterward, we measured the efficiency of our model by reducing some of the variables used, that is, reducing the correlation. Results:

- `Accuracy of logistic regression classifier >> 0.8850574712643678`
- `nAccuracy of logistic regression classifier after removing features >> 0.8620689655172413`

The accuracy of logistic regression classifier using all features is 85.05%. While the accuracy of logistic regression classifier after removing features with low correlation is 88.5%.

### B. Decision Tree Classifier

We use the method of the relevant library to create a decision tree model. And then we take the efficiency of this model as output.

By applying the correlation reduction, which we applied in our previous model, to this model, we get the efficiency as output.

- `Accuracy of Decision Trees >> 0.735632183908046`
- `Accuracy of decision Tree after removing features >> 0.7816091954022989`

The accuracy of the decision tree with all features is 73.56% while accuracy after removing low correlation features is 78.16%

### C. Random Forest Classifier

We begin to implement a random forest classifier. In this model, unlike the others, we find the score of each feature in the model, remove the low-scoring features from our model and print it out.

- `Accuracy of Random Forest Classifier >> 0.8390804597701149`
- `Feature: Age, Score: 0.09929`
- `Feature: Sex, Score: 0.03081`
- `Feature: cp, Score: 0.12283`
- `Feature: trtbps, Score: 0.08442`
- `Feature: chol, Score: 0.08079`
- `Feature: fbs, Score: 0.00906`
- `Feature: restecg, Score: 0.02134`
- `Feature: thalachh, Score: 0.13299`
- `Feature: exng, Score: 0.06644`
- `Feature: oldpeak, Score: 0.10789`
- `Feature: slp, Score: 0.05033`
- `Feature: caa, Score: 0.09979`
- `Feature: thall, Score: 0.09402`

### D. K Nearest Neighbours Classifier

We apply the K nearest neighbor classifier and output the accuracy that our model shows us.

- `Accuracy of K-Neighbours classifier >> 0.7011494252873564`

The accuracy of the model is 70.11%. Along with accuracy, we will also print the feature and its importance in the model.

Then, we will eliminate features with low importance and create another classifier and check the effect on the accuracy of the model.

As all the features have some contribution to the model, we will keep all the features.

### IV. CONCLUSION

After implementing four classification models and comparing their accuracy, we can conclude that for this dataset Logistic Regression Classifier is the appropriate model to be used.

## REFERENCES

[1] Sklearn, scipy, pandas, numpy, seaborn, matplotlib

[2] [Pandas Library](https://pandas.pydata.org/docs/user_guide/dsintro.html#dsintro)

[3] [Pearson Correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)

[4] [Spearman Correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)

[5] [LRC](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

[6] [DTC](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

[7] [RFC](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

[8] [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
