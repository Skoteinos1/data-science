import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('MedData.csv')
df.head()

# Scatterplot to see patterns in our observations
sns.scatterplot(data=df, x='Harmonexin', y='Outcome')
plt.show()

# Determine your independent and dependent variables
X = df['Harmonexin'].values
y = df['Outcome'].values  # Disease outcome: 1=yes, 0=no

# Making sure the input data is in ap appropriately shaped object
X.shape

X = X.reshape(-1,1)
X.shape

# Fitting the Logistic Regression model
clf = LogisticRegression().fit(X, y)

# Make a prediction
y_pred1 = clf.predict([[120]])
print("Outcome for Harmonexin level of 120: ", y_pred1)

# Make a prediction
y_pred2 = clf.predict([[220]])
print("Outcome for Harmonexin level of 120: ", y_pred2)

# Check it by doing math yourself
clf.coef_, clf.intercept_  # (array[[0.03]]), (array[[-5.03]])

# Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, tesst_size=0.25, random_state=0)

# fit a model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# make predictions on a test data
y_pred = log_reg.predict(X_test).reshape(-1, 1)
y_pred[:10]


# Measuring a model's performance
# Use score method to get accurqacy of model
score = log_reg.score(X_test, y_test)
print('Accuracy: {}'.format(score))  # Accuracy above 75% should be fine

from sklearn.metrics import confusion_matrix

classes = ['No', 'Yes']
conf_mat = confusion_matrix(y_test, y_pred)
cm_df = pd.Dataframe(conf_mat, columns=classes, index=classes)
cm_df

# just checking that the labels are accurate
# if they are, then the row totals should equal the respective sums below
(y_test == 1).sum(), (y_test == 0).sum()
# (62, 130)

# Clasification Rate (Accuracy) alson called the Hit rate
classification_rate = (conf_mat[0, 0] + conf_mat[1,1])/ conf_mat.sum()
print(f'{classification_rate:.2%}')

# Misclassification rate, the ones that model didn't get right
misclassification_rate = (conf_mat[0,1] + conf_mat[1,0])/conf_mat.sum()
print(f'{misclassification_rate:.2%}')


