







X.shape

X = X.reshape(-1,1)
X.shape

clf = LogisticRegression().fit(X, y)

# Make a prediction
y_pred1 = clf.predict([[120]])
print("Outcome for Harmonexin level of 120: ", y_pred1)

# Make a prediction
y_pred2 = clf.predict([[220]])
print("Outcome for Harmonexin level of 120: ", y_pred2)

# Check it by doing math yourself
clf.coef_, clf.intercept_

(array[[0.03]]), (array[[-5.03]])


# Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, tesst_size=0.25, random_state=0)

# fit a model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# make predictions on a test data
y_pred = log_reg.predict(X_test).reshape(-1, 1)
y_pred[:10]