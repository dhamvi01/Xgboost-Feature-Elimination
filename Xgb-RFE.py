import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

#! ----------------------------------------------------------------------------------------------------------------
pd.set_option('display.max_columns',200)             
data = pd.read_csv("C:\\Users\\vijaykumar.dhameliya\\Desktop\\1000P\\9. xgboost feature elimation\\pima-indians-diabetes.csv")
print(data.head())

x = data.iloc[:,0:8]
y = data.iloc[:,8]

model = xgb.XGBClassifier()
model.fit(x,y)

fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

# ====================================================================================================================

# Recursive feature elimination

from sklearn.feature_selection import RFE

model = xgb.XGBClassifier()

rfe = RFE(model,3)
rfe = rfe.fit(x,y)

print(rfe.support_)
print(rfe.ranking_)


# --------------------------------------------------------------------------------------------------------------------

### Select From Model

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = xgb.XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc_score = accuracy_score(y_test,y_pred)
print("Accuracy: %.2f%%" % (acc_score * 100))

import numpy as np
thresh = np.sort(model.feature_importances_)

for t in thresh:
    select = SelectFromModel(model,threshold=t,prefit=True)
    select_xtrain = select.transform(x_train)

    select_model = xgb.XGBClassifier()
    select_model.fit(select_xtrain,y_train)

    select_xtest = select.transform(x_test)
    y_pred_new = select_model.predict(select_xtest)
    z = [round(value) for value in y_pred_new]
    acc_score = accuracy_score(y_test,z)

    print('Thresh=%.3f, n=%d, Accuracy: %.2f%%' % (t,select_xtest.shape[1],acc_score*100))
























