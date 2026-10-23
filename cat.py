import pandas as pd 
import numpy as np 
import seaborn as sn
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             recall_score, precision_score, roc_auc_score)
from catboost import Pool, CatBoostClassifier

SEED = 42 
np.random.seed(SEED)
df = pd.read_csv("realistic_credit_scoring_dataset-1.csv")


X = df.drop(columns = "Loan_Status")
y = df["Loan_Status"].map({"Default": 1, "Paid": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

cat_test = Pool(X_test, y_test) 

CatBoost = CatBoostClassifier(iterations=5,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)

CatBoost.fit(X_train, y_train)
y_pred = CatBoost.predict(cat_test)
y_pred_proba = CatBoost.predict_proba(cat_test)[:,1]

acc_cat = accuracy_score(y_test, y_pred)
prec_cat = precision_score(y_test, y_pred)
rec_cat = recall_score(y_test, y_pred)
f1_cat = f1_score(y_test, y_pred)
roc_cat = roc_auc_score(y_test, y_pred_proba)


print( 
    "   CatBoost metrics   \n"
    f" accuracy:  {acc_cat: .3f} \n"
    f" precision: {prec_cat: .3f} \n"
    f" recall:    {rec_cat: .3f} \n"
    f" f1-score:  {f1_cat: .3f}  \n"
    f" roc_auc:   {roc_cat: .3f} \n"
)
