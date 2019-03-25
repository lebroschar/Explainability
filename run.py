"""Practice with some model explanation tools.

1) Make a random forest model

"""
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rfpimp
import seaborn as sns
import shap
from sklearn import decomposition
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

# Parameters
data_file = './data/applicants_mod_cat.csv'
train_cols = ['raw_FICO_retail', 'paymen_Bi-weekly', 'age', 'payment_amount',
              'paymen_Monthly', 'paymen_Semi-monthly', 'bank_a_3+ years', 
              'num_payments', 'paymen_Weekly', 'how_us_Loans', 'rent_or_own']
train_cols = ['raw_FICO_retail', 'paymen_Bi-weekly', #'raw_FICO_telecom',
              'age', 'payment_amount_approved',
              'num_loans', 'raw_FICO_bank_card', 'payment_amount',
              'paymen_Monthly', 'paymen_Semi-monthly', 'bank_a_3+ years',
              'app_at_lunch',
              'raw_FICO_money', 'num_payments', 'paymen_Weekly', 'how_us_Loans',
              'rent_or_own', 'addres_z_84047', 'breakeven_alpha',
              'raw_l2c_score',
              'bank_a_6 months or less', 'how_us_Fun', 'reside_7-12 months',
              'direct_deposit', 'reside_3+ years', 'low_raw_l2c',
              'addres_z_84118',
              'expected_proceeds', 'other__left_blank', 'bank_r_b_124302150',
              'bank_r_b_123103729', 'bank_r_b_324377516', 'bank_r_b_124002971',
              'pay_by_ach', 'bank_a_left_blank', 'addres_z_84010']
target_col = 'good_loan'
thresh = 0.0


# INITIALIZE ------------------------------------------------------------------

# Load data
data = pd.read_csv(data_file, index_col='customer_id')

# Create evaluation folds
fold_list = model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=2,
                                                    random_state=1111)


# LOOK AT DATA ----------------------------------------------------------------

print('Data Shape: ', data.shape)

# Print min/max/mean/std
print(data.agg(['min', 'mean', 'median', 'max', 'std']).transpose())

# Look at correlation
rfpimp.plot_corr_heatmap(data[train_cols], figsize=(10, 8))
plt.show(block=False)


# BUILD A CLASSIFIER ----------------------------------------------------------

# Get train/validation split
train_inds, valid_inds = next(fold_list.split(data[train_cols],
                                              data[target_col]))
X_train = data.loc[data.index[train_inds], train_cols]
y_train = data.loc[data.index[train_inds], target_col]
X_valid = data.loc[data.index[valid_inds], train_cols]
y_valid = data.loc[data.index[valid_inds], target_col]

# Normalize data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Apply PCA
"""
pca = decomposition.PCA()
X_train = pca.fit_transform(X_train)
X_valid = pca.transform(X_valid)
"""

# Build a logistic regression
logreg = linear_model.LogisticRegression()
logreg.fit(X_train, y_train)
preds_train = logreg.predict_proba(X_train)[:, 1]
preds_valid = logreg.predict_proba(X_valid)[:, 1]

inb_score = metrics.log_loss(y_train, preds_train)
oob_score = metrics.log_loss(y_valid, preds_valid)
print('LogReg - In-bag: %f, OOB: %f' % (inb_score, oob_score))

# Build a random forest
rf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=200,
                                     max_features=0.35,
                                     min_samples_leaf=10)
rf.fit(X_train, y_train)
preds_train = rf.predict_proba(X_train)[:, 1]
preds_valid = rf.predict_proba(X_valid)[:, 1]

inb_score = metrics.log_loss(y_train, preds_train)
oob_score = metrics.log_loss(y_valid, preds_valid)
print('RF - In-bag: %f, OOB: %f' % (inb_score, oob_score))


# LOOK AT FEATURE IMPORTANCE --------------------------------------------------

def sort_by_imp(train_cols, importance):
    return sorted(zip(train_cols, importance), key=lambda x: x[1],
                  reverse=True)

# Importances from models
logreg_imp = sort_by_imp(train_cols, logreg.coef_[0])
rf_imp = sort_by_imp(train_cols, rf.feature_importances_)

# Try permutation importance
perm_imps = {}
for label, model in [('rf', rf), ('logreg', logreg)]:
    perm = PermutationImportance(model, n_iter=10, cv='prefit').fit(
        X_valid, y_valid)
    perm_imp = sort_by_imp(train_cols, perm.feature_importances_)
    perm_imps[label] = perm_imp
    
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=np.array(perm.results_), ax=ax)
    _ = ax.set_xticklabels(train_cols, rotation=90)
    _ = ax.set_ylabel('Improvement in log_loss')
    _ = ax.set_title(label)
    fig.tight_layout()
    plt.show(block=False)

"""
train_cols = [x[0] for x in perm_imp if x[1] > 0]
"""

# Try out SHAP
