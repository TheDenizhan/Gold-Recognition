import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import ExtraTreesRegressor

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

import warnings

warnings.filterwarnings("ignore")

sns.set_style("darkgrid")

data = pd.read_csv('gld_price_data.csv')

data.head()

data.shape

data.info()

data.isna().sum()

data.describe()

data.duplicated().sum()

data.nunique()

data.hist(figsize=(24,4), layout=(1,5), color="g");

data.plot(kind="kde", subplots=True, layout=(5,1), figsize=(24,15),sharex=False, sharey=False);

data.plot(kind="box", subplots=True, layout=(1,5), figsize=(24,4),sharex=False, sharey=False);

plt.figure(figsize=(8,6))
numeric_data = data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_data.corr(), annot=True, cmap=plt.cm.CMRmap_r)


sns.pairplot(data.sample(n=100))

numeric_col_names = ['SPX','GLD', 'USO', 'SLV', 'EUR/USD']

fig, ax = plt.subplots(len(numeric_col_names), figsize=(16,16))

for i, col_val in enumerate(numeric_col_names):
    sns.distplot(data[col_val], hist=True, ax=ax[i])
    ax[i].set_title('Freq dist '+col_val, fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)
    ax[i].set_ylabel('Count', fontsize=8)

plt.show()

def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)

col_names = ['SPX','GLD', 'USO', 'SLV', 'EUR/USD']

fig, ax = plt.subplots(len(col_names), figsize=(10,25))

for i, col_val in enumerate(col_names):
    x = data[col_val][:1000]
    sns.distplot(x, ax=ax[i], rug=True, hist=False)
    outliers = x[percentile_based_outlier(x)]
    ax[i].plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    ax[i].set_title('Outlier detection - {}'.format(col_val), fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)

plt.show()

data_preprocessed = data.copy()

data_preprocessed.isnull().mean() * 100

date_columns = ['Date']
num_columns = data_preprocessed.select_dtypes(include=['float64', 'int64']).columns
target_col = 'GLD'

num_columns

data_preprocessed.head()

data_preprocessed['Date'] = pd.to_datetime(data_preprocessed['Date'])

data_preprocessed.reset_index(drop=True, inplace=True)

data_preprocessed.drop(['Date'], axis=1, inplace=True)

## train test split

X = data.drop(['Date','GLD'],axis=1)
Y = data['GLD']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

## Feature Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Feature Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.feature_selection import SelectKBest

fs = SelectKBest(k=3)
X_train_scaled = fs.fit_transform(X_train_scaled, y_train)
X_test_scaled = fs.transform(X_test_scaled)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

lr = LinearRegression().fit(X_train_scaled, y_train)
y_lr = lr.predict(X_test_scaled)

knn = KNeighborsRegressor(n_neighbors=3).fit(X_train_scaled, y_train)
y_knn = knn.predict(X_test_scaled)

dt = DecisionTreeRegressor().fit(X_train_scaled, y_train)
y_dt = dt.predict(X_test_scaled)

br = BayesianRidge().fit(X_train_scaled,y_train)
y_br = br.predict(X_test_scaled)

en = ElasticNet().fit(X_train_scaled,y_train)
y_en = en.predict(X_test_scaled)
gb = GradientBoostingRegressor().fit(X_train_scaled,y_train)
y_gb = gb.predict(X_test_scaled)
hr = HuberRegressor().fit(X_train_scaled,y_train)
y_hr = hr.predict(X_test_scaled)

svr = SVR().fit(X_train_scaled,y_train)
y_svr = svr.predict(X_test_scaled)

xgb = XGBRegressor().fit(X_train_scaled,y_train)
y_xgb = xgb.predict(X_test_scaled)

rf = RandomForestRegressor().fit(X_train_scaled,y_train)
y_rf = rf.predict(X_test_scaled)

et = ExtraTreesRegressor().fit(X_train_scaled,y_train)
y_et = et.predict(X_test_scaled)

lr_score = metrics.r2_score(y_test, y_lr)
knn_score = metrics.r2_score(y_test, y_knn)
dt_score = metrics.r2_score(y_test, y_dt)
br_score = metrics.r2_score(y_test, y_br)
en_score = metrics.r2_score(y_test, y_en)
gb_score = metrics.r2_score(y_test, y_gb)
hr_score = metrics.r2_score(y_test, y_hr)
svr_score = metrics.r2_score(y_test, y_svr)
xgb_score = metrics.r2_score(y_test, y_xgb)
rf_score = metrics.r2_score(y_test, y_rf)
et_score = metrics.r2_score(y_test, y_et)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("*"*20, "Accuracy", "*"*20)

print("-"*50)
print("| Linear Regression: ", lr_score)
print("-"*50)

print("-"*50)
print("| KNearest Neighbors: ", knn_score)
print("-"*50)

print("-"*50)
print("| Decision Tree: ", dt_score)
print("-"*50)

print("-"*50)
print("| Bayesian Ridge: ", br_score)
print("-"*50)

print("-"*50)
print("| Elastic Net: ", en_score)
print("-"*50)

print("-"*50)
print("| Gradient Boosting: ", gb_score)
print("-"*50)

print("-"*50)
print("| Huber: ", hr_score)
print("-"*50)

print("-"*50)
print("| Support Vectore Machine: ", svr_score)
print("-"*50)

print("-"*50)
print("| XGBoost: ", xgb_score)
print("-"*50)

print("-"*50)
print("| Random Forest: ", rf_score)
print("-"*50)

print("-"*50)
print("| Extra Tree: ", et_score)
print("-"*50)

metric_val = {
    "R2 score": {
        "Linear Regression ": lr_score * 100,
        "KNearest Neighbors": knn_score * 100,
        "Decision Tree": dt_score * 100,
        "Bayesian Ridge": br_score * 100,
        "Elastic Net": en_score * 100,
        "Gradient Boosting": gb_score * 100,
        "Huber ": hr_score * 100,
        "Support Vectore Machine": svr_score * 100,
        "XGBoost": xgb_score * 100,
        "Random Forest": rf_score * 100,
        "Extra Tree": et_score * 100
    }
}

ax = pd.DataFrame(metric_val).plot(kind="bar",
                                   figsize=(20, 10),
                                   legend=False,
                                   title="R2 Score",
                                   color='#FAC205');

for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.005, p.get_height() * 1.005))

    from sklearn.model_selection import RandomizedSearchCV

    n_estimators = [int(x) for x in np.linspace(start=1, stop=500, num=500)]
    criterion = ['squared_error', 'absolute_error']
    max_depth = [int(x) for x in np.linspace(10, 200, num=100)]
    min_samples_split = [2, 5, 10, 20]
    min_samples_leaf = [1, 2, 4, 8]
    max_features = [None, 'sqrt', 'log2']
    max_leaf_nodes = [int(x) for x in np.linspace(10, 200, num=100)]
    max_depth.append(None)

    random_grid = {'n_estimators': n_estimators,
                   'criterion': criterion,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_leaf_nodes': max_leaf_nodes}
    print(random_grid)

    et = ExtraTreesRegressor()
    et_random = RandomizedSearchCV(estimator=et, param_distributions=random_grid, n_iter=200, cv=3, verbose=0,
                                   random_state=42, n_jobs=-1)
    et_random.fit(X_train, y_train)

    et_random.best_params_

    et_tuned = ExtraTreesRegressor(**et_random.best_params_)
    et_tuned.fit(X_train_scaled, y_train)

    y_pred_et = et_tuned.predict(X_test_scaled)
    r2ett = metrics.r2_score(y_test, y_pred_et)

    print("-" * 30)
    print("Accuracy: ", r2ett)
    print("-" * 30)

    conclusion = {
        "R2 score": {
            "Baseline Model ": et_score * 100,
            "Model after hyperparameter tuning": r2ett * 100
        }
    }

    ax = pd.DataFrame(conclusion).plot(kind="bar",
                                       figsize=(10, 5),
                                       legend=False,
                                       title="R2 Score",
                                       color='m');

    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.005, p.get_height() * 1.005))
