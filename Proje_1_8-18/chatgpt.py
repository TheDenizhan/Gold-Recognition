
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import r2_score
import sklearn
from sklearn import metrics

# Set options
pd.set_option('display.max_columns', None)
sns.set_style("darkgrid")

# Load data
data = pd.read_csv('gld_price_data.csv')

# Data exploration
print(data.head())
print(data.shape)
print(data.info())
print(data.isna().sum())
print(data.describe())
print(data.duplicated().sum())
print(data.nunique())

# Visualizations
data.hist(figsize=(24, 4), layout=(1, 5), color="g")
data.plot(kind="kde", subplots=True, layout=(5, 1), figsize=(24, 15), sharex=False, sharey=False)
data.plot(kind="box", subplots=True, layout=(1, 5), figsize=(24, 4), sharex=False, sharey=False)
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap=plt.cm.CMRmap_r)
sns.pairplot(data.sample(n=100))

# Outlier detection
col_names = ['SPX', 'GLD', 'USO', 'SLV', 'EUR/USD']
fig, ax = plt.subplots(len(col_names), figsize=(10, 25))
for i, col_val in enumerate(col_names):
    x = data[col_val][:1000]
    sns.distplot(x, ax=ax[i], rug=True, hist=False)
    outliers = x[percentile_based_outlier(x)]
    ax[i].plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)
    ax[i].set_title('Outlier detection - {}'.format(col_val), fontsize=10)
    ax[i].set_xlabel(col_val, fontsize=8)

plt.show()

# Data preprocessing
data_preprocessed = data.copy()
data_preprocessed.isnull().mean() * 100

# Date handling
date_columns = ['Date']
num_columns = data_preprocessed.select_dtypes(include=['float64', 'int64']).columns
target_col = 'GLD'
data_preprocessed['Date'] = pd.to_datetime(data_preprocessed['Date'])
data_preprocessed.reset_index(drop=True, inplace=True)
data_preprocessed.drop(['Date'], axis=1, inplace=True)

# Train-test split
X = data.drop(['Date', 'GLD'], axis=1)
Y = data['GLD']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
fs = SelectKBest(k=3)
X_train_scaled = fs.fit_transform(X_train_scaled, y_train)
X_test_scaled = fs.transform(X_test_scaled)

# Model training and evaluation
models = [
    LinearRegression(), KNeighborsRegressor(n_neighbors=3),
    DecisionTreeRegressor(), BayesianRidge(),
    ElasticNet(), GradientBoostingRegressor(),
    HuberRegressor(), SVR(), XGBRegressor(),
    RandomForestRegressor(), ExtraTreesRegressor()
]

scores = {}
for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    score = r2_score(y_test, y_pred)
    scores[model_name] = score * 100

# Display results
print("*" * 20, "Accuracy", "*" * 20)
for model_name, score in scores.items():
    print("-" * 50)
    print(f"| {model_name}: {score}")
print("-" * 50)

# Model comparison visualization
metric_val = {"R2 score": scores}
ax = pd.DataFrame(metric_val).plot(kind="bar", figsize=(20, 10), legend=False, title="R2 Score", color='#FAC205')

for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.005, p.get_height() * 1.005))

# Hyperparameter tuning
random_grid = {'n_estimators': [int(x) for x in np.linspace(start=1, stop=500, num=500)],
               'criterion': ['squared_error', 'absolute_error'],
               'max_depth': [int(x) for x in np.linspace(10, 200, num=100)],
               'min_samples_split': [2, 5, 10, 20],
               'min_samples_leaf': [1, 2, 4, 8],
               'max_features': [None, 'sqrt', 'log2'],
               'max_leaf_nodes': [int(x) for x in np.linspace(10, 200, num=100)]}

et = ExtraTreesRegressor()
et_random = RandomizedSearchCV(estimator=et, param_distributions=random_grid, n_iter=200, cv=3, verbose=0,
                               random_state=42, n_jobs=-1)
et_random.fit(X_train, y_train)

et_random.best_params_

et_tuned = ExtraTreesRegressor(**et_random.best_params_)
et_tuned.fit(X_train_scaled, y_train)

y_pred_et = et_tuned.predict(X_test_scaled)
r2ett = r2_score(y_test, y_pred_et)

# Model comparison after hyperparameter tuning
conclusion = {
    "R2 score": {
        "Baseline Model ": scores['ExtraTreesRegressor'] * 100,
        "Model after hyperparameter tuning": r2ett * 100
    }
}

ax = pd.DataFrame(conclusion).plot(kind="bar", figsize=(10, 5), legend=False, title="R2 Score", color='m')

for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.show()
