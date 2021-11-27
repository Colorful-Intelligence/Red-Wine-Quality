#%% LIBRARIES

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from scipy import stats
from scipy.stats import norm,skew

from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.linear_model import LinearRegression , Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone

# XGBOOST
import xgboost as xgb
# WARNINGS
import warnings 
warnings.filterwarnings('ignore')

#%% Reading the dataset

data = pd.read_csv("winequality-red.csv")

#%% Exploratory Data Analysis - 1

print(data.shape) # (1599, 12)

print(data.head())

print(data.info()) # Dataset has no any missing value

"""
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed acidity         1599 non-null   float64
 1   volatile acidity      1599 non-null   float64
 2   citric acid           1599 non-null   float64
 3   residual sugar        1599 non-null   float64
 4   chlorides             1599 non-null   float64
 5   free sulfur dioxide   1599 non-null   float64
 6   total sulfur dioxide  1599 non-null   float64
 7   density               1599 non-null   float64
 8   pH                    1599 non-null   float64
 9   sulphates             1599 non-null   float64
 10  alcohol               1599 non-null   float64
 11  quality               1599 non-null   int64  
dtypes: float64(11), int64(1)
memory usage: 150.0 KB
"""

# Statistical Informaton about the dataset
describe = data.describe()
print(describe) # Dataset has a litle skewness

#%% Exploratory Data Analysis - 2 (Pair Plot & Correlaation Matrix)
sns.pairplot(data,diag_kind = "kde",markers = "+")
plt.show()

# Correlation Matrix
corr_matrix = data.corr()
sns.clustermap(corr_matrix,annot = True,fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()

#%% Exploratory Data Analysis - 3 (Box Plot)

for c in data.columns:
    plt.figure()
    sns.boxplot(x = c,data = data,orient = "v")


#%% OUTLIER VALUES

def detect_outliers(df,features):
    outlier_indices = []
    for c in features:
        # 1 st quartile
        Q1 = np.percentile(df[c],25)
        
        # 3 rd quartile
        Q3 = np.percentile(df[c],75)
        
        # IQR
        IQR = Q3 - Q1
        
        # Outlier step
        outlier_step = IQR * 1.5
   
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step)].index
        
        # store indeces
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)

    return multiple_outliers


data.loc[detect_outliers(data,['alcohol', 'chlorides','citric acid','density','fixed acidity','free sulfur dioxide','pH','quality','residual sugar','sulphates','total sulfur dioxide','volatile acidity'])]


# drop outliers
data = data.drop(detect_outliers(data,['alcohol', 'chlorides','citric acid','density','fixed acidity','free sulfur dioxide','pH','quality','residual sugar','sulphates','total sulfur dioxide','volatile acidity']),axis = 0).reset_index(drop = True)

# Data Shape After Outlier Detection
print(data.shape) # (1562, 12)

#%% Feature Engineering 

# Skewness
# quality dependent variable
sns.distplot(data.quality,fit = norm)
(mu,sigma) = norm.fit(data["quality"])
print("mu: {} , sigma : {}".format(mu,sigma)) # mu: 5.635083226632522 , sigma : 0.7969200632330968


# QQ Plot

variables = ['fixed acidity','volatile acidity','citric acid','free sulfur dioxide','total sulfur dioxide','pH','sulphates','alcohol', 'quality']

def draw_qqPlot(values):
    for i in values:
        plt.figure()
        stats.probplot(data[i],plot = plt)
        plt.show()
        
draw_qqPlot(variables)

#-------------------------------------------------------#

# LOG TRANSFORM
# We will perform log transformation to reduce skewness.


def log_Transform(values):
    for c in values:
        data[c] = np.log1p(data[c])
        
log_Transform(variables)
    

#-------------------------------------------------------#
# Pair plot after Log Transform
sns.pairplot(data,diag_kind = "kde",markers = "+")
plt.show()



## qq plot after log transform operation
draw_qqPlot(variables)


#%% Normalization Operation

data = (data - np.min(data)) / (np.max(data)-np.min(data))

y = data.quality.values
x = data.drop(["quality"],axis = 1)

#%% Train-Test Split

test_size = 0.3
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = test_size,random_state = 42)

#%% Linear Regression

lr = LinearRegression()
lr.fit(X_train,Y_train)
print("LR Coefficients: ",lr.coef_)

"""
LR Coefficients:  [ 0.05042779 -0.22636665 -0.04056969  0.00446684 -0.13817852  0.12458859
 -0.16130607  0.00359901 -0.07497574  0.25313286  0.31582308]
"""


y_predicted_dummy = lr.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Linear Regression MSE : ",mse) # Linear Regression MSE :  0.016245874755767366



# We are going to try to minimize the mean square error by using regülarization methods (Lasso , Ridge , Elasticnet & XGBOOST)

#%% RIDGE REGULARIZATION

ridge  = Ridge(random_state = 42,max_iter = 10000)

alphas = np.logspace(-4,-0.5,30)
tuned_parameters = [{"alpha": alphas}]
n_folds = 5

clf = GridSearchCV(ridge,tuned_parameters,cv = n_folds,scoring = "neg_mean_squared_error",refit = True)
clf.fit(X_train,Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Ridge Coeff : ",clf.best_estimator_.coef_)


"""
Ridge Coeff :  [ 0.05678874 -0.22408011 -0.03836089  0.00670223 -0.1284175   0.11977167
 -0.15611559 -0.00517168 -0.06702258  0.24746891  0.31135867]
"""


ridge = clf.best_estimator_

print("Ridge Best Estimator: ",ridge) # Ridge Best Estimator:  Ridge(alpha=0.31622776601683794, max_iter=10000, random_state=42)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)

print("Ridge MSE : ",mse) # Ridge MSE :  0.016214814820665027
print("-----------------------------------------")

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")

#%% LASSO REGULARIZATION

lasso  = Lasso(random_state = 42,max_iter = 10000)

alphas = np.logspace(-4,-0.5,30)
tuned_parameters = [{"alpha": alphas}]
n_folds = 5

clf = GridSearchCV(lasso,tuned_parameters,cv = n_folds,scoring = "neg_mean_squared_error",refit = True)
clf.fit(X_train,Y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Lasso Coeff : ",clf.best_estimator_.coef_)

"""
Lasso Coeff :  [ 0.04284167 -0.2172771  -0.02356776 -0.         -0.08466332  0.09583841
 -0.13293145 -0.         -0.04622307  0.22545023  0.31300599]
"""



lasso = clf.best_estimator_

print("Lasso Best Estimator: ",lasso) # Lasso Best Estimator:  Lasso(alpha=0.00023018073130224678, max_iter=10000, random_state=42)



y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)

print("Lasso MSE : ",mse) # Lasso MSE :  0.01620672933321048
print("-----------------------------------------")

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")


#%% ELASTICNET REGULARIZATION


parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}

eNet = ElasticNet(random_state=42, max_iter=10000)
clf = GridSearchCV(eNet, parametersGrid, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
clf.fit(X_train, Y_train)


print("ElasticNet Coef: ",clf.best_estimator_.coef_)

"""
ElasticNet Coef:  [ 0.04428433 -0.21577798 -0.02378151 -0.         -0.08678084  0.09646613
 -0.13337741 -0.         -0.04761272  0.22189747  0.31059396]
"""




print("ElasticNet Best Estimator: ",clf.best_estimator_)

"""

ElasticNet Best Estimator:  ElasticNet(alpha=0.0006995642156712634, l1_ratio=0.25, max_iter=10000,
           random_state=42)
"""




y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("ElasticNet MSE: ",mse) # ElasticNet MSE:  0.016189843161845224



#%% XGBOOST

parametersGrid = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500,1000]}

model_xgb = xgb.XGBRegressor()

clf = GridSearchCV(model_xgb, parametersGrid, cv = n_folds, scoring='neg_mean_squared_error', refit=True, n_jobs = 5, verbose=True)

clf.fit(X_train, Y_train)
model_xgb = clf.best_estimator_

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("XGBRegressor MSE: ",mse) # XGBRegressor MSE:  0.01334334486443362

# %% Averaging Models

class AveragingModels():
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)  


averaged_models = AveragingModels(models = (model_xgb, eNet))
averaged_models.fit(X_train, Y_train)

y_predicted_dummy = averaged_models.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Averaged Models MSE: ",mse) # Averaged Models MSE:  0.014973949123403682 (ElasticNet & XGBOOST)







































