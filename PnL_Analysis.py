import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


sns.set()

file_location = 'C:/Users/user/Downloads'

raw_file = pd.read_csv(fr'{file_location}/PnL_analysis.csv')

df = pd.DataFrame(raw_file)
df = df.rename(columns={'Unnamed: 0': 'date'})
df.set_index('date',inplace=True)
df['date'] = pd.to_datetime(df.index)



def parse_date_feature(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['week_of_year'] = df['week_of_year'].map(lambda x: 1 if x>52 else x)
    return df

df = parse_date_feature(df)

df[df.columns] = df[df.columns].apply(pd.to_numeric,downcast='float',errors='coerce')
df = df.fillna(-1)
df = df.drop(columns='date')

print(df.columns[27:])

corr = df.corr()

mask = np.triu(np.ones_like(corr,dtype=bool))
cmap = sns.diverging_palette(230,30,as_cmap=True)

daily_pnl = df['Daily_PnL'].mean()
cumulative_pnl = pd.DataFrame(df['Daily_PnL']).cumsum()
# fig = plt.subplots(figsize=(15,15))
# # sns.heatmap(corr,mask=mask,cmap=cmap,vmax=0.5,center=0,square=True,
# #             linewidth=0.5,cbar_kws={'shrink':0.5})

target_column = 'Daily_PnL'
columns_to_ignore = ['VIX','USDILS_Gamma','USDILS_Vega']
input_cols =[]

for c in df.columns:
    if c not in columns_to_ignore:
        if c !=target_column:
            input_cols.append(c)

X = df[input_cols]
y = df[target_column]

X_train , X_val , y_train , y_val = train_test_split(X,y , test_size=0.4,random_state=42)
X_val , X_test , y_val , y_test = train_test_split(X_val,y_val,test_size=0.5,random_state=42)

train_inputs = X_train.copy()
train_targets = y_train.copy()

val_inputs = X_val.copy()
val_targets = y_val.copy()

test_inputs = X_test.copy()
test_targets = y_test.copy()

scaler = MinMaxScaler()
scaler.fit(df[input_cols])

train_inputs[input_cols] = scaler.transform(train_inputs[input_cols])
val_inputs[input_cols] = scaler.transform(val_inputs[input_cols])
test_inputs[input_cols] = scaler.transform(test_inputs[input_cols])





X_train , X_val , X_test = train_inputs[input_cols] , val_inputs[input_cols] , test_inputs[input_cols]

OLS = LinearRegression(n_jobs=-1)
OLS.fit (X_train,y_train)

ols_train_pred = OLS.predict(X_train)
ols_val_pred = OLS.predict(X_val)

train_mae = mean_absolute_error(y_train,ols_train_pred)
val_mae = mean_absolute_error(y_val,ols_val_pred)
train_r2 = r2_score(train_targets,ols_train_pred)
val_r2 = r2_score(val_targets,ols_val_pred)

print(f'train mae : {train_mae: ,.0f}, val mae : {val_mae: ,.0f} ,\n'
      f' train r2 : {train_r2: ,.3f} , val r2 :{val_r2: ,.3f}')

tree = DecisionTreeRegressor(random_state=100,
                             max_depth=8,
                             max_leaf_nodes=8,
                             max_features=18,min_samples_split=40,
                             criterion='friedman_mse')

tree.fit(X_train,train_targets)
train_pred = tree.predict(X_train)
val_pred = tree.predict(X_val)

train_mae = mean_absolute_error(train_targets,train_pred)
val_mae = mean_absolute_error(val_targets,val_pred)
print(f'train mae :{train_mae: ,.0f}, val mae : {val_mae: ,.0f}')

tree_importance = tree.feature_importances_
tree_importance_df = pd.DataFrame({'feature':X_train.columns,
                                   'importance':tree_importance}).sort_values('importance',ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=tree_importance_df[0:20], x='importance',y='feature')
plt.title('Decision Tree Features Importance ')
plt.show()

def max_depth_error(ml):
    model = DecisionTreeRegressor(min_samples_split=ml,random_state=42)
    model.fit(X_train,train_targets)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_acc = mean_absolute_error(train_targets,train_pred)
    val_acc = mean_absolute_error(val_targets,val_pred)
    return {'Max depth':ml , 'Train Error': train_acc, 'Validation Error':val_acc}

def evalulate_and_plot(min_depth,max_depth):
    errors_df = pd.DataFrame([max_depth_error(ml) for ml in range(min_depth,max_depth)])

    plt.plot(errors_df['Max depth'],errors_df['Train Error'])
    plt.plot(errors_df['Max depth'],errors_df['Validation Error'])
    plt.xticks(range(min_depth,max_depth,2))
    plt.xlabel('Max depth')
    plt.legend(['Training' , 'Validation'])
    plt.show()

def test_params(**params):
    tree = DecisionTreeRegressor(random_state=42,**params).fit(X_train,train_targets)
    train_pred = tree.predict(X_train)
    val_pred = tree.predict(X_val)
    print(f'{mean_absolute_error(train_targets,train_pred) : ,.0f} , {mean_absolute_error(val_targets,val_pred): ,.0f} ')


rf = RandomForestRegressor(random_state=42,
                           n_jobs=-1,
                           n_estimators=100,
                           max_depth=10,
                           max_features='sqrt',
                           max_leaf_nodes=20,
                           min_samples_leaf=6,
                           criterion='absolute_error',
                           bootstrap=False).fit(X_train,train_targets)
rf_predict_train = rf.predict(X_train)
rf_predict_val = rf.predict(X_val)
rf_mae_train = mean_absolute_error(train_targets,rf_predict_train)
rf_mae_val= mean_absolute_error(val_targets,rf_predict_val)
rf_r2_train = r2_score(train_targets,rf_predict_train)
rf_r2_val = r2_score(val_targets,rf_predict_val)

print(f'train mae : {rf_mae_train: ,.0f} , val mae : {rf_mae_val: ,.0f} ')
print(f'train r^2 : {rf_r2_train: .2f} , val r^2: {rf_r2_val: .2f}')

rf_feature_importance = rf.feature_importances_
rf_feature_importance_df = pd.DataFrame({'feature': X_train.columns,
                                         'importance':rf_feature_importance}).sort_values(by='importance',ascending=False)
sns.barplot(data=rf_feature_importance_df[:20], x='importance',y='feature')
plt.show()

def train_and_evaluate(X_train, train_targets, X_val, val_targets, **params):
    model = RandomForestRegressor(n_jobs=-1, **params)
    model.fit(X_train,train_targets)
    train_mae = mean_absolute_error(model.predict(X_train),train_targets)
    val_mae = mean_absolute_error(model.predict(X_val),val_targets)
    return  model, train_mae, val_mae


X= train_inputs[input_cols]
x_test = val_inputs[input_cols]
targets = df[target_column]

def test_params_kfold(n_splits,**params):

    train_maes, val_maes, models= [],[],[]
    kfold = KFold(n_splits,shuffle=False)

    for train_idx , val_idx in kfold.split(X):
        X_train , train_targets = X.iloc[train_idx], targets.iloc[train_idx]
        X_val , val_targets = X.iloc[val_idx] , targets.iloc[val_idx]
        rf , train_mae , val_mae = train_and_evaluate(X_train,
                                                      train_targets,
                                                      X_val,
                                                      val_targets,
                                                      **params)
        models.append(rf)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        print(f'Train MAE : {np.mean(train_maes) : ,.0f} , Validation MAE : {np.mean(val_maes) : ,.0f}')
        return models

test_params_kfold(n_splits=10,
                  n_estimators=200,
                  max_depth=6,
                  max_features='sqrt',
                  min_samples_leaf=6,
                  min_samples_split=60,
                  max_leaf_nodes=30, bootstrap=True,
                  min_impurity_decrease=0.5,max_samples=50)


def predict_avg(predict_inputs,X,targets,n_splits, **params):
    kfold = KFold(n_splits=n_splits,shuffle=False)

    models = []
    for train_idxs , val_idxs in kfold.split(X):
        X_train , train_targets = X.iloc[train_idxs] , targets.iloc[train_idxs]
        X_val , val_targets = X.iloc[val_idxs] , targets.iloc[val_idxs]
        rf , train_mae , val_mae = train_and_evaluate(X_train,
                                                      train_targets,
                                                      X_val,
                                                      val_targets,**params)
    models.append(rf)

    return np.mean([model.predict(predict_inputs) for model in models],axis=0)

