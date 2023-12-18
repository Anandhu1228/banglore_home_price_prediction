import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
import pickle
import json

df = pd.read_csv('bangaluru_house_prices.csv')
df = df.drop(['area_type','society','balcony','availability'],axis='columns')
df = df.dropna()
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def get_avg(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df1 = df.copy()
df1['total_sqft'] = df1['total_sqft'].apply(get_avg)
df1['price_per_sqft'] = df1.price*100000/df1.total_sqft

df1.location = df1.location.apply(lambda x: x.strip())
loc_stats = df1.groupby('location')['location'].agg('count').sort_values(ascending = False)

loc_stats_less10 = loc_stats[loc_stats<=10]
df1.location = df1.location.apply(lambda x: 'other' if x in loc_stats_less10 else x)

def remove_outlier(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df2 = remove_outlier(df1)

def plot_scatter(df,location):
    bhk2 = df[(df.location == location) & (df.bhk ==2)]
    bhk3 = df[(df.location == location) & (df.bhk ==3)]
    plt.scatter(bhk2.total_sqft,bhk2.price,color = 'blue',label = '2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color = 'green',label = '4 BHK',marker='+',s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price Per Square Feet')
    plt.title(location)
    plt.legend
    plt.show()

def remove_bhk_outlier(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis = 'index')

df3 = remove_bhk_outlier(df2)

df4 = df3[df3.bath < (df3.bhk+2)]
df4 = df4.drop('price_per_sqft',axis='columns')

dummies = pd.get_dummies(df4.location)
dum = dummies.drop('other',axis='columns')

df5 = pd.concat([df4,dum],axis='columns')
df6 = df5.drop(['location','size'],axis='columns')

x = df6.drop('price',axis='columns')
y = df6.price

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = LinearRegression(fit_intercept=False)
model.fit(x_train,y_train)
scores = model.score(x_test,y_test)

'''model = BaggingRegressor(
    base_estimator=LinearRegression(fit_intercept=False),
    n_estimators=100,
    max_samples=.9,
    oob_score= True,random_state=4
)
model.fit(x_train,y_train)
score1 = model.oob_score_
print("MODEL OOB SCORE :  ",score1)
score2 = model.score(x_test,y_test)
print("MODEL SCORE :  ",score2)'''

'''cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
score = cross_val_score(LinearRegression(),x_train,y_train,cv=cv)

def find_best_model(x_train,y_train):
    models = {
        'linear_regression' : {
            'model' : LinearRegression(),
            'params' : {
                'fit_intercept' : [True,False]
            }
        },

        'lasso' : {
            'model' : Lasso(),
            'params' : {
                'alpha' : [1,2],
                'selection' : ['random','cyclic']
            }
        },
        'decision_tree' : {
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion': ['absolute_error', 'poisson', 'squared_error', 'friedman_mse'],
                'splitter' : ['best','random']
            }
        }
    }
    scores = []
    css = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for model_name,config in models.items():
        grid = GridSearchCV(config['model'],config['params'],cv=css,return_train_score=False)
        grid.fit(x_train,y_train)
        scores.append({
            'model' : model_name,
            'best_score' : grid.best_score_,
            'best_parameter' : grid.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_parameter'])

result = find_best_model(x_train,y_train)
print(result)'''

def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(x.columns==location)[0][0]

    xn = np.zeros(len(x.columns))
    xn[0] = sqft
    xn[1] = bath
    xn[2] = bhk
    if loc_index >= 0:
        xn[loc_index] = 1
    return model.predict([xn])[0]

with open('bangaluru_house_prices_model.pickle','wb') as f:
    pickle.dump(model,f)

columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open('columns.json','w') as f:
    f.write(json.dumps(columns))
