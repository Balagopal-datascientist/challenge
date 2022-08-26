import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
df_csv=pd.read_csv("D:\\Aishu\\challenge.csv",index_col=[0])
df_csv=df_csv.drop("type",axis=1)
y=df_csv["target"]
x=df_csv.drop("target",axis=1)
# print(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

def fit(x_train,y_train,flag):
    if flag=="svr":
        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVR())])
    elif flag=="rfr":
        pipe=Pipeline([('scaler', StandardScaler()), ('rfr', RandomForestRegressor())])
    pipe.fit(x_train,y_train)
    joblib.dump(pipe, 'model.pkl', compress = 1)
    return pipe
def predict(path,x_test):
    loaded_model = joblib.load(path)
    y_pred=loaded_model.predict(x_test)
    y_pred_df=pd.DataFrame(y_pred)
    y_pred_df.to_csv("y_pred.csv")
def evaluate(path,y_test,flag):
    # print(y_pred)
    pred_df=pd.read_csv(path,index_col=[0])
    print(pred_df)
    # print(y_test)
    if flag=="mean_absolute_error":
        print(mean_absolute_error(pred_df,y_test))

fit(x_train,y_train)
predict("model.pkl",x_test)
evaluate("y_pred.csv",y_test)