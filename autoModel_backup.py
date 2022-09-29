# Libararies Import
from tqdm import tqdm
from datetime import datetime

import os, warnings, pickle, gzip

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

class autoModel :

    def __init__(self,
                 trainPath, 
                 testPath,
                 scaling_cols:list,
                 gpu_option: bool=False,
                 gpu_id : str=None,
                 name:str=datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(" ","_").replace("-","").replace(":",""),
                 tuning:bool=False,
                 timeLen:int=1) :

        if gpu_option == True :
            from tensorflow.python.client import device_lib
            
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]= gpu_id
            
            print()
            print("="*100)
            print(f"\n사용할 gpu 수 : {len(device_lib.list_local_devices()[1:])}\n")
            print("="*100)

        self.name = name
        self.save_dir = os.path.join("./results", self.name)
        print(f"작업 경로 : {self.save_dir}")
        if not os.path.isdir(self.save_dir) :
            if not os.path.isdir("./results") :
                os.mkdir("./results")
            os.mkdir(self.save_dir)
        
        # if tuning == True :
        #     import optina

        self.train = pd.read_csv(trainPath)
        self.test = pd.read_csv(testPath)
        self.train.INST_DATE = pd.to_datetime(self.train.INST_DATE)
        self.test.INST_DATE = pd.to_datetime(self.test.INST_DATE)
        self.scaling(target_columns=scaling_cols)

        self.timeLen = timeLen
        if self.timeLen > 1 :
            print("="*100)
            print("Time Series Preprocessing ...")
            self.train = self.TimeSeriesInference(data=self.train)
            self.test = self.TimeSeriesInference(data=self.test)
            print("done!")
            print("="*100)

    def dataset(self) :
        return self.train, self.test

    
    def scaling(self, target_columns) :
        scaler = StandardScaler()
        self.train[target_columns] = scaler.fit_transform(self.train[target_columns])
        self.test[target_columns] = scaler.transform(self.test[target_columns])

        with gzip.open(f"{self.save_dir}/scaler.pickle", "wb") as f :
            pickle.dump(scaler, f)
        
        with gzip.open(f'{self.save_dir}/scaler.pickle','rb') as f:
            scaler = pickle.load(f)


    def TimeSeriesInference(self, data, scores:list=['sum', 'score', 'scored', 'scorel', 'scorelr']) :
        
        df = pd.DataFrame()
        for ids in tqdm(data.SENSORID.unique()) :
            target = data[data.SENSORID == ids]
            target['time_gap'] = list(map(lambda x : x.total_seconds(), target.INST_DATE.diff()))
            target['time_gap'].fillna(0, inplace=True)
            
            target.loc[target.STATUS == 0, scores] = 0
            target.loc[target.time_gap > 12, scores] = 0

            for i in range(len(scores)) :
                target[f"pred_0{i+1}"] = target[scores[i]].rolling(self.timeLen).sum().fillna(0).round(4)

            df = pd.concat([df, target], ignore_index=True)
        
        targets = ['time_gap'] + scores
        df.drop(columns=targets, inplace=True)
        
        return df

    def optimize_threshold(self, label, score) :
        from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score

        precision, recall, threshold = precision_recall_curve(label , score)
        best_cnt_dic = abs(precision - recall)
        threshold_fixed = threshold[np.argmin(best_cnt_dic)]

        return threshold_fixed


    def Qauntile_RF(self, 
                    X, Y,
                    training=True, 
                    quantiles = [0.01, 0.05, 0.50, 0.95 , 0.99],
                    q_tuning=False,
                    n_estimators=200,
                    min_sample_split=10,
                    warm_start=True,  
                    scoring="f1") :
        
        print("Quantile Regression with RF is Started!\n")
        if training==True :    
            # Model Training
            model = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_sample_split, warm_start=warm_start, n_jobs=-1)
            model.fit(X, Y)

            # Model dump
            with gzip.open(f'{self.save_dir}/QRF.pickle','wb') as f:
                pickle.dump(model, f)
        
        # Model Load
        with gzip.open(f'{self.save_dir}/QRF.pickle','rb') as f:
            model = pickle.load(f)

        # make df with every preds
        pred_Q = pd.DataFrame()
        for pred in tqdm(model.estimators_) :
            temp = pd.Series(pred.predict(X).round(6))
            pred_Q = pd.concat([pred_Q,temp],axis=1)
        
        # make df with quantile preds
        RF_actual_pred = pd.DataFrame()
        quantiles = quantiles

        for q in tqdm(quantiles):
            s = pred_Q.quantile(q=q, axis=1)
            RF_actual_pred = pd.concat([RF_actual_pred,s],axis=1,sort=False)
        
        RF_actual_pred.columns = quantiles
        RF_actual_pred['actual'] = Y
        RF_actual_pred['interval'] = RF_actual_pred[np.max(quantiles)] - RF_actual_pred[np.min(quantiles)]
        RF_actual_pred = RF_actual_pred.sort_values('interval')
        RF_actual_pred = RF_actual_pred.round(6)
        

        # Optimizing Threshold - to maximize recall and f1-score
        if q_tuning == True :
            pass
        else :    
            RF_actual_pred['QRF_S'] = RF_actual_pred[quantiles].sum(axis=1) / len(quantiles)
            if training == True :
                self.QRF_Threshold = self.optimize_threshold(label=Y, score=RF_actual_pred['QRF_S'])
                self.train['QRF_S'] = RF_actual_pred['QRF_S']
                RF_actual_pred['pred'] = (RF_actual_pred['QRF_S'] >= self.QRF_Threshold).astype('int')
                self.train['Q_RF'] = RF_actual_pred['pred']
            else :
                self.test['QRF_S'] = RF_actual_pred['QRF_S']
                RF_actual_pred['pred'] = (RF_actual_pred['QRF_S'] >= self.QRF_Threshold).astype('int')
                self.test['Q_RF'] = RF_actual_pred['pred']
        print(f"Threshold is {self.QRF_Threshold}")
        
        
        print(f"F1-SCORE : {f1_score(RF_actual_pred['actual'],RF_actual_pred['pred'])}")

        print(classification_report(RF_actual_pred['actual'],RF_actual_pred['pred']))
        print("="*100)



    def XGBR(self,
             X, Y,
             training:bool=True,
             booster:str='gbtree',
             drop_rate:float=0.5, 
             subsample:float=0.8,
             grow_policy:str='depthwise',
             n_esimators:int=200,
             scale_pos_weight:int=1,
             tree_method:str='gpu_hist',
             predictor:str='gpu_predictor') :

        print(f"XGB Regression with {booster} & {grow_policy} is Started!\n")
        if training==True :    
            # Model Training
            if (booster == 'dart') & (grow_policy=='depth_wise') :
                model = XGBRegressor(n_estimators=n_esimators, booster=booster, drop_rate=drop_rate, grow_policy=grow_policy, scale_pos_weight=scale_pos_weight, n_jobs=-1, tree_method=tree_method, subsample=subsample, predictor=predictor)    
            else :
                model = XGBRegressor(n_estimators=n_esimators, booster=booster, grow_policy=grow_policy, scale_pos_weight=scale_pos_weight, n_jobs=-1, tree_method=tree_method, subsample=subsample, predictor=predictor)
            model.fit(X, Y)
            with gzip.open(f'{self.save_dir}/XGB_{booster}_{grow_policy}.pickle','wb') as f:
                pickle.dump(model, f)

        # Model Load
        with gzip.open(f'{self.save_dir}/XGB_{booster}_{grow_policy}.pickle','rb') as f:
            model = pickle.load(f)

        score = model.predict(X)
        if grow_policy == 'depthwise' :
            if booster == 'dart' :
                score_name = 'dart_score'
            elif booster == 'gbtree' :
                score_name = 'xgb_score'
        else :
            score_name = 'lgbm_score'

        if training == True :
            self.xgb_Threshold = self.optimize_threshold(label=Y, score=score)
            self.train[score_name] = score
            pred = (score >= self.xgb_Threshold).astype('int')
            self.train[score_name.replace("score","pred")] = pred
        else :
            self.test[score_name] = score
            pred = (score >= self.xgb_Threshold).astype('int')
            self.test[score_name.replace("score","pred")] = pred
        
        print(f"Threshold : {self.xgb_Threshold}")
        print(f"F1-SCORE : {f1_score(Y, pred)}\n")
        print(classification_report(Y, pred))
        print("="*100)        

    def XGBC(self,
             X, Y,
             training:bool=True,
             booster:str='gbtree',
             drop_rate:float=0.5,
             subsample:float=0.8,
             grow_policy:str='depthwise',
             n_esimators:int=200,
             scale_pos_weight:int=1,
             tree_method:str='gpu_hist',
             predictor:str='gpu_predictor') :

        print(f"XGB Classification with {booster} & {grow_policy} is Started!\n")
        if training==True :    
            # Model Training
            if (booster == 'dart') & (grow_policy=='depth_wise') :
                model = XGBClassifier(n_estimators=n_esimators, booster=booster, drop_rate=drop_rate, grow_policy=grow_policy, scale_pos_weight=scale_pos_weight, n_jobs=-1, tree_method=tree_method, subsample=subsample, predictor=predictor)
            else :
                model = XGBClassifier(n_estimators=n_esimators, booster=booster, grow_policy=grow_policy, scale_pos_weight=scale_pos_weight, n_jobs=-1, tree_method=tree_method, subsample=subsample, predictor=predictor)
            model.fit(X, Y)
            with gzip.open(f'{self.save_dir}/XGB_{booster}_{grow_policy}.pickle','wb') as f:
                pickle.dump(model, f)

        # Model Load
        with gzip.open(f'{self.save_dir}/XGB_{booster}_{grow_policy}.pickle','rb') as f:
            model = pickle.load(f)

        score = model.predict_proba(X)[:,1]
        if grow_policy == 'depthwise' :
            if booster == 'dart' :
                score_name = 'dart_score'
            elif booster == 'gbtree' :
                score_name = 'xgb_score'
        else :
            score_name = 'lgbm_score'

        if training == True :
            self.xgb_Threshold = self.optimize_threshold(label=Y, score=score)
            self.train[score_name] = score
            pred = (score >= self.xgb_Threshold).astype('int')
            self.train[score_name.replace("score","pred")] = pred
        else :
            self.test[score_name] = score
            pred = (score >= self.xgb_Threshold).astype('int')
            self.test[score_name.replace("score","pred")] = pred
        
        print(f"Threshold : {self.xgb_Threshold}")
        print(f"F1-SCORE : {f1_score(Y, pred)}\n")
        print(classification_report(Y, pred))
        print("="*100)  

    def LOGIS(self,
              X, Y,
              training:bool=True,
              penalty:str='l2',
              C:float=100,
              solver:str='lbfgs',
              max_iter:int=1000,
              warm_start:bool=False,
              l1_latio:float=None
              ) :
        
        print(f"Logistic Regression is Started!\n")
        if training==True :    
            # Model Training
            if penalty == 'l2' :
                model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter, warm_start=warm_start, n_jobs=-1)    
            else :
                model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter, warm_start=warm_start, l1_latio=l1_latio, n_jobs=-1)
            model.fit(X, Y)
            with gzip.open(f'{self.save_dir}/LOG.pickle','wb') as f:
                pickle.dump(model, f)

        # Model Load
        with gzip.open(f'{self.save_dir}/LOG.pickle','rb') as f:
            model = pickle.load(f)

        score = model.predict_proba(X)[:,1]
        if training == True :
            self.xgb_Threshold = self.optimize_threshold(label=Y, score=score)
            self.train['LOG_S'] = score
            # pred = (score >= self.xgb_Threshold).astype('int')
            pred = model.predict(X)
            self.train['LOGIS'] = pred
        else :
            self.test['LOG_S'] = score
            # pred = (score >= self.xgb_Threshold).astype('int')
            pred = model.predict(X)
            self.test['LOGIS'] = pred
        
        print(f"Threshold : {self.xgb_Threshold}")
        print(f"F1-SCORE : {f1_score(Y, pred)}\n")
        print(classification_report(Y, pred))
        print("="*100)     



if __name__ == "__main__" :
    
    auto = autoModel(trainPath='./background/results/modeling/train.csv',
                     testPath='./background/results/modeling/test.csv',
                     scaling_cols=['TEMP', 'FLAME', 'SMOKE'],
                     gpu_option=True,
                     gpu_id="3,4,5",
                     name='test', 
                     tuning=False, 
                     timeLen=4)

    train, test = auto.dataset()
    var_cols = ['TEMP', 'FLAME', 'SMOKE', 'CODE1', 'CODE2', 'CODE4', 'CODE16', 'STATUS', 'pred_01', 'pred_02', 'pred_03', 'pred_04', 'pred_05']
    
    X_train, Y_train = train[var_cols], train['LABEL']
    X_test, Y_test = test[var_cols], test['LABEL']
    
    auto.Qauntile_RF(X=X_train, Y=Y_train, training=True, quantiles=[0.01,0.05,0.5,0.95,0.99])
    auto.Qauntile_RF(X=X_test, Y=Y_test, training=False, quantiles=[0.01,0.05,0.5,0.95,0.99])
    
    auto.XGBR(X=X_train, Y=Y_train, training=True, booster='gbtree')
    auto.XGBR(X=X_test, Y=Y_test, training=False, booster='gbtree')

    auto.XGBR(X=X_train, Y=Y_train, training=True, booster='dart')
    auto.XGBR(X=X_test, Y=Y_test, training=False, booster='dart')

    auto.XGBR(X=X_train, Y=Y_train, training=True, booster='dart', grow_policy='lossguide', drop_rate=0.1)
    auto.XGBR(X=X_test, Y=Y_test, training=False, booster='dart', grow_policy='lossguide', drop_rate=0.1)

    auto.XGBC(X=X_train, Y=Y_train, training=True, booster='gbtree')
    auto.XGBC(X=X_test, Y=Y_test, training=False, booster='gbtree')

    auto.XGBC(X=X_train, Y=Y_train, training=True, booster='dart')
    auto.XGBC(X=X_test, Y=Y_test, training=False, booster='dart')

    auto.XGBC(X=X_train, Y=Y_train, training=True, booster='dart', grow_policy='lossguide', drop_rate=0.1)
    auto.XGBC(X=X_test, Y=Y_test, training=False, booster='dart', grow_policy='lossguide', drop_rate=0.1)

    auto.LOGIS(X=X_train, Y=Y_train, C=0.01, training=True)
    auto.LOGIS(X=X_test, Y=Y_test, C=0.01, training=False)

    train, test = auto.dataset()

    print(train)
    print(test)