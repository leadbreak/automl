# Libararies Import
from tqdm import tqdm
from datetime import datetime

import os, warnings, pickle, gzip

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBRFClassifier, XGBRFRegressor, XGBRegressor, XGBClassifier

import optuna
import dask

warnings.filterwarnings("ignore")

class autoModel :

    
    def __init__(self,
                #  trainPath, 
                #  testPath,
                 scaling_cols:list=None,
                 gpu_option: bool=False,
                 gpu_id : str=None,
                 name:str=datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(" ","_").replace("-","").replace(":",""),
                 timeLen:int=1) :

        if gpu_option == True :
            from tensorflow.python.client import device_lib
            
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]= gpu_id
            
            print()
            print("="*100)
            gpu_num = len(device_lib.list_local_devices()[1:])
            print(f"\n사용할 gpu 수 : {gpu_num}\n")
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

        # if '.csv' in trainPath :    
        #     self.train = pd.read_csv(trainPath).reset_index(drop=True)
        #     self.test = pd.read_csv(testPath).reset_index(drop=True)
        # elif '.parquet' in trainPath :
        #     self.train = pd.read_parquet(trainPath).reset_index(drop=True)
        #     self.test = pd.read_parquet(testPath).reset_index(drop=True)
        # else : # 바로 파일로 입력한 경우
        #     self.train = trainPath
        #     self.test = testPath
            
        # self.train.INST_DATE = pd.to_datetime(self.train.INST_DATE)
        # self.test.INST_DATE = pd.to_datetime(self.test.INST_DATE)
        # if scaling_cols is not None :
        #     self.scaling(target_columns=scaling_cols)

        # self.timeLen = timeLen
        # if self.timeLen > 1 :
        #     print("="*100)
        #     print("Time Series Preprocessing ...")
        #     scores = [x for x in self.train.columns if "_S" in x]
        #     self.train = self.TimeSeriesInference(data=self.train, scores=scores)
        #     self.test = self.TimeSeriesInference(data=self.test, scores=scores)
        #     print("done!")
        #     print("="*100)

    
    def tuning_sklearn(self, 
                       trainX, trainY,
                       testX=None, testY=None,
                       classifiers:list=['SVC', 'RF', 'LR'],
                       n_trials:int=100,
                       cv:int=1) :
        self.X = trainX
        self.Y = trainY
        if testX is not None :
            self.testX = testX
            self.testY = testY
        else :
            self.testX = trainX
            self.testY = trainY
        self.classifiers = classifiers

        
        def objective(trial) :

            classifier_name = trial.suggest_categorical('classifier', self.classifiers)

            if classifier_name == 'SVC' :
                svc_params = {'C' : trial.suggest_float('C', 1e-3, 1e+3, step=0.01),
                            'kernel' : trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                            'gamma' : trial.suggest_categorical('gamma', ['scale', 'auto']),
                            'class_weight' : trial.suggest_categorical('class_weight', [None, 'balanced'])
                }
                classifier_obj = sklearn.svm.SVC(**svc_params)

            elif classifier_name == 'RF' :
                rf_params = {'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                             'min_samples_split' : trial.suggest_int('min_samples_split', 2, 1e+2),
                             'min_samples_leaf' : trial.suggest_float('min_samples_leaf', 1, 1e+3),
                             'min_impurity_decrease' : trial.suggest_float('min_impurity_decrease', 0, 1e-2),
                             'max_leaf_nodes' : trial.suggest_categorical('max_leaf_nodes', [10, 15, 20, 25, 30, 35, 40, 45, 50, None]),
                             'max_depth' : trial.suggest_categorical('max_depth', [ 2, 4, 6, 8, 10, 20, None]),
                             'max_features' : trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                             'warm_start' : trial.suggest_categorical('warm_start', [False, True]),
                            #  'class_weight' : trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
                }
                classifier_obj = RandomForestRegressor(**rf_params)

            elif classifier_name == 'LR' :
                lr_params = {'C' : trial.suggest_float('C', 1e-3, 1e+3, step=0.01),
                             'max_iter' : trial.suggest_int('max_iter', 1e+2, 1e+3, step=100),
                             'class_weight' : trial.suggest_categorical('class_weight', [None, 'balanced'])
                            }
                lr_penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])

                if lr_penalty == 'l1' :
                    
                    lr_solver = 'liblinear'

                elif lr_penalty == 'elasticnet' :
                    lr_solver = 'saga'
                    lr_params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1, step=0.001)

                else :
                    lr_solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
                    
                classifier_obj = LogisticRegression(solver=lr_solver, **lr_params)         
            
            if cv == 1 :
                if classifier_name == 'RF' :
                    pred = (classifier_obj.fit(self.X, self.Y).predict(self.testX) > 0.5).astype('int')
                else :
                    pred = classifier_obj.fit(self.X, self.Y).predict(self.testX)
                score = f1_score(self.testY, pred)
            else :
                if classifier_name == 'RF' :
                    raise "RF Model is based on Regression, please select other models or cv = 1"
                scores = sklearn.model_selection.cross_val_score(classifier_obj, self.X, self.Y, scoring='f1', n_jobs=-1, cv=cv)
                score = scores.mean()
            return score

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
                                    study_name=f'{self.classifiers}_parameter_opt',                                    
                                    direction="maximize", 
                                    sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        print("Best Score:", study.best_value)
        print("Best trial:", study.best_trial.params)

        return study  
        

    
    def tuning_xgbc(self, 
                   trainX, trainY,
                   testX=None, testY=None,
                   n_trials:int=100) :
        
        self.X = trainX
        self.Y = trainY
        if testX is not None :
            self.testX = testX
            self.testY = testY
        else :
            self.testX = trainX
            self.testY = trainY

        
        def objective(trial) :

            params = {'grow_policy' : trial.suggest_categorical('grow_policy', ['depthwise']),
                      'objective' : trial.suggest_categorical('objective', ['binary:logistic']),
                      'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                      'num_parallel_tree' : trial.suggest_categorical('num_parallel_tree', [1, 3, 5, 10, 50, 100, 200]),
                      'max_depth' : trial.suggest_categorical('max_depth', [2, 4, 6, 8, 10, 20]),
                      'learning_rate' : trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                      'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01), 
                      'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
                      'subsample' : trial.suggest_float('subsample', 0.5, 1, step=0.1),
                      'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1, step=0.1),
                      'reg_alpha' : trial.suggest_categorical('reg_alpha', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                      'reg_lambda' : trial.suggest_categorical('reg_lambda', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                      'max_delta_step' : trial.suggest_float('max_delta_step', 0, 10, step=0.1),
                      'tree_method' : trial.suggest_categorical('tree_method', ['gpu_hist']), 
                      'predictor' : trial.suggest_categorical('predictor', ['gpu_predictor']), 
                      'n_jobs' : trial.suggest_categorical('n_jobs', [-1]), 
                        }
    
            booster_param = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
            
            if booster_param == 'dart' :   
                params['rate_drop'] = trial.suggest_float('rate_drop', 0.1, 0.9, step=0.1)
                params['skip_drop'] = trial.suggest_float('skip_drop', 0, 0.5, step=0.1)
            

            classifier_obj = XGBClassifier(booster=booster_param, **params)     
        
            # scores = sklearn.model_selection.cross_val_score(classifier_obj, self.X, self.Y, scoring='f1', n_jobs=-1, cv=3)
            pred = classifier_obj.fit(self.X, self.Y).predict(self.testX)
            score = f1_score(self.testY, pred)
            return score

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
                                    study_name='xgbc_parameter_opt',                                    
                                    direction="maximize", 
                                    sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        print("Best Score:", study.best_value)
        print("Best trial:", study.best_trial.params) 

        return study
            
    
    def tuning_xgbr(self, 
                   trainX, trainY,
                   testX=None, testY=None,
                   n_trials:int=100) :
        
        self.X = trainX
        self.Y = trainY
        if testX is not None :
            self.testX = testX
            self.testY = testY
        else :
            self.testX = trainX
            self.testY = trainY

        
        def objective(trial) :

            params = {'grow_policy' : trial.suggest_categorical('grow_policy', ['depthwise']),
                      'objective' : trial.suggest_categorical('objective', ['reg:squarederror']),
                      'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                      'num_parallel_tree' : trial.suggest_categorical('num_parallel_tree', [1, 3, 5, 10, 50, 100, 200]),
                      'max_depth' : trial.suggest_categorical('max_depth', [2, 4, 6, 8, 10, 20]),
                      'learning_rate' : trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                      'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01), 
                      'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
                      'subsample' : trial.suggest_float('subsample', 0.5, 1, step=0.1),
                      'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1, step=0.1),
                      'reg_alpha' : trial.suggest_categorical('reg_alpha', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                      'reg_lambda' : trial.suggest_categorical('reg_lambda', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                      'max_delta_step' : trial.suggest_float('max_delta_step', 0, 10, step=0.1),
                      'tree_method' : trial.suggest_categorical('tree_method', ['gpu_hist']), 
                      'predictor' : trial.suggest_categorical('predictor', ['gpu_predictor']), 
                      'n_jobs' : trial.suggest_categorical('n_jobs', [-1]), 
                        }
    
            booster_param = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
            
            if booster_param == 'dart' :   
                params['rate_drop'] = trial.suggest_float('rate_drop', 0.1, 0.9, step=0.1)
                params['skip_drop'] = trial.suggest_float('skip_drop', 0, 0.5, step=0.1)

            classifier_obj = XGBRegressor(booster=booster_param, **params)     
        
            # scores = sklearn.model_selection.cross_val_score(classifier_obj, self.X, self.Y, scoring='f1', n_jobs=-1, cv=3)
            pred = (classifier_obj.fit(self.X, self.Y).predict(self.testX) >= 0.5).astype('int')
            score = f1_score(self.testY, pred)
            return score

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
                                    study_name='xgbr_parameter_opt',                                    
                                    direction="maximize", 
                                    sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        print("Best Score:", study.best_value)
        print("Best trial:", study.best_trial.params) 
        
        return study

    
    def tuning_lgbc(self, 
                   trainX, trainY,
                   testX=None, testY=None,
                   n_trials:int=100) :
        
        self.X = trainX
        self.Y = trainY
        if testX is not None :
            self.testX = testX
            self.testY = testY
        else :
            self.testX = trainX
            self.testY = trainY

        
        def objective(trial) :

            params = {'grow_policy' : trial.suggest_categorical('grow_policy', ['lossguide']),
                      'objective' : trial.suggest_categorical('objective', ['binary:logistic']),
                      'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                      'max_depth' : trial.suggest_categorical('max_depth', [6, 8, 10, 20, 25, 30]),
                      'num_parallel_tree' : trial.suggest_categorical('num_parallel_tree', [1, 3, 5, 10, 50, 100, 200]),
                      'learning_rate' : trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                      'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01), 
                      'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
                      'subsample' : trial.suggest_float('subsample', 0.5, 1, step=0.1),
                      'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1, step=0.1),
                      'reg_alpha' : trial.suggest_categorical('reg_alpha', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                      'reg_lambda' : trial.suggest_categorical('reg_lambda', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                      'max_delta_step' : trial.suggest_float('max_delta_step', 0, 10, step=0.1),
                      'tree_method' : trial.suggest_categorical('tree_method', ['gpu_hist']), 
                      'predictor' : trial.suggest_categorical('predictor', ['gpu_predictor']), 
                      'n_jobs' : trial.suggest_categorical('n_jobs', [-1]), 
                        }
                
            classifier_obj = XGBClassifier(**params)     
        
            # scores = sklearn.model_selection.cross_val_score(classifier_obj, self.X, self.Y, scoring='f1', n_jobs=-1, cv=3)
            pred = classifier_obj.fit(self.X, self.Y).predict(self.testX)
            score = f1_score(self.testY, pred)
            return score

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
                                    study_name='lgbmc_parameter_opt',                                    
                                    direction="maximize", 
                                    sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        print("Best Score:", study.best_value)
        print("Best trial:", study.best_trial.params)

        return study

    
    def tuning_lgbr(self, 
                   trainX, trainY,
                   testX=None, testY=None,
                   n_trials:int=100) :
        
        self.X = trainX
        self.Y = trainY
        if testX is not None :
            self.testX = testX
            self.testY = testY
        else :
            self.testX = trainX
            self.testY = trainY

        
        def objective(trial) :

            params = {'grow_policy' : trial.suggest_categorical('grow_policy', ['lossguide']),
                      'objective' : trial.suggest_categorical('objective', ['reg:squarederror']),
                      'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                      'max_depth' : trial.suggest_categorical('max_depth', [6, 8, 10, 20, 25, 30]),
                      'num_parallel_tree' : trial.suggest_categorical('num_parallel_tree', [1, 3, 5, 10, 50, 100, 200]),
                      'learning_rate' : trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                      'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01), 
                      'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
                      'subsample' : trial.suggest_float('subsample', 0.5, 1, step=0.1),
                      'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1, step=0.1),
                      'reg_alpha' : trial.suggest_categorical('reg_alpha', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                      'reg_lambda' : trial.suggest_categorical('reg_lambda', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                      'max_delta_step' : trial.suggest_float('max_delta_step', 0, 10, step=0.1),
                      'tree_method' : trial.suggest_categorical('tree_method', ['gpu_hist']), 
                      'predictor' : trial.suggest_categorical('predictor', ['gpu_predictor']), 
                      'n_jobs' : trial.suggest_categorical('n_jobs', [-1]), 
                        }
                
            classifier_obj = XGBRegressor(**params)    
        
            # scores = sklearn.model_selection.cross_val_score(classifier_obj, self.X, self.Y, scoring='f1', n_jobs=-1, cv=3)
            pred = (classifier_obj.fit(self.X, self.Y).predict(self.testX) >= 0.5).astype('int')
            score = f1_score(self.testY, pred)
            return score

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
                                    study_name='lgbmr_parameter_opt',                                    
                                    direction="maximize", 
                                    sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        print("Best Score:", study.best_value)
        print("Best trial:", study.best_trial.params)

        return study

    def dataset(self) :
        return self.train, self.test

    
    def code_dataset(self, df) :
        data = df.copy()

        # one-hot encoding
        data['BINARY_CODE'] = data['CODE'].map(lambda k:format(k, '05b'))
        data['CODE16'] = data['BINARY_CODE'].map(lambda k:int(k[0]))
        data['CODE4'] = data['BINARY_CODE'].map(lambda k:int(k[2]))
        data['CODE2'] = data['BINARY_CODE'].map(lambda k:int(k[3]))
        data['CODE1'] = data['BINARY_CODE'].map(lambda k:int(k[4]))

        # convert INST_DATE to datetime
        data['INST_DATE'] = pd.to_datetime(data.INST_DATE)

        data.drop(columns=['CODE', 'BINARY_CODE'], inplace=True)

        df = data.reset_index(drop=True).copy()

        return df

    
    def scaling(self, target_columns) :
        scaler = StandardScaler()
        self.train[target_columns] = scaler.fit_transform(self.train[target_columns])
        self.test[target_columns] = scaler.transform(self.test[target_columns])

        with gzip.open(f"{self.save_dir}/scaler.pickle", "wb") as f :
            pickle.dump(scaler, f)
        
        with gzip.open(f'{self.save_dir}/scaler.pickle','rb') as f:
            scaler = pickle.load(f)

    
    def TimeSeriesInference(self, data, scores:list) :
        
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
        from sklearn.metrics import precision_recall_curve
        print("="*100)
        print(f"\nThreshold Optimizing is started...")

        precision, recall, threshold = precision_recall_curve(label , score)
        best_cnt_dic = abs(precision - recall)
        threshold_fixed = threshold[np.argmin(best_cnt_dic)]
        recall_score = recall[np.argmin(best_cnt_dic)] 
        precision_score = precision[np.argmin(best_cnt_dic)]
        print(f"Recall Score : {recall_score}")
        print(f"Precision Score : {precision_score}")
        
        if (recall_score <= 0.1) or (precision_score <= 0.1) :            
            print("\nThreshold 최적화를 위한 모델 성능이 낮습니다.")
            print("Threshold를 초기화합니다.\n")
            threshold_fixed = 0.5

        print("Threshold Optimizing is done!")
        print(f"Optimized Threshold : {threshold_fixed}\n")
        print("="*100)
        return threshold_fixed

    
    def Qauntile_RF(self, 
                    X, Y,
                    training=True, 
                    quantiles = [0.01, 0.05, 0.50, 0.95 , 0.99],
                    q_tuning=False,
                    n_estimators=200,
                    min_samples_split:float=2,
                    min_samples_leaf:float=1,
                    min_impurity_decrease:float=0,
                    max_leaf_nodes:int=None,
                    max_depth:int=None,
                    warm_start=True,  
                    scoring="f1",
                    ) :
        
        print("Quantile Regression with RF is Started!\n")
        if training==True :    
            # Model Training
            model = RandomForestRegressor(n_estimators=n_estimators, 
                                          min_samples_split=min_samples_split, 
                                          min_samples_leaf=min_samples_leaf,
                                          min_impurity_decrease=min_impurity_decrease,
                                          max_leaf_nodes=max_leaf_nodes,
                                          max_depth=max_depth,
                                          warm_start=warm_start, 
                                          n_jobs=-1,
                                          )
            model.fit(X, Y)

            # Model dump
            with gzip.open(f'{self.save_dir}/QRF.pickle','wb') as f:
                pickle.dump(model, f)
        
        # Model Load
        with gzip.open(f'{self.save_dir}/QRF.pickle','rb') as f:
            model = pickle.load(f)

        print(model.get_params())

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
        RF_actual_pred['actual'] = pd.Series(Y)
        RF_actual_pred['interval'] = RF_actual_pred[np.max(quantiles)] - RF_actual_pred[np.min(quantiles)]
        RF_actual_pred = RF_actual_pred.sort_values('interval')
        RF_actual_pred = RF_actual_pred.round(6)
        

        # Optimizing Threshold - to maximize recall and f1-score
        if q_tuning == True :
            self.quantile_tuning(X, Y, quantiles=quantiles, scoring=scoring)
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
        print(f"\nConfusion Matrix : \n{confusion_matrix(RF_actual_pred['actual'],RF_actual_pred['pred'])}\n\n")
        print(classification_report(RF_actual_pred['actual'],RF_actual_pred['pred']))
        print("="*100)

    
    def quantile_tuning(self,
                        X, Y,
                        quantiles,
                        scoring) :
        
        pass

    
    def XGBRF(self,
              X, Y,
              training:bool=True,
              modelType:str='reg',
              booster:str='gbtree', 
              n_estimators:int=200,
              scale_pos_weight:float=1,
              tree_method:str='gpu_hist',
              predictor:str='gpu_predictor',
              n_jobs:int=-1
              
              ) :
        
        if training == True :
                
            if modelType.lower() == 'reg' :
                model = XGBRFRegressor(booster=booster, 
                                       n_estimators=n_estimators, 
                                       scale_pos_weight=scale_pos_weight, 
                                       tree_method = tree_method, 
                                       predictor=predictor, 
                                       n_jobs=n_jobs)
            else :
                model = XGBRFClassifier(booster=booster, 
                                        n_estimators=n_estimators, 
                                        scale_pos_weight=scale_pos_weight, 
                                        tree_method = tree_method, 
                                        predictor=predictor, 
                                        n_jobs=n_jobs)
            
            model.fit(X, Y)

            with gzip.open(f'{self.save_dir}/XGBRF.pickle','wb') as f:
                pickle.dump(model, f)

        with gzip.open(f'{self.save_dir}/XGBRF.pickle','rb') as f:
            model = pickle.load(f)

        print(model.get_xgb_params())

        if modelType.lower() == 'reg' :
            score = model.predict(X)
        else :
            score = model.predict_proba(X)[:, 1]
        
        score_name = 'XGBRF_S'
        if training == True :
            self.xgbrf_Threshold = self.optimize_threshold(label=Y, score=score)
            self.train[score_name] = score
            pred = (score >= self.xgbrf_Threshold).astype('int')
            self.train[score_name.replace("_S","")] = pred
        else :
            self.test[score_name] = score
            pred = (score >= self.xgbrf_Threshold).astype('int')
            self.test[score_name.replace("_S","")] = pred

        print(f"Threshold : {self.xgbrf_Threshold}")
        print(f"F1-SCORE : {f1_score(Y, pred)}\n")
        print(f"\nConfusion Matrix : \n{confusion_matrix(Y,pred)}\n\n")
        print(classification_report(Y, pred))
        print("="*100)    

    
    def XGBR(self,
             X, Y,
             training:bool=True,
             objective:str='reg:squarederror',
             num_parallel_tree:int=1,
             max_depth:int=6,
             learning_rate:float=0.3,
             gamma:float=0.0,
             min_child_weight:float=0.0,
             colsample_bytree:float=1.0,
             reg_alpha:float=0.0,
             reg_lambda:float=1.0,
             max_delta_step:float=0.0,
             skip_drop:float=0.0,
             booster:str='gbtree',
             rate_drop:float=0.5,
             subsample:float=1,
             grow_policy:str='depthwise',
             n_estimators:int=200,
             scale_pos_weight:int=1,
             tree_method:str='gpu_hist',
             predictor:str='gpu_predictor',
             n_jobs:int=-1
             ) :

        print(f"XGB Regression with {booster} & {grow_policy} is Started!\n")
        if training==True :    
            # Model Training
            if (booster == 'dart') & (grow_policy=='depth_wise') :
                model = XGBRegressor( n_estimators=n_estimators,
                                      objective=objective,
                                      num_parallel_tree=num_parallel_tree,
                                      booster=booster, 
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      gamma=gamma,
                                      min_child_weight=min_child_weight,
                                      colsample_bytree=colsample_bytree,
                                      reg_alpha=reg_alpha,
                                      reg_lambda=reg_lambda,
                                      max_delta_step=max_delta_step,
                                      skip_drop=skip_drop,                                      
                                      rate_drop=rate_drop, 
                                      grow_policy=grow_policy, 
                                      subsample=subsample, 
                                      scale_pos_weight=scale_pos_weight, 
                                      tree_method=tree_method,                                       
                                      predictor=predictor,
                                      n_jobs=n_jobs
                                      )
            else :
                model = XGBRegressor( n_estimators=n_estimators, 
                                      objective=objective,
                                      num_parallel_tree=num_parallel_tree,
                                      booster=booster, 
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      gamma=gamma,
                                      min_child_weight=min_child_weight,
                                      colsample_bytree=colsample_bytree,
                                      reg_alpha=reg_alpha,
                                      reg_lambda=reg_lambda,
                                      max_delta_step=max_delta_step,
                                      grow_policy=grow_policy, 
                                      subsample=subsample, 
                                      scale_pos_weight=scale_pos_weight, 
                                      tree_method=tree_method,                                       
                                      predictor=predictor,
                                      n_jobs=n_jobs
                                      )
            model.fit(X, Y)
            with gzip.open(f'{self.save_dir}/XGB_{booster}_{grow_policy}.pickle','wb') as f:
                pickle.dump(model, f)

        # Model Load
        with gzip.open(f'{self.save_dir}/XGB_{booster}_{grow_policy}.pickle','rb') as f:
            model = pickle.load(f)

        print(model.get_xgb_params())

        score = model.predict(X)
        if grow_policy == 'depthwise' :
            if booster == 'dart' :
                score_name = 'DART_S'
            elif booster == 'gbtree' :
                score_name = 'XGB_S'
            else :
                score_name = 'XGBL_S'
        else :
            score_name = 'LGBM_S'
        
        if num_parallel_tree > 1 :
            score_name = 'XGBRF_S'

        if training == True :
            self.xgb_Threshold = self.optimize_threshold(label=Y, score=score)
            self.train[score_name] = score
            pred = (score >= self.xgb_Threshold).astype('int')
            self.train[score_name.replace("_S","")] = pred
        else :
            self.test[score_name] = score
            pred = (score >= self.xgb_Threshold).astype('int')
            self.test[score_name.replace("_S","")] = pred
        
        print(f"Threshold : {self.xgb_Threshold}")
        print(f"F1-SCORE : {f1_score(Y, pred)}\n")
        print(f"\nConfusion Matrix : \n{confusion_matrix(Y,pred)}\n\n")
        print(classification_report(Y, pred))
        print("="*100)        

    
    def XGBC(self,
             X, Y,
             training:bool=True,
             objective:str='binary:logistic',
             num_parallel_tree:int=1,
             max_depth:int=6,
             learning_rate:float=0.3,
             gamma:float=0.0,
             min_child_weight:float=0.0,
             colsample_bytree:float=1.0,
             reg_alpha:float=0.0,
             reg_lambda:float=1.0,
             max_delta_step:float=0.0,
             skip_drop:float=0.0,
             booster:str='gbtree',
             rate_drop:float=0.5,
             subsample:float=1,
             grow_policy:str='depthwise',
             n_estimators:int=200,
             scale_pos_weight:int=1,
             tree_method:str='gpu_hist',
             predictor:str='gpu_predictor',
             n_jobs:int=-1
             ) :

        print(f"XGB Classification with {booster} & {grow_policy} is Started!\n")
        if training==True :    
            # Model Training
            if (booster == 'dart') & (grow_policy=='depth_wise') :
                model = XGBClassifier(objective=objective,
                                      n_estimators=n_estimators, 
                                      num_parallel_tree=num_parallel_tree,
                                      booster=booster, 
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      gamma=gamma,
                                      min_child_weight=min_child_weight,
                                      colsample_bytree=colsample_bytree,
                                      reg_alpha=reg_alpha,
                                      reg_lambda=reg_lambda,
                                      max_delta_step=max_delta_step,
                                      skip_drop=skip_drop,                                      
                                      rate_drop=rate_drop, 
                                      grow_policy=grow_policy, 
                                      subsample=subsample, 
                                      scale_pos_weight=scale_pos_weight, 
                                      tree_method=tree_method,                                       
                                      predictor=predictor,
                                      n_jobs=n_jobs
                                      )
            else :
                model = XGBClassifier(objective=objective,
                                      n_estimators=n_estimators, 
                                      num_parallel_tree=num_parallel_tree,
                                      booster=booster, 
                                      max_depth=max_depth,
                                      learning_rate=learning_rate,
                                      gamma=gamma,
                                      min_child_weight=min_child_weight,
                                      colsample_bytree=colsample_bytree,
                                      reg_alpha=reg_alpha,
                                      reg_lambda=reg_lambda,
                                      max_delta_step=max_delta_step,
                                      grow_policy=grow_policy, 
                                      subsample=subsample, 
                                      scale_pos_weight=scale_pos_weight, 
                                      tree_method=tree_method,                                       
                                      predictor=predictor,
                                      n_jobs=n_jobs
                                      )
            model.fit(X, Y)
            with gzip.open(f'{self.save_dir}/XGB_{booster}_{grow_policy}.pickle','wb') as f:
                pickle.dump(model, f)

        # Model Load
        with gzip.open(f'{self.save_dir}/XGB_{booster}_{grow_policy}.pickle','rb') as f:
            model = pickle.load(f)

        print(model.get_xgb_params())

        score = model.predict_proba(X)[:,1]
        if grow_policy == 'depthwise' :
            if booster == 'dart' :
                score_name = 'DART_S'
            elif booster == 'gbtree' :
                score_name = 'XGB_S'
            else :
                score_name = 'XGBL_S'
        else :
            score_name = 'LGBM_S'
        
        if num_parallel_tree > 1 :
            score_name = 'XGBRF_S'

        if training == True :
            self.xgb_Threshold = self.optimize_threshold(label=Y, score=score)
            self.train[score_name] = score
            pred = (score >= self.xgb_Threshold).astype('int')
            self.train[score_name.replace("_S","")] = pred
        else :
            self.test[score_name] = score
            pred = (score >= self.xgb_Threshold).astype('int')
            self.test[score_name.replace("_S","")] = pred
        
        print(f"Threshold : {self.xgb_Threshold}")
        print(f"F1-SCORE : {f1_score(Y, pred)}\n")
        print(f"\nConfusion Matrix : \n{confusion_matrix(Y,pred)}\n\n")
        print(classification_report(Y, pred))
        print("="*100)  

    
    def LOGIS(self,
              X, Y,
              training:bool=True,
              penalty:str='l2',
              C:float=100,
              solver:str='lbfgs',
              max_iter:int=1000,
              class_weight=None,
              warm_start:bool=False,
              l1_ratio:float=None,
              n_jobs:int=-1
              ) :
        
        print(f"Logistic Regression is Started!\n")
        if training==True :    
            # Model Training
            if penalty == 'l2' :
                model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter, warm_start=warm_start, class_weight=class_weight, n_jobs=n_jobs)    
            else :
                model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter, warm_start=warm_start, l1_ratio=l1_ratio, class_weight=class_weight, n_jobs=n_jobs)
            model.fit(X, Y)
            with gzip.open(f'{self.save_dir}/LOG.pickle','wb') as f:
                pickle.dump(model, f)

        # Model Load
        with gzip.open(f'{self.save_dir}/LOG.pickle','rb') as f:
            model = pickle.load(f)

        print(model.get_params())

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
        print(f"\nConfusion Matrix : \n{confusion_matrix(Y,pred)}\n\n")
        print(classification_report(Y, pred))
        print("="*100)     

    
    def auto_run(self, 
                 X_train, Y_train,
                 X_test=None, Y_test=None,
                 runList:list=['all']) :
        
        if 'all' in runList :
            if X_test is not None :
                self.Qauntile_RF(X=X_train, Y=Y_train, training=True, quantiles=[0.01,0.05,0.5,0.95,0.99])
                self.Qauntile_RF(X=X_test, Y=Y_test, training=False, quantiles=[0.01,0.05,0.5,0.95,0.99])
                
                self.XGBRF(X=X_train, Y=Y_train, training=True, modelType='reg')
                self.XGBRF(X=X_test, Y=Y_test, training=False, modelType='reg')

                self.XGBRF(X=X_train, Y=Y_train, training=True, modelType='clf')
                self.XGBRF(X=X_test, Y=Y_test, training=False, modelType='clf')

                self.XGBR(X=X_train, Y=Y_train, training=True, booster='gbtree')
                self.XGBR(X=X_test, Y=Y_test, training=False, booster='gbtree')

                self.XGBR(X=X_train, Y=Y_train, training=True, booster='dart', rate_drop=0.5)
                self.XGBR(X=X_test, Y=Y_test, training=False, booster='dart', rate_drop=0.5)

                self.XGBR(X=X_train, Y=Y_train, training=True, grow_policy='lossguide')
                self.XGBR(X=X_test, Y=Y_test, training=False, grow_policy='lossguide')

                self.XGBC(X=X_train, Y=Y_train, training=True, booster='gbtree')
                self.XGBC(X=X_test, Y=Y_test, training=False, booster='gbtree')

                self.XGBC(X=X_train, Y=Y_train, training=True, booster='dart', rate_drop=0.5)
                self.XGBC(X=X_test, Y=Y_test, training=False, booster='dart', rate_drop=0.5)

                self.XGBC(X=X_train, Y=Y_train, training=True, grow_policy='lossguide')
                self.XGBC(X=X_test, Y=Y_test, training=False, grow_policy='lossguide')

                self.LOGIS(X=X_train, Y=Y_train, C=100, training=True)
                self.LOGIS(X=X_test, Y=Y_test, C=100, training=False)
            else :
                self.Qauntile_RF(X=X_train, Y=Y_train, training=True, quantiles=[0.01,0.05,0.5,0.95,0.99])
                
                self.XGBR(X=X_train, Y=Y_train, training=True, booster='gbtree')

                self.XGBR(X=X_train, Y=Y_train, training=True, booster='dart', rate_drop=0.5)

                self.XGBR(X=X_train, Y=Y_train, training=True, grow_policy='lossguide')

                self.XGBC(X=X_train, Y=Y_train, training=True, booster='gbtree')

                self.XGBC(X=X_train, Y=Y_train, training=True, booster='dart', rate_drop=0.5)

                self.XGBC(X=X_train, Y=Y_train, training=True, grow_policy='lossguide')

                self.LOGIS(X=X_train, Y=Y_train, C=100, training=True)
