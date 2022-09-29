# Libararies Import
from tqdm import tqdm
from datetime import datetime

import os, warnings, pickle, gzip

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRFClassifier, XGBRFRegressor, XGBRegressor, XGBClassifier

import optuna

warnings.filterwarnings("ignore")

class autoModel :

    def __init__(self,
                 trainPath, 
                 testPath,
                 scaling_cols:list=None,
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

        if '.csv' in trainPath :    
            self.train = pd.read_csv(trainPath)
            self.test = pd.read_csv(testPath)
        elif '.parquet' in trainPath :
            self.train = pd.read_parquet(trainPath)
            self.test = pd.read_parquet(testPath)
        else :
            raise "지원하지 않는 데이터 타입입니다. csv나 parquet을 지원합니다."
        self.train.INST_DATE = pd.to_datetime(self.train.INST_DATE)
        self.test.INST_DATE = pd.to_datetime(self.test.INST_DATE)
        if scaling_cols is not None :
            self.scaling(target_columns=scaling_cols)

        self.timeLen = timeLen
        if self.timeLen > 1 :
            print("="*100)
            print("Time Series Preprocessing ...")
            scores = ['QRF_S', 'XGBRF_S', 'DART_S', 'LGBM_S', 'LOGIS_S']
            self.train = self.TimeSeriesInference(data=self.train, scores=scores)
            self.test = self.TimeSeriesInference(data=self.test, scores=scores)
            print("done!")
            print("="*100)


    def tuning_sklearn(self, 
                       X, Y,
                       classifiers:list=['SVC', 'RF', 'LR'],
                       n_trials:int=100) :
        self.X = X
        self.Y = Y
        self.classifiers = classifiers

        def objective(trial) :

            classifier_name = trial.suggest_categorical('classifier', self.classifiers)

            if classifier_name == 'SVC' :
                svc_params = {'C' : trial.suggest_float('C', 1e-3, 1e+3, step=0.01),
                            'kernel' : trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                            'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01),
                            'class_weight' : trial.suggest_categorical('class_weight', [None, 'balanced'])
                }
                classifier_obj = sklearn.svm.SVC(**svc_params)

            elif classifier_name == 'RF' :
                rf_params = {'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                            'min_weight_fraction_leaf' : trial.suggest_float('min_weight_fraction_leaf', 1e-3, 0.5),
                            'min_samples_split' : trial.suggest_float('min_samples_split', 1e-2, 1),
                            'min_samples_leaf' : trial.suggest_float('min_samples_leaf', 1, 1e+3),
                            'min_impurity_decrease' : trial.suggest_float('min_impurity_decrease', 0, 1e-2),
                            'max_leaf_nodes' : trial.suggest_categorical('max_leaf_nodes', [10, 15, 20, 25, 30, 35, 40, 45, 50, None]),
                            'max_depth' : trial.suggest_categorical('max_depth', [ 2, 4, 6, 8, 10, 20]),
                            'warm_start' : trial.suggest_categorical('warm_start', [False, True])
                }
                classifier_obj = RandomForestClassifier(**rf_params)

            elif classifier_name == 'LR' :
                lr_penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
                if lr_penalty == 'l1' :
                    lr_params = {'C' : trial.suggest_float('C', 1e-3, 1e+3, step=0.01),
                                 'max_iter' : trial.suggest_int('max_iter', 1e+2, 1e+3, step=100),
                                 'class_weight' : trial.suggest_categorical('class_weight', [None, 'balanced'])
                                }
                    classifier_obj = LogisticRegression(penalty=lr_penalty, solver='liblinear', **lr_params)    
                elif lr_penalty == 'elasticnet' :
                        
                    lr_params = {'l1_ratio' : trial.suggest_float('l1_ratio', 0, 1),
                                 'C' : trial.suggest_float('C', 1e-3, 1e+3, step=0.01),
                                 'max_iter' : trial.suggest_int('max_iter', 1e+2, 1e+3, 100),
                                 'class_weight' : trial.suggest_categorical('class_weight', [None, 'balanced'])
                    }
                    classifier_obj = LogisticRegression(penalty=lr_penalty, solver='saga', **lr_params)
                else :
                    lr_params = {'solver' : trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
                                 'C' : trial.suggest_float('C', 1e-3, 1e+3, step=0.01),
                                 'max_iter' : trial.suggest_int('max_iter', 1e+2, 1e+3, 100),
                                 'class_weight' : trial.suggest_categorical('class_weight', [None, 'balanced'])
                    }
                    classifier_obj = LogisticRegression(penalty=lr_penalty, **lr_params)         
            
            scores = sklearn.model_selection.cross_val_score(classifier_obj, self.X, self.Y, scoring='f1', n_jobs=-1, cv=3)
            score = scores.mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        print(study.best_trial)
        


    def tuning_xgbc(self, 
                   X, Y,
                   classifiers:list=['XGBRF', 'XGB'],
                   n_trials:int=100) :
        
        self.X = X
        self.Y = Y
        self.classifiers = classifiers

        def objective(trial) :

            classifier_name = trial.suggest_categorical('classifier', self.classifiers)

            if classifier_name == 'XGBRF' :
                grow_param = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                if grow_param == 'depthwise' :
                    booster_param = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
                    if booster_param == 'dart' :   
                            
                        xgbrf_params = {'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                                        'max_depth' : trial.suggest_categorical('max_depth', [ 2, 4, 6, 8, 10, 20]),
                                        'learning_rate' : trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                                        'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01), 
                                        'drop_rate' : trial.suggest_float('drop_rate', 0.1, 0.9, step=0.1),
                                        'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
                                        'subsample' : trial.suggest_float('subsample', 0.5, 1, step=0.1),
                                        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1, step=0.1),
                                        'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0.3, 1, step=0.1),
                                        'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                                        'reg_alpha' : trial.suggest_categorical('reg_alpha ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                        'reg_lambda' : trial.suggest_categorical('reg_lambda ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                        'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                        }
                        classifier_obj = XGBRFClassifier(grow_policy=grow_param, booster=booster_param, tree_method = 'gpu_hist', predictor='gpu_predictor',  **xgbrf_params)
                    else :
                        xgbrf_params = {'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                                        'max_depth' : trial.suggest_categorical('max_depth', [ 2, 4, 6, 8, 10, 20]),
                                        'learning_rate' : trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                                        'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01),
                                        'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
                                        'subsample' : trial.suggest_float('subsample', 0.5, 1, step=0.1),
                                        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1, step=0.1),
                                        'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0.3, 1, step=0.1),
                                        'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                                        'reg_alpha' : trial.suggest_categorical('reg_alpha ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                        'reg_lambda' : trial.suggest_categorical('reg_lambda ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                        'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                        }
                        classifier_obj = XGBRFClassifier(grow_policy=grow_param, booster=booster_param, tree_method = 'gpu_hist', predictor='gpu_predictor',  **xgbrf_params)
                
                elif grow_param == 'lossguide' : # lossguide

                    xgbrf_params = {'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                                    'max_depth' : trial.suggest_categorical('max_depth', [ 2, 4, 6, 8, 10, 20]),
                                    'max_leaves' : trial.suggest_categorical('max_leaves', [0, 2, 4, 6, 8, 10, 20, 25]),
                                    'learning_rate' : trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                                    'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01),
                                    'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
                                    'subsample' : trial.suggest_float('subsample', 0.5, 1, step=0.1),
                                    'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1, step=0.1),
                                    'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0.3, 1, step=0.1),
                                    'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                                    'reg_alpha' : trial.suggest_categorical('reg_alpha ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                    'reg_lambda' : trial.suggest_categorical('reg_lambda ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                    'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                    }
                    classifier_obj = XGBRFClassifier(grow_policy=grow_param, tree_method = 'gpu_hist', predictor='gpu_predictor',  **xgbrf_params)      
        
            elif classifier_name == 'XGB' :
                grow_param = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                if grow_param == 'depthwise' :
                    booster_param = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])
                    if booster_param == 'dart' :   
                            
                        xgb_params = {'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                                      'num_parallel_tree' : trial.suggest_categorical('num_parallel_tree', [10, 50, 100, 200, 300, 500]),
                                      'max_depth' : trial.suggest_categorical('max_depth', [ 2, 4, 6, 8, 10, 20]),
                                      'learning_rate' : trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                                      'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01),
                                      'drop_rate' : trial.suggest_float('drop_rate', 0.1, 0.9, step=0.1),
                                      'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
                                      'subsample' : trial.suggest_float('subsample', 0.5, 1, step=0.1),
                                      'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1, step=0.1),
                                      'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0.3, 1, step=0.1),
                                      'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                                      'reg_alpha' : trial.suggest_categorical('reg_alpha ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                      'reg_lambda' : trial.suggest_categorical('reg_lambda ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                      'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                        }
                        classifier_obj = XGBClassifier(grow_policy=grow_param, booster=booster_param, tree_method = 'gpu_hist', predictor='gpu_predictor',  **xgb_params)
                    else :
                        xgb_params = {'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                                      'num_parallel_tree' : trial.suggest_categorical('num_parallel_tree', [10, 50, 100, 200, 300, 500]),
                                      'max_depth' : trial.suggest_categorical('max_depth', [ 2, 4, 6, 8, 10, 20]),
                                      'learning_rate' : trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                                      'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01),
                                      'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
                                      'subsample' : trial.suggest_float('subsample', 0.5, 1, step=0.1),
                                      'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1, step=0.1),
                                      'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0.3, 1, step=0.1),
                                      'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                                      'reg_alpha' : trial.suggest_categorical('reg_alpha ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                      'reg_lambda' : trial.suggest_categorical('reg_lambda ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                      'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                        }
                        classifier_obj = XGBClassifier(grow_policy=grow_param, booster=booster_param, tree_method = 'gpu_hist', predictor='gpu_predictor',  **xgb_params)
                
                elif grow_param == 'lossguide' : # lossguide
                
                    xgb_params = {'n_estimators' : trial.suggest_int('n_estimators', 100, 500, step=100),
                                    'num_parallel_tree' : trial.suggest_categorical('num_parallel_tree', [10, 50, 100, 200, 300, 500]),
                                    'max_depth' : trial.suggest_categorical('max_depth', [ 2, 4, 6, 8, 10, 20]),
                                    'max_leaves' : trial.suggest_categorical('max_leaves', [0, 2, 4, 6, 8, 10, 20, 25]),
                                    'learning_rate' : trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                                    'gamma' : trial.suggest_float('gamma', 0, 0.5, step=0.01),
                                    'min_child_weight' : trial.suggest_int('min_child_weight', 0, 10),
                                    'subsample' : trial.suggest_float('subsample', 0.5, 1, step=0.1),
                                    'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.3, 1, step=0.1),
                                    'colsample_bylevel' : trial.suggest_float('colsample_bylevel', 0.3, 1, step=0.1),
                                    'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                                    'reg_alpha' : trial.suggest_categorical('reg_alpha ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                    'reg_lambda' : trial.suggest_categorical('reg_lambda ', [0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100] ),
                                    'colsample_bynode' : trial.suggest_float('colsample_bynode', 0.3, 1, step=0.1),
                    }
                    classifier_obj = XGBClassifier(grow_policy=grow_param, tree_method = 'gpu_hist', predictor='gpu_predictor',  **xgb_params) 

            
            scores = sklearn.model_selection.cross_val_score(classifier_obj, self.X, self.Y, scoring='f1', n_jobs=-1, cv=3)
            score = scores.mean()
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        print(study.best_trial)   
            


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
        from sklearn.metrics import precision_recall_curve

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
              n_estimators:int=200,
              scale_pos_weight:float=1,
              tree_method:str='gpu_hist',
              predictor:str='gpu_predictor'
              ) :
        
        if training == True :
                
            if modelType.lower() == 'reg' :
                model = XGBRFRegressor(n_estimators=n_estimators, n_jobs=-1, scale_pos_weight=scale_pos_weight, tree_method = tree_method, predictor=predictor)
            else :
                model = XGBRFClassifier(n_estimators=n_estimators, n_jobs=-1, scale_pos_weight=scale_pos_weight, tree_method = tree_method, predictor=predictor)
            
            model.fit(X, Y)

            with gzip.open(f'{self.save_dir}/XGBRF.pickle','wb') as f:
                pickle.dump(model, f)

        with gzip.open(f'{self.save_dir}/XGBRF.pickle','rb') as f:
            model = pickle.load(f)

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
        print(classification_report(Y, pred))
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
                score_name = 'DART_S'
            elif booster == 'gbtree' :
                score_name = 'XGB_S'
        else :
            score_name = 'LGBM_S'

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
                score_name = 'DART_S'
            elif booster == 'gbtree' :
                score_name = 'XGB_S'
        else :
            score_name = 'LGBM_S'

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

                self.XGBR(X=X_train, Y=Y_train, training=True, booster='dart', drop_rate=0.5)
                self.XGBR(X=X_test, Y=Y_test, training=False, booster='dart', drop_rate=0.5)

                self.XGBR(X=X_train, Y=Y_train, training=True, grow_policy='lossguide')
                self.XGBR(X=X_test, Y=Y_test, training=False, grow_policy='lossguide')

                self.XGBC(X=X_train, Y=Y_train, training=True, booster='gbtree')
                self.XGBC(X=X_test, Y=Y_test, training=False, booster='gbtree')

                self.XGBC(X=X_train, Y=Y_train, training=True, booster='dart', drop_rate=0.5)
                self.XGBC(X=X_test, Y=Y_test, training=False, booster='dart', drop_rate=0.5)

                self.XGBC(X=X_train, Y=Y_train, training=True, grow_policy='lossguide')
                self.XGBC(X=X_test, Y=Y_test, training=False, grow_policy='lossguide')

                self.LOGIS(X=X_train, Y=Y_train, C=100, training=True)
                self.LOGIS(X=X_test, Y=Y_test, C=100, training=False)
            else :
                self.Qauntile_RF(X=X_train, Y=Y_train, training=True, quantiles=[0.01,0.05,0.5,0.95,0.99])
                
                self.XGBR(X=X_train, Y=Y_train, training=True, booster='gbtree')

                self.XGBR(X=X_train, Y=Y_train, training=True, booster='dart', drop_rate=0.5)

                self.XGBR(X=X_train, Y=Y_train, training=True, grow_policy='lossguide')

                self.XGBC(X=X_train, Y=Y_train, training=True, booster='gbtree')

                self.XGBC(X=X_train, Y=Y_train, training=True, booster='dart', drop_rate=0.5)

                self.XGBC(X=X_train, Y=Y_train, training=True, grow_policy='lossguide')

                self.LOGIS(X=X_train, Y=Y_train, C=100, training=True)
