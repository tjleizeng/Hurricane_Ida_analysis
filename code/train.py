import numpy as np
import pandas as pd
import geopandas as gpd
from types import SimpleNamespace
import pyproj
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pyproj
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

import argparse
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

utm = pyproj.CRS('EPSG:32618')

P = SimpleNamespace()
P.data = "../data"
CRS_DEG = 'EPSG:4326'
CRS_M = 'EPSG:3857'


# args: test_ratio, random_state, covid_factor
def get_arguments(argv):
      parser = argparse.ArgumentParser(description='Ida Prediction Models')
      args = parser.parse_args(argv)
      return args

def get_X_names(wave = 0, covid = False):
      X_names = ['DIST_IDA',\
                  'DIST_COAST', \
                   'URBAN',\
                   'AGE_MINOR', \
                   'AGE_SENIOR', \
                   'RACE_HISP', \
                   'RACE_AFAM', \
                   'RACE_ASIAN',\
                   'NO_EMPLOY', \
                   'NO_INSUR',\
                   'NO_HSDP', \
                   'NO_VEH',\
                   'POP_DISABLE', 'POP_POOR']

      if covid:
            X_names += [f'CASE_{wave}', f'VAC_{wave}']
      return X_names

if __name__ == '__main__':
      args = get_arguments(sys.argv[1:])
      args.ratio = 1
      # if os.path.exists(f'{P.data}/res_{args.ratio*100}_{args.covid}_{args.seed}.pkl'):
      #       print(f'Already done: {P.data}/res_{args.ratio*100}_{args.covid}_{args.seed}')
      #       continue
      # load data
      df_input = pd.read_parquet(P.data+"/df_input_merged.parquet")
      
      df_input['POP'] = df_input['e_totpop']
      df_input['CASE_0'] = df_input['Cases_wave0']
      df_input['CASE_1'] = df_input['Cases_wave1']
      df_input['CASE_2'] = df_input['Cases_wave2']
      df_input['CASE_3'] = df_input['Cases_wave3']

      df_input['RATIO_0'] = df_input['Positive_wave0']*100
      df_input['RATIO_1'] = df_input['Positive_wave1']*100
      df_input['RATIO_2'] = df_input['Positive_wave2']*100
      df_input['RATIO_3'] = df_input['Positive_wave3']*100

      df_input['VAC_0'] = df_input['Completed_wave0']
      df_input['VAC_1'] = df_input['Completed_wave1']
      df_input['VAC_2'] = df_input['Completed_wave2']
      df_input['VAC_3'] = df_input['Completed_wave3']

      df_input['POP_DENSITY'] = df_input['POP']/df_input['area_sqmi']
      # df_input['POP_DENSITY_LOG'] = np.log10(df_input['POP_DENSITY']+1)
      df_input['AGE_MINOR'] = df_input['ep_age17']
      df_input['AGE_SENIOR'] = df_input['ep_age65']
      df_input['RACE_HISP'] = df_input['ep_hisp']
      df_input['RACE_AFAM'] = df_input['ep_afam']
      df_input['RACE_ASIAN'] = df_input['ep_asian']
      df_input['NO_HSDP'] = df_input['ep_nohsdp']
      df_input['NO_EMPLOY'] = df_input['ep_unemp']
      df_input.loc[df_input['NO_EMPLOY']<0,'NO_EMPLOY'] = np.nan
      df_input['POP_POOR'] = df_input['ep_pov150']
      df_input['NO_VEH'] = df_input['ep_noveh']
      df_input.loc[df_input['NO_VEH']<0,'NO_VEH'] = np.nan
      df_input['NO_INSUR'] = df_input['ep_uninsur']
      df_input.loc[df_input['NO_INSUR']<0,'NO_INSUR'] = np.nan
      df_input['POP_DISABLE'] = df_input['ep_disabl']
      df_input.loc[df_input['POP_DISABLE']<0,'POP_DISABLE'] = np.nan

      df_input['DIST_IDA'] = df_input['dist_to_Ida']
      df_input['DIST_COAST'] = df_input['dist_to_south_bound']
      df_input['DIST_NICHOLAS'] = df_input['dist_to_Nicholas']

      df_input['EVAC_0'] = df_input['disp_ratio_wave0']
      df_input['EVAC_1'] = df_input['disp_ratio_wave1']
      df_input['EVAC_2'] = df_input['disp_ratio_wave2']
      df_input['EVAC_3'] = df_input['disp_ratio_wave3']

      df_input['RETURN_0'] = df_input['return_ratio_wave0_7days']
      df_input['RETURN_1'] = df_input['return_ratio_wave1_7days']
      df_input['RETURN_2'] = df_input['return_ratio_wave2_7days']
      df_input['RETURN_3'] = df_input['return_ratio_wave3_7days']

      df_input['DIST_0'] = df_input['dist_wave0'].clip(lower = 1)
      df_input['DIST_1'] = df_input['dist_wave1'].clip(lower = 1)
      df_input['DIST_2'] = df_input['dist_wave2'].clip(lower = 1)
      df_input['DIST_3'] = df_input['dist_wave3'].clip(lower = 1)

      df_input['DURATION_0'] = df_input['duration_wave0']
      df_input['DURATION_1'] = df_input['duration_wave1']
      df_input['DURATION_2'] = df_input['duration_wave2']
      df_input['DURATION_3'] = df_input['duration_wave3']

      df_input['URBAN'] = df_input['urban'].astype(int)
      merged_tract = gpd.read_file(P.data + '/map/merged_tract.gpkg')
      merged_tract = merged_tract[['id','geometry']].to_crs(CRS_M)
      df_input = pd.merge(merged_tract, df_input, on = 'id').set_geometry('geometry')

      df_input.dropna(inplace= True)

      normalize_factors =  ['DIST_IDA', 'DIST_COAST',  \
            'CASE_0', 'VAC_0', \
      'CASE_1', 'VAC_1', \
      'CASE_2', 'VAC_2', \
      'CASE_3', 'VAC_3', \
      'AGE_MINOR', 'AGE_SENIOR', \
      'RACE_HISP', 'RACE_AFAM', 'RACE_ASIAN', \
      'NO_HSDP', 'NO_EMPLOY',  'NO_VEH', 'NO_INSUR', 'POP_DISABLE', 'POP_POOR', \
      ]
      factors = normalize_factors + ['URBAN']
      targets = ['EVAC_0', 'RETURN_0', 'DIST_0', 'DURATION_0',\
            'EVAC_1', 'RETURN_1', 'DIST_1', 'DURATION_1',\
            'EVAC_2', 'RETURN_2', 'DIST_2', 'DURATION_2',\
            'EVAC_3', 'RETURN_3', 'DIST_3', 'DURATION_3']
      
      # split the ind into 5 folds for cross validation with random seed 42
      np.random.seed(42)
      cv_ind = np.random.permutation(np.arange(len(df_input)))
      cv_ind = np.array_split(cv_ind, 5)
      
      for fold in range(5): # fold for train/test split
            args.seed = fold
            for covid in [True, False]:
                  args.covid = covid
                  # get the train, test data, using 4 folds for train, the other fold for test
                  tmp_ind, test_ind = np.concatenate([cv_ind[i] for i in range(5) if i != fold]), cv_ind[fold]
                  if args.ratio < 1:
                        train_ind = train_test_split(tmp_ind, test_size = 1-args.ratio, random_state=args.seed)[0] # use how many percent of data for training
                  else:
                        train_ind = tmp_ind
                  df_train = df_input.iloc[train_ind].reset_index(drop=True)
                  df_test = df_input.iloc[test_ind].reset_index(drop=True)
                  # data normalization except for binary variables
                  scaler = StandardScaler()
                  scaler.fit(df_train[normalize_factors])
                  df_train[normalize_factors] = scaler.transform(df_train[normalize_factors])
                  df_test[normalize_factors] = scaler.transform(df_test[normalize_factors])

                  scaler_y = {} # for normalized y, need to do this for ML models using stochastic optimization 
                  for target in targets:
                        scaler_y[target] = StandardScaler()
                        scaler_y[target].fit(df_train[[target]].values)

                  
                  # load the res if exists
                  res = {}

                  print("Linear Regression")
                  for w in range(1, 4):
                        y_names = [f'EVAC_{w}', f'RETURN_{w}', f'DIST_{w}', f'DURATION_{w}']
                        for k in range(4):
                              X_names = get_X_names(wave = w, covid = args.covid)
                              X_train = df_train[X_names].values
                              y_train = df_train[y_names[k]].values
                              model = LinearRegression()
                              baseline = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()

                              best_ind = list(range(X_train.shape[1]))
                              # iterate 2 ** len(X_names) times to decide which variables to keep
                              if args.covid:
                                    num_X = len(X_names) - 2
                              else:
                                    num_X = len(X_names)
                              for m in range(2 ** num_X):
                                    tmp = m
                                    tmp_ind = []
                                    for i in range(num_X):
                                          if tmp % 2 == 1:
                                                tmp_ind.append(i)
                                          tmp = tmp // 2
                                    if args.covid:
                                          tmp_ind += [len(X_names)-2, len(X_names)-1]
                                    if len(tmp_ind) == 0:
                                          continue
                                    X_train_ = X_train[:, tmp_ind]
                                   
                                    model = LinearRegression()
                                    tmp_score = cross_val_score(model, X_train_, y_train, cv=5, scoring='neg_mean_squared_error').mean()
                                    if tmp_score > baseline:
                                          baseline = tmp_score
                                          best_ind = tmp_ind

                              X_names = [X_names[i] for i in best_ind]
                              X_train = df_train[X_names].values
                              X_test = df_test[X_names].values
                              y_test = df_test[y_names[k]].values

                              model = LinearRegression()
                              model.fit(X_train, y_train)
                              y_pred = model.predict(X_test)
                              df_test['resid'] = y_pred - y_test

                              res[f'linear_{w*4+k}'] = [[np.sum(np.abs((y_test - y_pred))),\
                                                            mean_squared_error(y_test, y_pred) * len(y_test),\
                                                            np.sum(np.abs((y_test - y_pred) / (y_test+1e-4))) * 100,\
                                                            2*np.sum(np.abs((y_test - y_pred) / (y_pred+y_test+1e-4))) * 100,\
                                                            np.sum(y_test)], model, scaler, None, X_names, df_test.reset_index(drop=True)]
                  
                  print("Logistic Regression")
                  for w in range(1, 4):
                        y_names = [f'EVAC_{w}', f'RETURN_{w}', f'DIST_{w}', f'DURATION_{w}']
                        for k in range(4):
                              X_names = get_X_names(wave = w, covid = args.covid)
                              X_train = df_train[X_names].values
                              y_train = df_train[y_names[k]].values
                              if k == 1:
                                    y_exp = np.log(100 - y_train + 1e-6)
                              else:
                                    y_exp = np.log(y_train + 1e-6)
                              model = LinearRegression()
                              baseline = cross_val_score(model, X_train, y_exp, cv=5, scoring='neg_mean_squared_error').mean()

                              base_ind = list(range(X_train.shape[1]))
                              # iterate 2 ** len(X_names) times to decide which variables to keep
                              if args.covid:
                                    num_X = len(X_names) - 2
                              else:
                                    num_X = len(X_names)
                              for m in range(2 ** num_X):
                                    tmp = m
                                    tmp_ind = []
                                    for i in range(num_X):
                                          if tmp % 2 == 1:
                                                tmp_ind.append(i)
                                          tmp = tmp // 2
                                    if args.covid:
                                          tmp_ind += [len(X_names)-2, len(X_names)-1]
                                    if len(tmp_ind) == 0:
                                          continue
                                   
                                    X_train_ = X_train[:, tmp_ind]
                                    model = LinearRegression()
                                    tmp_score = cross_val_score(model, X_train_, y_exp, cv=5, scoring='neg_mean_squared_error').mean()
                                                
                                    if tmp_score > baseline:
                                          baseline = tmp_score 
                                          best_ind = tmp_ind
                              
                              X_names = [X_names[i] for i in best_ind]
                              X_train = df_train[X_names].values
                              X_test = df_test[X_names].values
                              y_test = df_test[y_names[k]].values

                              model = LinearRegression()
                              model.fit(X_train, y_exp)
                              y_pred = model.predict(X_test)

                              if k == 1:
                                    y_pred = 100 - np.exp(y_pred) - 1e-6
                              else:
                                    y_pred = np.exp(y_pred) - 1e-6

                              df_test['resid'] = y_pred - y_test

                              res[f'logit_{w*4+k}'] = [[np.sum(np.abs((y_test - y_pred))),\
                                                            mean_squared_error(y_test, y_pred) * len(y_test),\
                                                            np.sum(np.abs((y_test - y_pred) / (y_test+1e-4))) * 100,\
                                                            2*np.sum(np.abs((y_test - y_pred) / (y_pred+y_test+1e-4))) * 100,\
                                                            np.sum(y_test)], model, scaler, None, X_names, df_test.reset_index(drop=True)]
                  
                  # Random Forest
                  print("Random Forest")
                  for w in range(1, 4):
                        y_names = [f'EVAC_{w}', f'RETURN_{w}', f'DIST_{w}', f'DURATION_{w}']
                        for k in range(4):
                              X_names = get_X_names(wave = w, covid = args.covid)
                              X_train = df_train[X_names].values
                              y_train = scaler_y[y_names[k]].transform(df_train[y_names[k]].values.reshape(-1, 1)).flatten()
                              
                              best_score = -100000
                              best_estimator = -1
                              best_depth = -1

                              for n_estimators in range(5,51,5):
                                    for max_depth in range(2,21,2):
                                          # define the model
                                          model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                                          cv = KFold(n_splits=5, random_state=42, shuffle=True)

                                          score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1).mean()

                                          if score > best_score:
                                                best_score = score
                                                best_estimator = n_estimators
                                                best_depth = max_depth
                                          
                              X_test = df_test[X_names].values
                              y_test = df_test[y_names[k]].values

                              model =  RandomForestRegressor(n_estimators=best_estimator, max_depth=best_depth, random_state=42)
                              model.fit(X_train, y_train)
                              y_pred = scaler_y[y_names[k]].inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()

                              df_test['resid'] = y_pred - y_test

                              res[f'rf_{w*4+k}'] = [[np.sum(np.abs((y_test - y_pred))),\
                                                      mean_squared_error(y_test, y_pred) * len(y_test),\
                                                      np.sum(np.abs((y_test - y_pred) / (y_test+1e-4))) * 100,\
                                                      2*np.sum(np.abs((y_test - y_pred) / (y_pred+y_test+1e-4))) * 100,\
                                                      np.sum(y_test)], model, scaler, scaler_y[y_names[k]], X_names, df_test.reset_index(drop=True)]
                  
                  # XGBoost
                  print("XGBoost")
                  for w in range(1, 4):
                        y_names = [f'EVAC_{w}', f'RETURN_{w}', f'DIST_{w}', f'DURATION_{w}']
                        for k in range(4):
                              X_names = get_X_names(wave = w, covid = args.covid)
                              X_train = df_train[X_names].values
                              y_train = scaler_y[y_names[k]].transform(df_train[y_names[k]].values.reshape(-1, 1)).flatten()
                              
                              best_score = -100000
                              best_estimator = -1
                              best_depth = -1

                              for n_estimators in range(2,21,2):
                                    for max_depth in range(1,11):
                                          # define the model
                                          model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                                          cv = KFold(n_splits=5, random_state=42, shuffle=True)

                                          score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1).mean()

                                          if score > best_score:
                                                best_score = score
                                                best_estimator = n_estimators
                                                best_depth = max_depth
                                          
                              X_test = df_test[X_names].values
                              y_test = df_test[y_names[k]].values

                              model =  xgb.XGBRegressor(n_estimators=best_estimator, max_depth=best_depth, random_state=42)
                              model.fit(X_train, y_train)
                              y_pred = scaler_y[y_names[k]].inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()

                              df_test['resid'] = y_pred - y_test

                              res[f'xgb_{w*4+k}'] = [[np.sum(np.abs((y_test - y_pred))),\
                                                            mean_squared_error(y_test, y_pred) * len(y_test),\
                                                            np.sum(np.abs((y_test - y_pred) / (y_test+1e-4))) * 100,\
                                                            2*np.sum(np.abs((y_test - y_pred) / (y_pred+y_test+1e-4))) * 100,\
                                                            np.sum(y_test)], model, scaler, scaler_y[y_names[k]], X_names, df_test.reset_index(drop=True)]
                              

                  # SVM
                  print("SVM")
                  for w in range(1, 4):
                        y_names = [f'EVAC_{w}', f'RETURN_{w}', f'DIST_{w}', f'DURATION_{w}']
                        for k in range(4):
                              X_names = get_X_names(wave = w, covid = args.covid)
                              X_train = df_train[X_names].values
                              y_train = scaler_y[y_names[k]].transform(df_train[y_names[k]].values.reshape(-1, 1)).flatten()
                              
                              best_score = -100000
                              best_C = -1
                              best_epsilon = -1

                              for C in np.arange(0.1, 2.1, 0.2):
                                    for epsilon in np.arange(0.1, 1.1, 0.1):
                                          # define the model
                                          model = SVR(C=C, epsilon=epsilon)
                                          cv = KFold(n_splits=5, random_state=42, shuffle=True)

                                          score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv).mean()

                                          if score > best_score:
                                                best_score = score
                                                best_C = C
                                                best_epsilon = epsilon
                                          
                              X_test = df_test[X_names].values
                              y_test = df_test[y_names[k]].values

                              model = SVR(C=best_C, epsilon=best_epsilon)
                              model.fit(X_train, y_train)
                              y_pred = scaler_y[y_names[k]].inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()

                              df_test['resid'] = y_pred - y_test

                              res[f'svm_{w*4+k}'] = [[np.sum(np.abs((y_test - y_pred))),\
                                                            mean_squared_error(y_test, y_pred) * len(y_test),\
                                                            np.sum(np.abs((y_test - y_pred) / (y_test+1e-4))) * 100,\
                                                            2*np.sum(np.abs((y_test - y_pred) / (y_pred+y_test+1e-4))) * 100,\
                                                            np.sum(y_test)], model, scaler, scaler_y[y_names[k]], X_names, df_test.reset_index(drop=True)]
                  
                  # MLP
                  print("MLP")
                  batch_sizes =  [32, 64, 128]
                  lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]
                  activations = ['relu']
                  shapes = [[64], [64, 64], [64, 64, 64]]

                  for w in range(1, 4):
                        y_names = [f'EVAC_{w}', f'RETURN_{w}', f'DIST_{w}', f'DURATION_{w}']
                        for k in range(4):
                              X_names = get_X_names(wave = w, covid = args.covid)
                              X_train = df_train[X_names].values
                              y_train = scaler_y[y_names[k]].transform(df_train[y_names[k]].values.reshape(-1, 1)).flatten()
                              
                              best_score = -100000
                              best_batch_size = -1
                              best_lr = -1
                              best_activation = ''
                              best_shape = shapes[0]

                              for shape in shapes:
                                    for activation in activations:
                                          for batch_size in batch_sizes:
                                                for lr in lrs:
                                                      # define the model
                                                      model = MLPRegressor(hidden_layer_sizes=(shape), activation=activation, solver='adam', batch_size=batch_size, learning_rate_init=lr, random_state=42, max_iter=10000, early_stopping=True)
                                                      cv = KFold(n_splits=5, random_state=42, shuffle=True)

                                                      score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv).mean()

                                                      if score > best_score:
                                                            best_score = score
                                                            best_batch_size = batch_size
                                                            best_lr = lr
                                                            best_activation = activation
                                                            best_shape = shape
                                          
                              X_test = df_test[X_names].values
                              y_test = df_test[y_names[k]].values

                              model = MLPRegressor(hidden_layer_sizes=(best_shape), activation=best_activation, solver='adam', batch_size=best_batch_size, learning_rate_init=best_lr, max_iter=10000, random_state=42, validation_fraction = 0.1, early_stopping=True)
                              model.fit(X_train, y_train)
                              y_pred = scaler_y[y_names[k]].inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()

                              df_test['resid'] = y_pred - y_test

                              res[f'nn_{w*4+k}'] = [[np.sum(np.abs((y_test - y_pred))),\
                                                            mean_squared_error(y_test, y_pred) * len(y_test),\
                                                            np.sum(np.abs((y_test - y_pred) / (y_test+1e-4))) * 100,\
                                                            2*np.sum(np.abs((y_test - y_pred) / (y_pred+y_test+1e-4))) * 100,\
                                                            np.sum(y_test)], model, scaler, scaler_y[y_names[k]], X_names, df_test.reset_index(drop=True)]
                  
                  with open(f'{P.data}/res_{int(args.ratio*100)}_{args.covid}_{args.seed}.pkl', 'wb') as f:
                        pickle.dump(res, f)
                  
                  print(f"res_{int(args.ratio*100)}_{args.covid}_{args.seed}.pkl Done!")