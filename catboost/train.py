import os
import pandas as pd

import yaml
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

import sys 
sys.path.append('tools/')
from utils import create_experiment_folder
from dataloader import CustomDatasetRegression
from normalization import norm_max

import warnings
warnings.filterwarnings("ignore")

EXP_PATH = 'catboost/experiments'
ARGS = {
    'iterations': 500, 
    'depth': 7,
    'verbose': False
}
DYNAMICS = [
    'linear',
    'linear_short',
    'steps_up',
    # 'steps_up_pulse',
    # 'steps_down',
    # 'steps_down_pulse'
]
SENSORS = ['R1','R2', 'R3']
GASES = ['NO', 'CH4', 'H2S', 'SO2', 'HCOH', 'CO', 'H2', 'NO2', 'NH3']


def train_catboost_old(
        path, 
        exp_path, 
        dynamics, 
        sensors, 
        gases, 
        metric
    ):

    global ARGS
    exp_path = create_experiment_folder(exp_path)
    with open(os.path.join(exp_path, 'config.yaml'), 'w') as file:
        yaml.dump(ARGS, file, default_flow_style=False)

    metric_dict = {}
    for sensor in sensors:
        metric_dict[sensor] = pd.DataFrame(index=dynamics, columns=gases)

    for dynamic in dynamics:
        print(f'Dynamic {dynamic}')
        for gas in gases:
            dataset_train = CustomDatasetRegression(
                path=path,
                dynamic=dynamic, 
                gas=gas, 
                step='trn', 
                output_type='np'
            )
            dataset_valid = CustomDatasetRegression(
                path=path,
                dynamic=dynamic, 
                gas=gas, 
                step='tst', 
                output_type='np'
            )
            for sensor in sensors:
                try:
                    x_train, y_train = dataset_train.get_all_data(sensor, 'conc')
                    x_valid, y_valid = dataset_valid.get_all_data(sensor, 'conc')

                    model = CatBoostRegressor(**ARGS)
                    model.fit(x_train, y_train)

                    y_valid_pred = model.predict(x_valid)
                    metric_value = metric(y_valid, y_valid_pred)
                    metric_dict[sensor].loc[dynamic, gas] = metric_value

                    cb_model_name = f'cb_{sensor}_{dynamic}_{gas}.cbm'
                    model.save_model(os.path.join(exp_path, cb_model_name))
                except Exception as e:
                    print(f'Sensor {sensor} on {dynamic} dynamic of {gas} gas: {e}')
                    
    for sensor in sensors:
        metric_dict[sensor].to_csv(os.path.join(exp_path, f'{sensor}.csv'), index=False)


def test_catboost_old(
        data_path,
        exp_path,
        results_path,
        dynamics,
        sensors,
        gases,
        metric
    ):
    results_path = os.path.join(results_path, data_path.split('/')[-1] + '_on_' + exp_path.split('/')[-1])
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    metric_dict = {}
    for sensor in sensors:
        metric_dict[sensor] = pd.DataFrame(index=dynamics, columns=gases)

    for dynamic in dynamics:
        print(f'Dynamic {dynamic}')
        for gas in gases:
            dataset_test = CustomDatasetRegression(
                path=data_path,
                dynamic=dynamic,
                gas=gas,
                step='tst',
                output_type='np'
            )
            for sensor in sensors:
                try:
                    x_test, y_test = dataset_test.get_all_data(sensor, 'conc')

                    cb_model_name = f'cb_{sensor}_{dynamic}_{gas}.cbm'
                    model_path = os.path.join(exp_path, cb_model_name)

                    model = CatBoostRegressor()
                    model.load_model(model_path)

                    y_test_pred = model.predict(x_test)
                    metric_value = metric(y_test, y_test_pred)
                    metric_dict[sensor].loc[dynamic, gas] = metric_value
                except Exception as e:
                    print(f'Sensor {sensor} on {dynamic} dynamic of {gas} gas: {e}')

    for sensor in sensors:
        metric_dict[sensor].to_csv(os.path.join(results_path, f'{sensor}_test_metrics.csv'))

def train_catboost_new(
        path, 
        exp_path, 
        dynamics, 
        sensors, 
        gases, 
        metric,
        normalize=None
    ):

    global ARGS
    exp_path = create_experiment_folder(exp_path)
    with open(os.path.join(exp_path, 'config.yaml'), 'w') as file:
        yaml.dump(ARGS, file, default_flow_style=False)

    metric_dict = {}
    for sensor in sensors:
        metric_dict[sensor] = pd.DataFrame(index=dynamics, columns=gases)

    for dynamic in dynamics:
        print(f'Dynamic {dynamic}')
        for gas in gases:
            dataset_full = CustomDatasetRegression(
                path=path,
                dynamic=dynamic, 
                gas=gas, 
                step='full', 
                output_type='np',
                normalize=normalize
            )
            for sensor in sensors:
                x_train, y_train = dataset_full.get_all_data_by_step(sensor, 'conc', 'trn')
                x_valid, y_valid = dataset_full.get_all_data_by_step(sensor, 'conc', 'tst')

                model = CatBoostRegressor(**ARGS)
                model.fit(x_train, y_train)

                y_valid_pred = model.predict(x_valid)
                metric_value = metric(y_valid, y_valid_pred)
                metric_dict[sensor].loc[dynamic, gas] = metric_value

                cb_model_name = f'cb_{sensor}_{dynamic}_{gas}.cbm'
                model.save_model(os.path.join(exp_path, cb_model_name))
    
    for sensor in sensors:
        metric_dict[sensor].to_csv(os.path.join(exp_path, f'{sensor}.csv'), index=False)  

train_catboost_new(
    path='data/exp4_melted',
    exp_path='catboost/experiments', 
    dynamics=DYNAMICS, 
    sensors=SENSORS, 
    gases=GASES, 
    metric=r2_score,
    normalize=norm_max
)
