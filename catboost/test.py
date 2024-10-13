import os
import pandas as pd


from catboost import CatBoostRegressor

from sklearn.metrics import r2_score

import sys 
sys.path.append('tools/')
from utils import create_experiment_folder
from dataloader import CustomDatasetRegression
from normalization import norm_max

import warnings
warnings.filterwarnings("ignore")


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



def test_catboost_old(
        data_path,
        exp_path,
        results_path,
        dynamics,
        sensors,
        gases,
        metric,
        normalize=None
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
                step='full',
                output_type='np',
                normalize=norm_max
            )
            for sensor in sensors:
                try:
                    x_test, y_test = dataset_test.get_all_data_by_step(sensor, 'conc', 'tst')

                    cb_model_name = f'cb_{sensor}_{dynamic}_{gas}.cbm'
                    model_path = os.path.join(exp_path, cb_model_name)

                    model = CatBoostRegressor()
                    model.load_model(model_path)

                    y_test_pred = model.predict(x_test)
                    metric_value = metric(y_test, y_test_pred)
                    #print(metric_value)
                    metric_dict[sensor].loc[dynamic, gas] = metric_value
                except Exception as e:
                    print(f'Sensor {sensor} on {dynamic} dynamic of {gas} gas: {e}')

    for sensor in sensors:
        metric_dict[sensor].to_csv(os.path.join(results_path, f'{sensor}_test_metrics.csv'))

test_catboost_old(
    data_path='/home/chernov.kirill9/projects/gases/gases-sensors/data/exp2_melted',
    exp_path='/home/chernov.kirill9/projects/gases/gases-sensors/catboost/experiments/normed_max_exp3', 
    results_path='/home/chernov.kirill9/projects/gases/gases-sensors/catboost/results', 
    dynamics=DYNAMICS, 
    sensors=SENSORS, 
    gases=GASES, 
    metric=r2_score,
    normalize=norm_max
)
