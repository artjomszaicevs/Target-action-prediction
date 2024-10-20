# import joblib
import datetime

import dill
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score


def filter_data(df):
    print ('filter_data')
    columns_to_drop = ['device_model', 'device_browser', ]
    return df.drop(columns=columns_to_drop, axis=1).copy()


def anomaly_repl(df):
    print('anomaly_repl')
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries2 = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries2

    anomaly_list = ['device_screen_height', 'device_screen_width']
    for feature in anomaly_list:
        boundaries = calculate_outliers(df[feature])
        df.loc[df[feature] < boundaries[0], feature] = round(boundaries[0])
        df.loc[df[feature] > boundaries[1], feature] = round(boundaries[1])
    return df.copy()




def create_features(df):
    print('create_features')

    df['device_screen_resolution'] = df['device_screen_resolution'].astype(str)
    df.loc[:, 'device_screen_width'] = df['device_screen_resolution'].apply(
        lambda x: int(x.split('x')[0]) if 'x' in x else None)

    df.loc[:, 'device_screen_height'] = df['device_screen_resolution'].apply(
        lambda x: int(x.split('x')[1]) if 'x' in x and len(x.split('x')) == 2 else None)

    df.loc[:, 'free_trafic'] = df['utm_medium'].apply(
        lambda x: 1 if x in ['organic', 'referral', '(none)'] else 0)
    df.loc[:, 'payed_trafic'] = df['utm_medium'].apply(
        lambda x: 1 if x not in ['organic', 'referral', '(none)'] else 0)
    df.loc[:, 'soc_mdeia_adv'] = df['utm_source'].apply(
        lambda x: 1 if x in ('QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt',
                             'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                             'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm') else 0)
    df.loc[:, 'other_adv'] = df['utm_source'].apply(
        lambda x: 1 if x not in ('QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt',
                             'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
                             'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm') else 0)


    columns_to_drop = ['device_screen_resolution', 'utm_medium', 'utm_source']
    return df.drop(columns=columns_to_drop, axis=1).copy()


def transform_features(df):
    print ('transform_features')

    city_counts = df['geo_city'].value_counts()
    country_counts = df['geo_country'].value_counts()
    campaign_counts = df['utm_campaign'].value_counts()
    adcontent_counts = df['utm_adcontent'].value_counts()
    brand_counts = df['device_brand'].value_counts()

# 1000 500 10 6 1
    df['geo_city'] = df['geo_city'].apply(lambda x: 'other_cities' if city_counts.get(x, 0) <= 1000 else x)
    df['geo_country'] = df['geo_country'].apply(lambda x: 'other_countries' if country_counts.get(x, 0) <= 100 else x)
    df['utm_campaign'] = df['utm_campaign'].apply(lambda x: 'other_utm_campaign' if campaign_counts.get(x, 0) <= 5 else x)
    # df['utm_adcontent'] = df['utm_adcontent'].apply(lambda x: 'other_utm_adcontent' if adcontent_counts.get(x, 0) <= 20 else x)
    df['device_brand'] = df['device_brand'].apply(lambda x: 'other_device_brand' if brand_counts.get(x, 0) <= 1 else x)


    return df.copy()



def main():

    df = pd.read_csv('data/df_ready.csv')

    target_values = [
        'sub_car_claim_click', 'sub_car_claim_submit_click',
        'sub_open_dialog_click', 'sub_custom_question_submit_click',
        'sub_call_number_click', 'sub_callback_submit_click',
        'sub_submit_success', 'sub_car_request_submit_click'
    ]

    df['event_action'] = df['event_action'].apply(lambda x: 1 if x in target_values else 0)

    X = df.drop('event_action', axis=1)
    y = df['event_action']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)



    numerical_transformer = Pipeline(steps=[
        ('inputer', SimpleImputer(strategy = 'median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        # ('inputer', SimpleImputer(strategy='most_frequent')),
        ('inputer', SimpleImputer(strategy = 'constant', fill_value = 'other')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])



    preprocessor = Pipeline(steps=[
        ('delete_cols', FunctionTransformer(filter_data)),
        ('create_features', FunctionTransformer(create_features)),
        ('transform_features', FunctionTransformer(transform_features)),
        ('anomaly_repl', FunctionTransformer(anomaly_repl)),
        ('column_transformer', column_transformer)
    ])



    models = (
        LogisticRegression(C=0.1,
                           penalty='l2',
                           solver='saga',
                           max_iter=50,
                           warm_start=True,
                           class_weight='balanced',
                           random_state=42,
                           n_jobs=-1
                           ),
        # RandomForestClassifier(n_estimators=50,
        #                        max_depth=6,
        #                        min_samples_split=60,
        #                        min_samples_leaf=11,
        #                        max_features='sqrt',
        #                        bootstrap=True,
        #                        # max_leaf_nodes=10,
        #                        class_weight='balanced',
        #                        n_jobs=-1
        #                        ),
    )

    roc_auc = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=2, scoring=roc_auc)
        print(f'model: {type(model).__name__}, roc_auc_score_mean: {score.mean():.4f}, roc_auc_score_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)

    # feature_importances = best_pipe.named_steps['classifier'].feature_importances_
    #
    # f_imp_list = list(zip(X.columns, feature_importances))
    # f_imp_list.sort(key=lambda x: x[1], reverse=True)
    # print(f_imp_list)

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    # joblib.dump(best_pipe, 'price_cat_pipe.pkl')
    with open('model.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'prediction',
                'author': 'Artjom Zaicev',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)

if __name__ == '__main__':
    main()
