# %%

import random
import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import plot
from feature_engine.encoding import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

# %%

data = pd.read_pickle('../data/aus_weather_cln.pkl')

# %%

data['Date'] = pd.to_datetime(data['Date'])

# %%

data['RainToday'] = data['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)
data['RainTomorrow'] = data['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)

# %%

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Quarter'] = data['Date'].dt.quarter

# %%

location_encoder = OrdinalEncoder(encoding_method='arbitrary',
                                  variables=['Location'])
location_encoder.fit(data)
# location_encoder.encoder_dict_
data = location_encoder.transform(data)

# %% md

### regression feature selection

# %%

temp = data.drop(['RainTomorrow', 'Date'], axis=1)
y = data['RainTomorrow']
features = SelectKBest(score_func=f_regression, k=10)
selected_features = features.fit_transform(temp, y)

# %%

input_feature = ['Location', 'Rainfall', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
                 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'RainToday']
train_x, test_x, train_y, test_y = train_test_split(temp[input_feature], y, test_size=0.2, random_state=64)


# %%

def splitter(index, column, dataset):
    left = dataset[dataset.iloc[:, column] < dataset.iloc[index, column]]
    right = dataset[dataset.iloc[:, column] >= dataset.iloc[index, column]]

    return left, right


def gini_score(groups, classes):
    score = 0.0
    n_instance = np.sum([len(group) for group in groups])
    for group in groups:
        class_values = 0.0

        if len(group) == 0:
            continue

        for cls in classes:
            class_values += np.power(np.sum(group.iloc[:, -1] == cls) / len(group), 2)

        # weighted score
        score += (len(group) / n_instance) * (1 - class_values)

    return score


def to_terminal(group):
    outcome = list(group.iloc[:, -1])

    return max(outcome, key=outcome.count)


def find_split_point(dataset):
    classes = set(dataset.iloc[:, -1])
    features = list(dataset.columns)
    dataset = dataset.reset_index().drop('index', axis=1)
    t_index, t_value, t_score, t_groups = 99999, 99999, 99999, None

    for col in range(len(features) - 1):
        for index in dataset.index:
            groups = splitter(index, col, dataset)
            score = gini_score(groups, classes)

            if score < t_score:
                t_index, t_score, t_value, t_groups = col, score, dataset.iloc[index, col], groups

    return {"index": t_index,
            "value": t_value,
            "groups": t_groups}


def rec_split(node, depth, max_depth, minimum_batch):
    left, right = node['groups']
    del (node['groups'])

    # check for no split
    if left.empty or right.empty:
        node['left'] = node["right"] = to_terminal(left + right)
        return

        # check for max depth
    if depth >= max_depth:
        node['left'], node["right"] = to_terminal(left), to_terminal(right)
        return

    # process left branch
    if len(left) <= minimum_batch:
        node["left"] = to_terminal(left)
    else:
        node["left"] = find_split_point(left)
        rec_split(node["left"], depth + 1, max_depth, minimum_batch)

    # process right branch
    if len(right) <= minimum_batch:
        node["right"] = to_terminal(right)
    else:
        node["right"] = find_split_point(right)
        rec_split(node["right"], depth + 1, max_depth, minimum_batch)


def build_tree(train_data, max_depth, minimum_batch):
    root = find_split_point(train_data)
    rec_split(root, 1, max_depth, minimum_batch)

    return root

#%%

dataset = pd.concat([train_x, train_y], axis=1)
tree = build_tree(dataset, 5, 20)
tree

