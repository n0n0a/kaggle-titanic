import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

pd.set_option('display.max_columns', 100)


def main():
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')

    combine = [train_df, test_df]

    # for dataset in combine:
    #     dataset['Cabin'].fillna('NA', inplace=True)
    #     dataset['Cabin_Prefix'] = dataset['Cabin'].map(lambda x: x[0])

    # pref_mean = train_df[['Cabin_Prefix', 'Survived']].groupby(['Cabin_Prefix'])['Survived'].mean()
    # print(pref_mean)
    #
    # for dataset in combine:
    #     counts = dataset['Ticket'].value_counts()
    #     dataset['Cabin_TargetEncoding'] = dataset['Cabin_Prefix'].dropna().map(lambda x: pref_mean[x])

    train_df = train_df.drop(['Cabin', 'Ticket'], axis=1)
    test_df = test_df.drop(['Cabin', 'Ticket'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # print(pd.crosstab(train_df['Title'], train_df['Survived']))
    # print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

    train_df['Embarked'].fillna('NA', inplace=True)

    for dataset in combine:
        for c in dataset.select_dtypes(include=object).columns.values:
            if c == 'Sex' or c == 'Embarked' or c == 'Title' or c == 'Pclass':
                le = LabelEncoder()
                dataset[c] = le.fit_transform(dataset[c].values)


    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

    guess_ages = np.zeros((2, 3))
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                   (dataset['Pclass'] == j + 1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                            'Age'] = guess_ages[i, j]

        dataset['Age'] = dataset['Age'].astype(int)

    # grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
    # grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    # grid.add_legend()
    # plt.savefig('../data/fillna.png')

    for dataset in combine:
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
    combine = [train_df, test_df]

    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

    freq_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
        dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    # print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
    #                                                                                             ascending=True))

    for dataset in combine:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]
    print(train_df.head())

    X_train = train_df.drop('Survived', axis=1)
    Y_train = train_df['Survived']
    X_test = test_df.drop(['PassengerId'], axis=1)

    params = {'objective': 'binary', 'seed': 71,  'metrics': 'binary_logloss'}
    num_round = 500

    scores = []
    y_preds = []
    kf = KFold(n_splits=4, shuffle=True, random_state=71)
    for tr_idx, va_idx in kf.split(X_train):
        tr_x, va_x = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        tr_y, va_y = Y_train.iloc[tr_idx], Y_train.iloc[va_idx]

        lgb_tr = lgb.Dataset(tr_x, tr_y)
        lgb_va = lgb.Dataset(va_x, va_y)

        model = lgb.train(params, lgb_tr,
                          num_boost_round=num_round,
                          valid_names=['train', 'valid'],
                          valid_sets=[lgb_tr, lgb_va])

        va_pred = model.predict(va_x)
        score = log_loss(va_y, va_pred)
        scores.append(score)
        y_preds.append(model.predict(X_test))

    print(scores)
    Y_pred = sum(y_preds) / len(y_preds)
    Y_pred = np.where(Y_pred < 0.5, 0, 1)
    print(Y_pred)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)
    Y_pred = random_forest.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
    submission.to_csv("../output/submission.csv", index=False)


if __name__ == '__main__':
    main()