import pandas as pd
from pathlib import Path
from app.domain.model.titanic_schema import TitanicSchema

"""
PassengerId  고객ID,
Survived 생존여부,
Pclass 승선권 1 = 1등석, 2 = 2등석, 3 = 3등석,
Name,
Sex,
Age,
SibSp 동반한 형제, 자매, 배우자,
Parch 동반한 부모, 자식,
Ticket 티켓번호,
Fare 요금,
Cabin 객실번호,
Embarked 승선한 항구명 C = 쉐브루, Q = 퀸즈타운, S = 사우스햄튼
"""

class TitanicService:
    def __init__(self, data_dir: str = './app/store/'):
        self.dataset = TitanicSchema(
            context=data_dir,
            fname='',
            train=None,
            test=None,
            id='',
            label=''
        )

    def new_model(self, fname: str) -> pd.DataFrame:
        path = Path(self.dataset.context) / fname
        return pd.read_csv(path)

    def preprocess(self, train_fname: str, test_fname: str) -> TitanicSchema:
        print("-------- 모델 전처리 시작 --------")
        ds = self.dataset
        ds.train = self.new_model(train_fname)
        ds.test = self.new_model(test_fname)
        ds.id = ds.test['PassengerId']
        ds.label = ds.train['Survived']
        ds.train = ds.train.drop('Survived', axis=1)

        ds = self.drop_feature(ds, 'SibSp', 'Parch', 'Cabin', 'Ticket')
        ds = self.extract_title_from_name(ds)
        title_mapping = self.remove_duplicate_title(ds)
        ds = self.title_nominal(ds, title_mapping)
        ds = self.drop_feature(ds, 'Name')
        ds = self.gender_nominal(ds)
        ds = self.drop_feature(ds, 'Sex')
        ds = self.embarked_nominal(ds)
        ds = self.age_ratio(ds)
        ds = self.drop_feature(ds, 'Age')
        ds = self.pclass_ordinal(ds)
        ds = self.fare_ordinal(ds)
        ds = self.drop_feature(ds, 'Fare')

        return ds

    def drop_feature(self, dataset: TitanicSchema, *features: str) -> TitanicSchema:
        for col in features:
            dataset.train.drop(columns=col, inplace=True)
            dataset.test.drop(columns=col, inplace=True)
        return dataset

    def extract_title_from_name(self, dataset: TitanicSchema) -> TitanicSchema:
        for df in [dataset.train, dataset.test]:
            df['Title'] = df['Name'].str.extract('([A-Za-z]+)\\.', expand=False)
        return dataset

    def remove_duplicate_title(self, dataset: TitanicSchema) -> dict:
        all_titles = set(dataset.train['Title'].unique()) | set(dataset.test['Title'].unique())
        print(f"📌 전체 직함 목록: {sorted(all_titles)}")
        return {
            'Mr': 1, 'Ms': 2, 'Mrs': 3, 'Master': 4,
            'Royal': 5, 'Rare': 6
        }

    def title_nominal(self, dataset: TitanicSchema, title_mapping: dict) -> TitanicSchema:
        for df in [dataset.train, dataset.test]:
            df['Title'] = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            df['Title'] = df['Title'].replace(
                ['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona', 'Mme'], 'Rare')
            df['Title'] = df['Title'].replace({'Mlle': 'Mr', 'Miss': 'Ms'})
            df['Title'] = df['Title'].map(title_mapping)
        return dataset

    def gender_nominal(self, dataset: TitanicSchema) -> TitanicSchema:
        mapping = {'male': 1, 'female': 2}
        for df in [dataset.train, dataset.test]:
            df['Gender'] = df['Sex'].map(mapping)
        return dataset

    def embarked_nominal(self, dataset: TitanicSchema) -> TitanicSchema:
        mapping = {'S': 1, 'C': 2, 'Q': 3}
        for df in [dataset.train, dataset.test]:
            df['Embarked'] = df['Embarked'].fillna('S')
            df['Embarked'] = df['Embarked'].map(mapping)
        return dataset

    def age_ratio(self, dataset: TitanicSchema) -> TitanicSchema:
        mapping = {'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4,
                   'Young Adult': 5, 'Adult': 6, 'Senior': 7}
        for df in [dataset.train, dataset.test]:
            df['Age'] = df['Age'].fillna(-1)

        max_age = max(dataset.train['Age'].max(), dataset.test['Age'].max())
        bins = [-1] + [max_age * i / 7 for i in range(1, 8)]
        labels = list(mapping.keys())[1:]  # 'Baby' ~ 'Senior'

        for df in [dataset.train, dataset.test]:
            df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
            df['AgeGroup'] = df['AgeGroup'].map(mapping)
        return dataset

    def fare_ordinal(self, dataset: TitanicSchema) -> TitanicSchema:
        mapping = {'Free': 1, 'Lower': 2, 'Mid': 3, 'Premium': 4}
        bins = [0, 0.01, 20, 150, 870]
        labels = list(mapping.keys())

        for df in [dataset.train, dataset.test]:
            df['Fare'] = df['Fare'].fillna(0)
            df['FareGroup'] = pd.cut(df['Fare'], bins=bins, labels=labels, include_lowest=True)
            df['FareGroup'] = df['FareGroup'].map(mapping)
        return dataset

    def pclass_ordinal(self, dataset: TitanicSchema) -> TitanicSchema:
        return dataset