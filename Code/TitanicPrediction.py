import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#Read the csv files and display the first 5 rows  of the training set
df_train = pd.read_csv('../titanic_train.csv')
df_test = pd.read_csv('../titanic_test.csv')

df_train.head()

#Now we take a look at the features and the missing data in each column
print('features in training set: \n')
df_train.info()
print('\n')
print('features in testing set: \n')
df_test.info()

#This is the description of the dataset
df_train.describe()

g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=50)
#The number of males are bigger than the number of females but it seems that womens are more likely to survive
print(df_train.groupby(['Survived','Sex'])['Survived'].count())
df_train['Sex'].value_counts().plot.bar()
#Females are privileged over mans, they have bigger survival rates
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

sns.factorplot('Sex', 'Survived', data = df_train, kind='bar')

#Encode categorical variables to numbers, that is for mathematical equations of the model that we gonna use(logistic_regression)
#But it is possible to not map them to numbers and use algorithms like decision trees or random forests
df_train.loc[df_train['Sex'] == 'male', 'Sex'] = 0
df_train.loc[df_train['Sex'] == 'female', 'Sex'] = 1

df_test.loc[df_test['Sex'] == 'male', 'Sex'] = 0
df_test.loc[df_test['Sex'] == 'female', 'Sex'] = 1

#The higher ticket class passengers have, the higher chances they have to survive
print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
pd.crosstab(df_train['Pclass'],df_train['Survived'],margins=True).style.background_gradient(cmap='summer_r')

#Here we create a new feature called : Family = Parch + SibSp
df_train['Family'] = df_train['Parch'] + df_train['SibSp']
df_test['Family'] = df_test['Parch'] + df_test['SibSp']

#We drop the features: Parch and SibSp: they are no longer useful
df_train = df_train.drop(['Parch'], axis = 1)
df_train = df_train.drop(['SibSp'], axis = 1)
df_test = df_test.drop(['Parch'], axis = 1)
df_test = df_test.drop(['SibSp'], axis = 1)

#When a passenger has a big family, he has less chances to survive
df_train[['Family', 'Survived']].groupby(['Family'], as_index=False).sum().sort_values(by='Survived', ascending=False)

#We continue by dropping the Ticket feature and the Cabin feature
df_train = df_train.drop(['Ticket'], axis = 1)
df_train = df_train.drop(['Cabin'], axis = 1)

df_test = df_test.drop(['Ticket'], axis = 1)
df_test= df_test.drop(['Cabin'], axis = 1)

#The port of embarkation is an important feature to decide whether the passenger will survive or not
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()

print('The minimum value in the fare feature is {}'.format(df_train['Fare'].min()))
print('The maximum value in the fare feature is {}'.format(df_train['Fare'].max()))
print('The mean value in the fare feature is {}'.format(df_train['Fare'].mean()))

#Feature scaling / Mean Normalization
df_train['Fare'] = (df_train['Fare'] - df_train['Fare'].mean())/df_train['Fare'].std()
df_test['Fare'] = (df_test['Fare'] - df_test['Fare'].mean())/df_test['Fare'].std()

#After removing these features, it's time to fill the missing values
imputer = Imputer(missing_values = np.nan, strategy = 'median', axis = 0)
df_train[['Age']] = imputer.fit_transform(df_train[['Age']])
df_test[['Age']] = imputer.fit_transform(df_test[['Age']])

df_train.loc[ df_train['Age'] <= 16, 'Age'] = 0
df_train.loc[(df_train['Age'] > 16) & (df_train['Age'] <= 32), 'Age'] = 1
df_train.loc[(df_train['Age'] > 32) & (df_train['Age'] <= 48), 'Age'] = 2
df_train.loc[(df_train['Age'] > 48) & (df_train['Age'] <= 64), 'Age'] = 3
df_train.loc[ df_train['Age'] > 64, 'Age'] = 4

df_test.loc[ df_test['Age'] <= 16, 'Age'] = 0
df_test.loc[(df_test['Age'] > 16) & (df_test['Age'] <= 32), 'Age'] = 1
df_test.loc[(df_test['Age'] > 32) & (df_test['Age'] <= 48), 'Age'] = 2
df_test.loc[(df_test['Age'] > 48) & (df_test['Age'] <= 64), 'Age'] = 3
df_test.loc[ df_test['Age'] > 64, 'Age'] = 4


#The embarked feature has only two missing values, we fill them with the most occured one
only_S = df_train[df_train['Embarked'] == 'S'].count()
print(only_S['Embarked']) #646
only_C = df_train[df_train['Embarked'] == 'C'].count()
print(only_C['Embarked']) #168
only_Q = df_train[df_train['Embarked'] == 'Q'].count()
print(only_Q['Embarked']) #77

#The most occured one is 'S'
df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_test['Embarked'] = df_test['Embarked'].fillna('S')

df_train.loc[df_train['Embarked'] == 'S', 'Embarked'] = 0
df_train.loc[df_train['Embarked'] == 'C', 'Embarked'] = 1
df_train.loc[df_train['Embarked'] == 'Q', 'Embarked'] = 2

df_test.loc[df_test['Embarked'] == 'S', 'Embarked'] = 0
df_test.loc[df_test['Embarked'] == 'C', 'Embarked'] = 1
df_test.loc[df_test['Embarked'] == 'Q', 'Embarked'] = 2

#The fare feature has missing data in the test set
df_test[['Fare']] = imputer.fit_transform(df_test[['Fare']])

#Here, we deal with the Name feature
#We first extract the title that we save in a new feature called: 'Title' 
#We drop the rest of the name
df_train['Title'] = df_train['Name'].str.split(', ', expand=True)[1].str.split('. ', expand=True)[0]
df_test['Title'] = df_test['Name'].str.split(', ', expand=True)[1].str.split('. ', expand=True)[0]

#We don't need this feature anymore now
df_train = df_train.drop(['Name'], axis = 1)
df_test = df_test.drop(['Name'], axis = 1)
#These are the all the titles of the training set and the test set
print(df_train['Title'].unique())
print(df_test['Title'].unique())

df_train.head()

print(df_train['Title'].value_counts())
df_train['Title'].value_counts().plot.bar()

df_train[df_train['Title'] == 'Mr']['Survived'].value_counts()

df_train[df_train['Title'] == 'Miss']['Survived'].value_counts()

df_train['Title'] = df_train['Title'].map({'Mr' : 0 , 'Master' : 3, 'Don' : 4, 'Major' : 4, 'Sir' : 4, 
                                           'Mrs' : 2 , 'Miss' : 1, 'Mme' : 4, 'Ms' : 4, 'Lady' : 4, 'Mlle': 4, 
                                           'Rev' : 4 , 'Col' : 4, 'Capt' : 4, 'th' : 4, 'Jonkheer' : 4, 'Dr' : 4})

df_test['Title'] = df_test['Title'].map({'Mr' : 0 , 'Master' : 3, 'Don' : 4, 'Major' : 4, 'Sir' : 4, 
                                           'Mrs' : 2 , 'Miss' : 1, 'Mme' : 4, 'Ms' : 4, 'Lady' : 4, 'Mlle': 4, 
                                           'Rev' : 4 , 'Col' : 4, 'Capt' : 4, 'th' : 4, 'Jonkheer' : 4, 'Dr' : 4, 'Dona' : 4})
                                         

# Any results you write to the current directory are saved as output.

#This is how our dataset looks like now
df_train.head()

X_train = df_train.drop(['Survived'], axis = 1)
Y_train = df_train[['Survived']]
X_test = df_test

X_train.shape, Y_train.shape, X_test.shape
X_train.head()

#It's time to predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)

svc = SVC(kernel = 'linear')
svc.fit(X_train, Y_train)
Y_pred2 = svc.predict(X_test)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred3 = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
