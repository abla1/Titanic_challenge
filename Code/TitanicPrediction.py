<<<<<<< HEAD
import numpy
import pandas
import seaborn
import matplotlib.pyplot
=======
<<<<<<< HEAD
[Code goes here]
######## Imports ########
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from IPython.display import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
#######Import Data##########
TrainFile = pd.read_csv(r"C:\Users\enam1\Documents\dataanalysis\Titanic\titanic_train.csv")
TestFile = pd.read_csv(r"C:\Users\enam1\Documents\dataanalysis\Titanic\titanic_test.csv")
######## Data Discovery#########
=======
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
>>>>>>> fe3e175464ba66be8fa404d664c5eb07946a53f1

from sklearn.preprocessing import Imputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#Read the csv files and display the first 5 rows  of the training set
training_data = pandas.read_csv('../titanic_train.csv')
test_data = pandas.read_csv('../titanic_test.csv')


####################################################################################################

# 					Observation phase

####################################################################################################

training_data.head()

#Now we take a look at the features and the missing data in each column
print('features in training set: \n')
training_data.info()
print('\n')
print('features in testing set: \n')
test_data.info()

#This is the description of the dataset
training_data.describe()

g = seaborn.FacetGrid(training_data, col='Survived')
g.map(matplotlib.pyplot.hist, 'Age', bins=50)
#The number of males are bigger than the number of females but it seems that womens are more likely to survive
print(training_data.groupby(['Survived','Sex'])['Survived'].count())
training_data['Sex'].value_counts().plot.bar()

####################################################################################################

# 					Adaptation phase

####################################################################################################

#Females are privileged over mans, they have bigger survival rates
training_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

seaborn.factorplot('Sex', 'Survived', data = training_data, kind='bar')

#Encode categorical variables to numbers, that is for mathematical equations of the model that we gonna use(logistic_regression)
#But it is possible to not map them to numbers and use algorithms like decision trees or random forests
training_data.loc[training_data['Sex'] == 'male', 'Sex'] = 0
training_data.loc[training_data['Sex'] == 'female', 'Sex'] = 1

test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0
test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1

#The higher ticket class passengers have, the higher chances they have to survive
print(training_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
pandas.crosstab(training_data['Pclass'],training_data['Survived'],margins=True).style.background_gradient(cmap='summer_r')

#Here we create a new feature called : Family = Parch + SibSp
training_data['Family'] = training_data['Parch'] + training_data['SibSp']
test_data['Family'] = test_data['Parch'] + test_data['SibSp']

#We drop the features: Parch and SibSp: they are no longer useful
training_data = training_data.drop(['Parch'], axis = 1)
training_data = training_data.drop(['SibSp'], axis = 1)
test_data = test_data.drop(['Parch'], axis = 1)
test_data = test_data.drop(['SibSp'], axis = 1)

#When a passenger has a big family, he has less chances to survive
training_data[['Family', 'Survived']].groupby(['Family'], as_index=False).sum().sort_values(by='Survived', ascending=False)

#We continue by dropping the Ticket feature and the Cabin feature
training_data = training_data.drop(['Ticket'], axis = 1)
training_data = training_data.drop(['Cabin'], axis = 1)

test_data = test_data.drop(['Ticket'], axis = 1)
test_data= test_data.drop(['Cabin'], axis = 1)

#The port of embarkation is an important feature to decide whether the passenger will survive or not
training_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()

print('The minimum value in the fare feature is {}'.format(training_data['Fare'].min()))
print('The maximum value in the fare feature is {}'.format(training_data['Fare'].max()))
print('The mean value in the fare feature is {}'.format(training_data['Fare'].mean()))

#Feature scaling / Mean Normalization
training_data['Fare'] = (training_data['Fare'] - training_data['Fare'].mean())/training_data['Fare'].std()
test_data['Fare'] = (test_data['Fare'] - test_data['Fare'].mean())/test_data['Fare'].std()

#After removing these features, it's time to fill the missing values
imputer = Imputer(missing_values = numpy.nan, strategy = 'median', axis = 0)
training_data[['Age']] = imputer.fit_transform(training_data[['Age']])
test_data[['Age']] = imputer.fit_transform(test_data[['Age']])

####################################################################################################

# 					Encoding phase

####################################################################################################

training_data.loc[ training_data['Age'] <= 16, 'Age'] = 0
training_data.loc[(training_data['Age'] > 16) & (training_data['Age'] <= 32), 'Age'] = 1
training_data.loc[(training_data['Age'] > 32) & (training_data['Age'] <= 48), 'Age'] = 2
training_data.loc[(training_data['Age'] > 48) & (training_data['Age'] <= 64), 'Age'] = 3
training_data.loc[ training_data['Age'] > 64, 'Age'] = 4

test_data.loc[ test_data['Age'] <= 16, 'Age'] = 0
test_data.loc[(test_data['Age'] > 16) & (test_data['Age'] <= 32), 'Age'] = 1
test_data.loc[(test_data['Age'] > 32) & (test_data['Age'] <= 48), 'Age'] = 2
test_data.loc[(test_data['Age'] > 48) & (test_data['Age'] <= 64), 'Age'] = 3
test_data.loc[ test_data['Age'] > 64, 'Age'] = 4


#The embarked feature has only two missing values, we fill them with the most occured one
only_S = training_data[training_data['Embarked'] == 'S'].count()
print(only_S['Embarked']) #646
only_C = training_data[training_data['Embarked'] == 'C'].count()
print(only_C['Embarked']) #168
only_Q = training_data[training_data['Embarked'] == 'Q'].count()
print(only_Q['Embarked']) #77

#The most occured one is 'S'
training_data['Embarked'] = training_data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].fillna('S')

training_data.loc[training_data['Embarked'] == 'S', 'Embarked'] = 0
training_data.loc[training_data['Embarked'] == 'C', 'Embarked'] = 1
training_data.loc[training_data['Embarked'] == 'Q', 'Embarked'] = 2

test_data.loc[test_data['Embarked'] == 'S', 'Embarked'] = 0
test_data.loc[test_data['Embarked'] == 'C', 'Embarked'] = 1
test_data.loc[test_data['Embarked'] == 'Q', 'Embarked'] = 2

#The fare feature has missing data in the test set
test_data[['Fare']] = imputer.fit_transform(test_data[['Fare']])

#Here, we deal with the Name feature
#We first extract the title that we save in a new feature called: 'Title' 
#We drop the rest of the name
training_data['Title'] = training_data['Name'].str.split(', ', expand=True)[1].str.split('. ', expand=True)[0]
test_data['Title'] = test_data['Name'].str.split(', ', expand=True)[1].str.split('. ', expand=True)[0]

#We don't need this feature anymore now
training_data = training_data.drop(['Name'], axis = 1)
test_data = test_data.drop(['Name'], axis = 1)
#These are the all the titles of the training set and the test set
print(training_data['Title'].unique())
print(test_data['Title'].unique())

training_data.head()

print(training_data['Title'].value_counts())
training_data['Title'].value_counts().plot.bar()

training_data[training_data['Title'] == 'Mr']['Survived'].value_counts()

training_data[training_data['Title'] == 'Miss']['Survived'].value_counts()

training_data['Title'] = training_data['Title'].map({'Mr' : 0 , 'Master' : 3, 'Don' : 4, 'Major' : 4, 'Sir' : 4, 
                                           'Mrs' : 2 , 'Miss' : 1, 'Mme' : 4, 'Ms' : 4, 'Lady' : 4, 'Mlle': 4, 
                                           'Rev' : 4 , 'Col' : 4, 'Capt' : 4, 'th' : 4, 'Jonkheer' : 4, 'Dr' : 4})

test_data['Title'] = test_data['Title'].map({'Mr' : 0 , 'Master' : 3, 'Don' : 4, 'Major' : 4, 'Sir' : 4, 
                                           'Mrs' : 2 , 'Miss' : 1, 'Mme' : 4, 'Ms' : 4, 'Lady' : 4, 'Mlle': 4, 
                                           'Rev' : 4 , 'Col' : 4, 'Capt' : 4, 'th' : 4, 'Jonkheer' : 4, 'Dr' : 4, 'Dona' : 4})
                                         

# Any results you write to the current directory are saved as output.

#This is how our dataset looks like now
training_data.head()

X_train = training_data.drop(['Survived'], axis = 1)
Y_train = training_data[['Survived']]
X_test = test_data

X_train.shape, Y_train.shape, X_test.shape
X_train.head()

####################################################################################################

# 					prediciton phase

####################################################################################################

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
>>>>>>> 514ec8006851e92543050a95d80e5f6c978c5fdd
