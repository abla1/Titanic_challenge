<<<<<<< HEAD
####################################################################################################

# 					Packages Imports

############################################################################################################ Imports ########
import numpy
import pandas
import seaborn
import matplotlib.pyplot
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
####################################################################################################

# 					Loading Data

####################################################################################################
TrainFile = pd.read_csv(r"C:\Users\enam1\Documents\dataanalysis\Titanic\titanic_train.csv")
TestFile = pd.read_csv(r"C:\Users\enam1\Documents\dataanalysis\Titanic\titanic_test.csv")

full_data = [TrainFile, TestFile]
####################################################################################################

# 					Observation phase

####################################################################################################

training_data.head()
TrainFile.info()
TrainFile.describe()


## Looking at how many values are missing
print('\033[1m'+"Checking if train_df contains any null value:-"+'\033[0m')
print(TrainFile.isnull().sum())
print('\n')
print('\033[1m'+"Checking if test_df contains any null value:-"+'\033[0m')
print(TestFile.isnull().sum())


#Pairing for any relationships visible
g = sns.pairplot(TrainFile[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',
       u'SibSp']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])


##Another form of correlation for visualization
sns.heatmap(TrainFile.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()

## Looking at Age vs Survived
facet = sns.FacetGrid(TrainFile, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, TrainFile['Age'].max()))
facet.add_legend()
plt.show()

Lived = TrainFile[TrainFile["Survived"] == 1]
Died = TrainFile[TrainFile["Survived"] == 0]
Lived["Age"].plot.hist(alpha=0.5,color='green',bins=30)
Died["Age"].plot.hist(alpha=0.5,color='red',bins=30)
plt.legend(['Lived','Died'])
plt.show()


## Looking at Fare vs Survived
Lived["Fare"].plot.hist(alpha=0.5,color='green',bins=30)
Died["Fare"].plot.hist(alpha=0.5,color='red',bins=30)
plt.legend(['Lived','Died'])
plt.show()

facet = sns.FacetGrid(TrainFile, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, TrainFile['Fare'].max()))
facet.add_legend()
plt.show()

##Zooming in to see graph for first 200
facet = sns.FacetGrid(TrainFile, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, TrainFile['Fare'].max()))
facet.add_legend()
plt.xlim(0, 200)

######Family
Lived["SibSp"].plot.hist(alpha=0.5,color='green',bins=30)
Died["SibSp"].plot.hist(alpha=0.5,color='red',bins=30)
plt.legend(['Lived','Died'])
plt.show()


Lived["Parch"].plot.hist(alpha=0.5,color='green',bins=30)
Died["Parch"].plot.hist(alpha=0.5,color='red',bins=30)
plt.legend(['Lived','Died'])
plt.show()

##Since values are messy, adapting data here by adding both sibs and parch to make family size
##Make family size from parents and siblings data
TrainFile["FamilySize"] = TrainFile["SibSp"] + TrainFile["Parch"] + 1
TestFile["FamilySize"] = TestFile["SibSp"] + TestFile["Parch"] + 1

facet = sns.FacetGrid(TrainFile, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, TrainFile['FamilySize'].max()))
facet.add_legend()
plt.show()

############ Gender
TrainFile["Survived"].value_counts().plot.pie(figsize = (4, 4),
                                        autopct= '%.2f',
                                        fontsize = 10,
                                        title = "Pie Chart of Survival on Ship")
										
										
sns.countplot(x="Survived", hue="Sex", data=TrainFile)

facet = sns.FacetGrid(TrainFile, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Sex',shade= True)
facet.set(xlim=(0, TrainFile['Sex'].max()))
facet.add_legend()
plt.show()

##Embarking Location
:
TrainFile["Embarked"].describe()


facet = sns.FacetGrid(TrainFile, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Embarked',shade= True)
facet.set(xlim=(0, TrainFile['Embarked'].max()))
facet.add_legend()
plt.show()


##Cabin
facet = sns.FacetGrid(TrainFile, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Cabin',shade= True)
facet.set(xlim=(0, TrainFile['Cabin'].max()))
facet.add_legend()
plt.show()

##Title
##Have to translate and label to get values for observation
TrainFile['Title'] = TrainFile['Name'].str.split(', ', expand=True)[1].str.split('. ', expand=True)[0]
TestFile['Title'] = TestFile['Name'].str.split(', ', expand=True)[1].str.split('. ', expand=True)[0]

TrainFile['Title'].value_counts().plot.bar()

TrainFile[TrainFile['Title'] == 'Mr']['Survived'].value_counts()

TrainFile[TrainFile['Title'] == 'Miss']['Survived'].value_counts()

facet = sns.FacetGrid(TrainFile, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Title',shade= True)
facet.set(xlim=(0, TrainFile['Title'].max()))
facet.add_legend()
plt.show()



####################################################################################################

# 					Adaptation phase

####################################################################################################

##Gender
num = LabelEncoder()
TrainFile["Sex"] = num.fit_transform(TrainFile["Sex"].astype("str"))
TestFile["Sex"] = num.fit_transform(TestFile["Sex"].astype("str"))



##AGE
print('Highest Age:',TrainFile['Age'].max(),'   Lowest Age:',TrainFile['Age'].min())
TestFile['Age'].fillna(TestFile['Age'].median(), inplace = True)
TrainFile['Age'].fillna(TrainFile['Age'].median(), inplace = True)


###33##FARE
## Only test file had missing values
TestFile['Fare'].fillna(TestFile['Fare'].median(), inplace = True)

#######Family Members
#No adaptation needed

##Embarked
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
	

	
######Cabin
for dataset in full_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
    
TestFile['Cabin'].fillna(TestFile['Cabin'].median(), inplace = True)
TrainFile['Cabin'].fillna(TrainFile['Cabin'].median(), inplace = True)


####################################################################################################

# 					Encoding phase

####################################################################################################

#Group Age to make it easier for classification since it looks like there is a clear delineation between the ages and what it does.

for dataset in full_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
	
####FARE
##Grouping and encoding

for dataset in full_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[(dataset['Fare'] > 100) & (dataset['Fare'] <= 160), 'Fare'] = 3,
    dataset.loc[ dataset['Fare'] > 160, 'Fare'] = 4


##Family
family = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in full_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family)


##embarking
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
	
	
	
	
####Cabin
for dataset in full_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
    
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in full_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)



###	Title

TrainFile['Title'] = TrainFile['Title'].map({'Mr' : 0 , 'Master' : 3, 'Don' : 4, 'Major' : 4, 'Sir' : 4, 
                                           'Mrs' : 2 , 'Miss' : 1, 'Mme' : 4, 'Ms' : 4, 'Lady' : 4, 'Mlle': 4, 
                                           'Rev' : 4 , 'Col' : 4, 'Capt' : 4, 'th' : 4, 'Jonkheer' : 4, 'Dr' : 4})

TestFile['Title'] = TestFile['Title'].map({'Mr' : 0 , 'Master' : 3, 'Don' : 4, 'Major' : 4, 'Sir' : 4, 
                                           'Mrs' : 2 , 'Miss' : 1, 'Mme' : 4, 'Ms' : 4, 'Lady' : 4, 'Mlle': 4, 
                                           'Rev' : 4 , 'Col' : 4, 'Capt' : 4, 'th' : 4, 'Jonkheer' : 4, 'Dr' : 4, 'Dona' : 4})
	
	

##Cleaning up rest of data to make it ready for model classification training
features_to_drop = ['Ticket', 'Name', 'SibSp', 'Parch']
test_data = test.drop("PassengerId", axis=1).copy()
train = TrainFile.drop(features_to_drop, axis=1)
test = TestFile.drop(features_to_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape
####################################################################################################

# 					Models

####################################################################################################

##KNN
knn = KNeighborsClassifier(n_neighbors = 13)
knnscore = cross_val_score(knn, train_data, target, cv=k_fold, n_jobs=1, scoring="accuracy")
round(np.mean(knnscore)*100, 2)

##Decision Tree
Dec = DecisionTreeClassifier()
Decscore = cross_val_score(Dec, train_data, target, cv=k_fold, n_jobs=1, scoring="accuracy")
round(np.mean(Decscore)*100, 2)

##Naiive Bayes
NB = GaussianNB()
NBscore = cross_val_score(NB, train_data, target, cv=k_fold, n_jobs=1, scoring="accuracy")
round(np.mean(NBscore)*100, 2)

##Random Forest
rafo = RandomForestClassifier(n_estimators=100)
rafoscore = cross_val_score(rafo, train_data, target, cv=k_fold, n_jobs=1, scoring="accuracy")
round(np.mean(rafoscore)*100, 2)

##SVM
clf = SVC(kernel = 'linear')
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring="accuracy")
round(np.mean(score)*100, 2)

####################################################################################################

# 					prediciton phase

####################################################################################################

#It's time to predict

svc = SVC(kernel = 'linear')
svc.fit(train_data, test_data)

prediction = clf.predict(test_data)



random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred3 = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
>>>>>>> 514ec8006851e92543050a95d80e5f6c978c5fdd
