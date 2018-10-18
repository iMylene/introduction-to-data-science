# Kickstart project
# Group D: 
# - Mylène Martodihardjo
# - Rik Timer
# - Vincent van de Langenberg
# - Pra Jiawan



### Import necessary modules and functions
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 8
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

### Import the data files into a Pandas DataFrame
df_train = pd.read_csv('KS_train_data.csv',delimiter=',')
df_test = pd.read_csv('KS_test_data.csv',delimiter=',')

subcat = df_train.groupby(['subcategory'])['subcategory'].count()
fundedSubcats = df_train.groupby(['subcategory'])['funded'].sum()
subcatRatio = fundedSubcats/subcat

# Get ratio from (sub)category or country
# Get ratio from (sub)category or country
def GetRatio(name,what):
    if what == 'subcat':
        return subcatRatio[name]
    elif what == 'country':
        return countryRatio[name]

df_train['subcatRatio'] = df_train['subcategory'].apply(lambda x: GetRatio(x,'subcat'))
df_test['subcatRatio'] = df_test['subcategory'].apply(lambda x: GetRatio(x,'subcat'))

# Fill in the missing country values
df_train.loc[df_train['country'].isnull(),'country']
df_train['country'] = df_train['country'].fillna('?');

# Make new country_numeric column, represented by a numeric value
df_train.country = pd.Categorical(df_train.country)
df_train['country_numeric'] = df_train.country.cat.codes

# Funded project in a country / amount of projects in a country
sumProjPerCountry = df_train.groupby(['country'])['country'].count()
fundedProjPerCountry = df_train.groupby(['country'])['funded'].sum()
countryRatio = fundedProjPerCountry/sumProjPerCountry
df_train['countryRatio'] = df_train['country'].apply(lambda x: GetRatio(x,'country'))


showTheseCols = ['funded','staff_pick','subcatRatio','countryRatio','fx_rate','launched_at','goal']
sns.heatmap(df_train[showTheseCols].corr(), square=True, annot=True);
plt.savefig('heatmap.png')

### Set trainingset and df_test
trainSize = 80000
testSize =  100000-trainSize
trainSet = df_train.sample(trainSize,replace=False)
testSet =  df_test.sample(testSize,replace=False)

cols = ['subcatRatio', 'staff_pick']
x_train = trainSet[cols].values
y_train = trainSet['funded']
x_test  = df_test[cols].values  


KNN = KNeighborsClassifier(n_neighbors=7)
KNN.fit(x_train,y_train)
predictKNN = KNN.predict(x_train)
print (KNN.score(x_train,y_train))

df_test['fundedEC'] = KNN.predict(x_test)
y_testEC = df_test['fundedEC']
logLossKNN = log_loss(y_train,predictKNN,normalize=True)
print(logLossKNN)

df_test.to_csv('predictionsEC.tsv', sep='\t', columns=['project_id','fundedEC'])

### Method 2: Logistic Regression
# Instantiate and fit a Logistic Regression model, then print its score
modelLR = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
modelLR.fit(x_train,y_train)
trainPredictLR = modelLR.predict(x_train)
print (modelLR.score(x_train,y_train))

df_test['fundedLR'] = modelLR.predict(x_test)

y_testLR = df_test['fundedLR']

logLossResultsLR = log_loss(y_train,trainPredictLR,normalize=True)
print(logLossResultsLR)

df_test.to_csv('predictionsLR.tsv', sep='\t', columns=['project_id','fundedLR'])
###


resultMatrixEC = confusion_matrix(y_train,predictKNN, labels=[True, False])
resultMatrixLR = confusion_matrix(y_train,trainPredictLR, labels=[True, False])

print(resultMatrixEC)
print(resultMatrixLR)
