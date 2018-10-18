# Kickstart project
# Group D: 
# - Mylène Martodihardjo
# - Rik Timer
# - Vincent van de Langenberg
# - Pra Jiawan



### Import necessary modules and functions
import sklearn.datasets
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 8

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')
###



### Import the data files into a Pandas DataFrame
df_train = pd.read_csv('KS_train_data.csv',delimiter=',')
#df_train = pd.read_csv('/home/rik/Downloads/KS_train_data.csv', delimiter=',')

df_test = pd.read_csv('KS_test_data.csv',delimiter=',')
#df = pd.read_csv('/home/rik/Downloads/KS_test_data.csv', delimiter=',')
###



### Make methods
# Get continent from country
def GetConti(country):
    if country in UN:
        return "?"
    elif country in AF:
        return "AF"
    elif country in AN:
        return "AN"
    elif country in AS:
        return "AS"
    elif country in EU:
        return "EU"
    elif country in NA:
        return "NA"
    elif country in OC:
        return "OC"
    elif country in SA:
        return "SA"
    else:
        return "other"
    
# Get ratio from (sub)category or country
def GetRatio(name,what):
    if what == 'subcat':
        return subcatRatio[name]
    elif what == 'cat':
        return catRatio[name]
    elif what == 'country':
        return countryRatio[name]
    elif what == 'continent':
        return continentRatio[name]
###
  

      
### Column options
# The imported dataframe's column headers
#cols = df.columns
cols_train = df_train.columns
cols_test = df_test.columns

# The removed columns in the testdata
notInTestCols = ['funded', 'pledged', 'usd_pledged', 'converted_pledged_amount', 'backers_count']

# The columns containing unix timestamps
unixcols = ['created_at','deadline', 'launched_at']

# Show these cols for check
showTheseCols = list(cols)
showTheseCols = [x for x in showTheseCols if x not in notInTestCols[1:5]]

# The cols we keep (in test data set) = cols + extra made
to_keep_for_training = list(cols_train)
to_keep_for_training.extend(['category_numeric','country_numeric','continent_numeric','subcatRatio','catRatio','countryRatio','continentRatio'])
to_keep_for_training = [x for x in to_keep_for_training if x not in notInTestCols[1:5]]   

#elke ratio kan wel: want funded bestaat niet
to_keep_for_test = list(cols_test)
to_keep_for_test.extend(['category_numeric','country_numeric','continent_numeric'])#,'subcatRatio','catRatio','countryRatio','continentRatio'])
to_keep_for_test = [x for x in to_keep_for_test if x not in notInTestCols[1:5]]   
###    



### Necessary lists 
# Africa
AF = ['AO','BF','BI','BJ','BW','CD','CF','CG','CI','CM','CV','DJ','DZ','EG','EH','ER','ET','GA','GH','GM','GN','GQ','GW','KE','KM','LR','LS','LY','MA','MG','ML','MR','MU','MW','MZ','NA','NE','NG','RE','RW','SC','SD','SH','SL','SN','SO','SS','ST','SZ','TD','TG','TN','TZ','UG','YT','ZA','ZM','ZW']

# Antarctica
AN = ['AQ','BV','GS','HM','TF']

# Asia
AS = ['AE','AF','AM','AP','AZ','BD','BH','BN','BT','CC','CN','CX','CY','GE','HK','ID','IL','IN','IO','IQ','IR','JO','JP','KG','KH','KP','KR','KW','KZ','LA','LB','LK','MM','MN','MO','MV','MY','NP','OM','PH','PK','PS','QA','SA','SG','SY','TH','TJ','TL','TM','TW','UZ','VN','YE']

# Europe
EU = ['AD','AL','AT','AX','BA','BE','BG','BY','CH','CZ','DE','DK','EE','ES','EU','FI','FO','FR','FX','GB','GG','GI','GR','HR','HU','IE','IM','IS','IT','JE','LI','LT','LU','LV','MC','MD','ME','MK','MT','NL','NO','PL','PT','RO','RS','RU','SE','SI','SJ','SK','SM','TR','UA','VA','XK']

# North America
NA = ['AG','AI','AN','AW','BB','BL','BM','BS','BZ','CA','CR','CU','DM','DO','GD','GL','GP','GT','HN','HT','JM','KN','KY','LC','MF','MQ','MS','MX','NI','PA','PM','PR','SV','TC','TT','US','VC','VG','VI']

# Oceania
OC = ['AS','AU','CK','FJ','FM','GU','KI','MH','MP','NC','NF','NR','NU','NZ','PF','PG','PN','PW','SB','TK','TO','TV','UM','VU','WF','WS']

# South America
SA = ['AR','AW','BO','BR','BQ','CL','CO','CW', 'EC','FK','GF','GY','PE','PY','SR','SX','UY','VE']

# Unknown Country, Unknown Continent: in case you did for got to run the country code, nan is also here.
UN = ['?',np.nan]    
###

### Plots
# The Pearson correlation matrix assumes continuous variables.
sns.heatmap(df[showTheseCols].corr(), square = True, annot = True);
sns.clustermap(df[showTheseCols].corr());
sns.heatmap(df[to_keep].corr(), square = True, annot = True);
sns.clustermap(df[to_keep].corr());
###


### Handle values: missing values and creating new numeric values
# Fill in the missing country values
#df.loc[df['country'].isnull(),'country']
#df['country'] = df['country'].fillna('?');

df_train.loc[df_train['country'].isnull(),'country']
df_train['country'] = df_train['country'].fillna('?');

df_test.loc[df_test['country'].isnull(),'country']
df_test['country'] = df_test['country'].fillna('?');

# Make new category_numeric column, represented by a numeric value
#df.category = pd.Categorical(df.category)
#df['category_numeric'] = df.category.cat.codes

df_train.category = pd.Categorical(df_train.category)
df_train['category_numeric'] = df_train.category.cat.codes

df_test.category = pd.Categorical(df_test.category)
df_test['category_numeric'] = df_test.category.cat.codes

# Make new country_numeric column, represented by a numeric value
#df.country = pd.Categorical(df.country)
#df['country_numeric'] = df.country.cat.codes

df_train.country = pd.Categorical(df_train.country)
df_train['country_numeric'] = df_train.country.cat.codes

df_test.country = pd.Categorical(df_test.country)
df_test['country_numeric'] = df_test.country.cat.codes

# Make new continent_numeric column, remove unknowns
#df['continent_numeric'] = df['country'].apply(lambda x: GetConti(x))
#df = df[df.continent_numeric != '?']
#df = df[df.continent_numeric != 'other']
#df.continent_numeric = pd.Categorical(df.continent_numeric)
#df['continent_numeric'] = df.continent_numeric.cat.codes

df_train['continent_numeric'] = df_train['country'].apply(lambda x: GetConti(x))
df_train = df_train[df_train.continent_numeric != '?']
df_train = df_train[df_train.continent_numeric != 'other']
df_train.continent_numeric = pd.Categorical(df_train.continent_numeric)
df_train['continent_numeric'] = df_train.continent_numeric.cat.codes

df_test['continent_numeric'] = df_test['country'].apply(lambda x: GetConti(x))
df_test = df_test[df_test.continent_numeric != '?']
df_test = df_test[df_test.continent_numeric != 'other']
df_test.continent_numeric = pd.Categorical(df_test.continent_numeric)
df_test['continent_numeric'] = df_test.continent_numeric.cat.codes

# Funded subcategory / amount in a subcategory ratio
#subcat = df.groupby(['subcategory'])['subcategory'].count()
#fundedSubcats = df.groupby(['subcategory'])['funded'].sum()
#subcatRatio = fundedSubcats/subcat
#df['subcatRatio'] = df['subcategory'].apply(lambda x: GetRatio(x,'subcat'))
#subcatList = df['subcategory'].unique()

subcat = df_train.groupby(['subcategory'])['subcategory'].count()
fundedSubcats = df_train.groupby(['subcategory'])['funded'].sum()
subcatRatio = fundedSubcats/subcat
df_train['subcatRatio'] = df_train['subcategory'].apply(lambda x: GetRatio(x,'subcat'))

#subcat = df_test.groupby(['subcategory'])['subcategory'].count()
#fundedSubcats = df_test.groupby(['subcategory'])['funded'].sum()
#subcatRatio = fundedSubcats/subcat
#df_test['subcatRatio'] = df_test['subcategory'].apply(lambda x: GetRatio(x,'subcat'))

# Funded category / amount in a category ratio
#cat = df.groupby(['category'])['category'].count()
#fundedCats = df.groupby(['category'])['funded'].sum()
#catRatio = fundedCats/cat
#df['catRatio'] = df['category'].apply(lambda x: GetRatio(x,'cat'))

cat = df_train.groupby(['category'])['category'].count()
fundedCats = df_train.groupby(['category'])['funded'].sum()
catRatio = fundedCats/cat
df_train['catRatio'] = df_train['category'].apply(lambda x: GetRatio(x,'cat'))

#cat = df_test.groupby(['category'])['category'].count()
#fundedCats = df_test.groupby(['category'])['funded'].sum()
#catRatio = fundedCats/cat
#df_test['catRatio'] = df_test['category'].apply(lambda x: GetRatio(x,'cat'))

# Funded project in a country / amount of projects in a country ratio
#sumProjPerCountry = df.groupby(['country'])['country'].count()
#fundedProjPerCountry = df.groupby(['country'])['funded'].sum()
#countryRatio = fundedProjPerCountry/sumProjPerCountry
#df['countryRatio'] = df['country'].apply(lambda x: GetRatio(x,'country'))

sumProjPerCountry = df_train.groupby(['country'])['country'].count()
fundedProjPerCountry = df_train.groupby(['country'])['funded'].sum()
countryRatio = fundedProjPerCountry/sumProjPerCountry
df_train['countryRatio'] = df_train['country'].apply(lambda x: GetRatio(x,'country'))

#sumProjPerCountry = df_test.groupby(['country'])['country'].count()
#fundedProjPerCountry = df_test.groupby(['country'])['funded'].sum()
#countryRatio = fundedProjPerCountry/sumProjPerCountry
#df_test['countryRatio'] = df_test['country'].apply(lambda x: GetRatio(x,'country'))

# Funded project in a continent / amount of projects in a continent ratio
#sumProjPerContinent = df.groupby(['continent_numeric'])['continent_numeric'].count()
#fundedProjPerContinent = df.groupby(['continent_numeric'])['funded'].sum()
#continentRatio = fundedProjPerContinent/sumProjPerContinent
#df['continentRatio'] = df['continent_numeric'].apply(lambda x: GetRatio(x,'continent'))

sumProjPerContinent = df_train.groupby(['continent_numeric'])['continent_numeric'].count()
fundedProjPerContinent = df_train.groupby(['continent_numeric'])['funded'].sum()
continentRatio = fundedProjPerContinent/sumProjPerContinent
df_train['continentRatio'] = df_train['continent_numeric'].apply(lambda x: GetRatio(x,'continent'))

#sumProjPerContinent = df_test.groupby(['continent_numeric'])['continent_numeric'].count()
#fundedProjPerContinent = df_test.groupby(['continent_numeric'])['funded'].sum()
#continentRatio = fundedProjPerContinent/sumProjPerContinent
#df_test['continentRatio'] = df_test['continent_numeric'].apply(lambda x: GetRatio(x,'continent'))
###


### Set trainingset and df_test
trainSize = 20000 #len(df_train)
testSize =  100000-trainSize #70000 #len(df_test)

#verwijder dit voor echte datasets
df = df_train
df_train = df.sample(trainSize, replace=False) #df_train.sample(trainSize, replace=False)
df_test  = df.sample(testSize, replace=False) #df_test.sample(testSize, replace=False)
to_keep_for_training = ['subcatRatio', 'catRatio']
to_keep_for_test = ['subcatRatio', 'catRatio']
#

x_test  = df_test[to_keep_for_training].values  
y_test  = df_test['funded']

x_train = df_train[to_keep_for_test].values
y_train = df_train['funded']
###

### Method 1:  Evolutionary Computation
# Instantiate and fit an Evolutionary Computation model, then print its score
modelEC = KNeighborsClassifier(n_neighbors=7)
modelEC.fit(x_train,y_train)
print (modelEC.score(x_train,y_train))

#predEC_train = modelEC.predict(x_train)
predEC_test = modelEC.predict(x_test)
#print(predEC_train)
#print(predEC_test)

scoring = modelEC.score(x_test,y_test)
#print(scoring)
###



### Method 2: Logistic Regression
# Instantiate and fit a Logistic Regression model, then print its score
modelLR = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
modelLR.fit(x_train,y_train)
print (modelLR.score(x_train,y_train))

# NearestNeighbors
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# h = .02  # step size in the mesh
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

#predLR_train = modelLR.predict(x_train)
predLR_test = modelLR.predict(x_test)

#print(modelLR.predict_proba(x_test))
#print(predLR_train)
#print(predLR_test)
###