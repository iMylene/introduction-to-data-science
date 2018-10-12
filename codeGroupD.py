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

from scipy.stats import norm
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')
###



### Import the data files into a Pandas DataFrame
df = pd.read_csv('KS_train_data.csv',delimiter=',')
#df_train = pd.read_csv('KS_train_data.csv',delimiter=',')
#df = pd.read_csv('/home/rik/Downloads/KS_train_data.csv', delimiter=',')

df_test = pd.read_csv('KS_test_data.csv',delimiter=',')
#df = pd.read_csv('/home/rik/Downloads/KS_train_data.csv', delimiter=',')
###



### Make functions
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
cols = df.columns

# The removed columns in the testdata
notInTestCols = ['funded', 'pledged', 'usd_pledged', 'converted_pledged_amount', 'backers_count']

# The columns containing unix timestamps
unixcols = ['created_at','deadline', 'launched_at']

# The used cols (in test data set) = cols + extra made
usedCols = list(cols)
usedCols.extend(['category_numeric','country_numeric','continent_numeric','subcatRatio','catRatio','countryRatio','continentRatio'])
usedCols = [x for x in usedCols if x not in notInTestCols[1:5]]    
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



# The Pearson correlation matrix assumes continuous variables.
######sns.heatmap(df[cols].corr(), square = True, annot = True);
######sns.clustermap(df.corr());
###



### Handle values: missing values and creating new numeric values
# Fill in the missing country values
df.loc[df['country'].isnull(),'country']
df['country'] = df['country'].fillna('?');

# Make new category_numeric column, represented by a numeric value
df.category = pd.Categorical(df.category)
df['category_numeric'] = df.category.cat.codes

# Make new country_numeric column, represented by a numeric value
df.country = pd.Categorical(df.country)
df['country_numeric'] = df.country.cat.codes

# Make new continent_numeric column, remove unknowns
df['continent_numeric'] = df['country'].apply(lambda x: GetConti(x))
df = df[df.continent_numeric != '?']
df = df[df.continent_numeric != 'other']
df.continent_numeric = pd.Categorical(df.continent_numeric)
df['continent_numeric'] = df.continent_numeric.cat.codes

# Funded subcategory / amount in a subcategory ratio
subcat = df.groupby(['subcategory'])['subcategory'].count()
fundedSubcats = df.groupby(['subcategory'])['funded'].sum()
subcatRatio = fundedSubcats/subcat
df['subcatRatio'] = df['subcategory'].apply(lambda x: GetRatio(x,'subcat'))
#subcatList = df['subcategory'].unique()

# Funded category / amount in a category ratio
cat = df.groupby(['category'])['category'].count()
fundedCats = df.groupby(['category'])['funded'].sum()
catRatio = fundedCats/cat
df['catRatio'] = df['category'].apply(lambda x: GetRatio(x,'cat'))

# Funded project in a country / amount of projects in a country ratio
sumProjPerCountry = df.groupby(['country'])['country'].count()
fundedProjPerCountry = df.groupby(['country'])['funded'].sum()
countryRatio = fundedProjPerCountry/sumProjPerCountry
df['countryRatio'] = df['country'].apply(lambda x: GetRatio(x,'country'))

# Funded project in a continent / amount of projects in a continent ratio
sumProjPerContinent = df.groupby(['continent_numeric'])['continent_numeric'].count()
fundedProjPerContinent = df.groupby(['continent_numeric'])['funded'].sum()
continentRatio = fundedProjPerContinent/sumProjPerContinent
df['continentRatio'] = df['continent_numeric'].apply(lambda x: GetRatio(x,'continent'))
###



testset = df.sample(n=100, replace=False)

cols = ['subcatRatio', 'catRatio']
newdf = testset[cols].values

print(newdf)

# X = newdf
# y = testset['funded']
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold
# paramgrid = {"kernel": ["rbf"],
#              "C"     : np.logspace(-9, 9, num=25, base=10),
#              "gamma" : np.logspace(-9, 9, num=25, base=10)}
# random.seed(1)
#
# from evolutionary_search import EvolutionaryAlgorithmSearchCV
# cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
#                                    params=paramgrid,
#                                    scoring="accuracy",
#                                    cv=StratifiedKFold(n_splits=4),
#                                    verbose=1,
#                                    population_size = 100,
#                                    gene_mutation_prob = 0.10,
#                                    gene_crossover_prob= 0.5,
#                                    tournament_size=3,
#                                    generations_number=10,
#                                    n_jobs=4)
#
# cv.fit(X, y)
X = testset[cols].values  # we only take the first two features.
Y = testset['funded']

logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('SubcatRatio')
plt.ylabel('CatRatio')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()