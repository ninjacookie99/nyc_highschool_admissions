"""
Created on Tue May  4 14:12:04 2021

@author: sahil
"""

#%%     Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn import linear_model # library for multiple linear regression
from sklearn.metrics import mean_squared_error # to get the RMSE score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%%     Importing our data

# Import using pandas method
df = pd.read_csv ('middleSchoolData.csv') 

df = df.replace('^\s*$', np.nan, regex=True) # change whitespaces to nans if any

# drop unwanted columns
df.drop('dbn',axis = 1,inplace = True) # drop the dbn column
df.drop('school_name',axis = 1,inplace = True) # drop the school name column

# store rows and columns
num_rows, num_cols = df.shape #to store number of rows and columns

#%%     First glance of the data / Exploratory Data Analysis

descriptiveStatistics = df.describe()
descriptiveStatistics = descriptiveStatistics[['applications', 'acceptances', 
                                               'per_pupil_spending','avg_class_size',
                                               'school_size',
                                               'student_achievement',
                                               'math_scores_exceed']]
descriptiveStatistics.to_excel('descriptiveStatistics.xlsx')

# checking what proportion of the data is missing
count = 0
for r in range(num_rows):
    for c in range(num_cols):
        if np.isnan(df.iloc[r,c]) == True:
            count += 1
# number of missing observations
print(f"There are {count} number of observations that are missing altogether")

# proportion of missing data
propMissing = count / num_rows * num_cols # 22.48% percent of the data is missing
print(f"{propMissing:.2f}% of the data is missing")

plt.style.use('ggplot') # use ggplot style
# first four variables
plt.figure(figsize=(8,5))
for i,title in enumerate(df.columns[:4]):
    plt.subplot(2,2,i+1)
    plt.hist(df.iloc[:,i],bins = 50,color = 'red')
    plt.ylabel('Count',fontsize = 10)
    plt.title(title.title().replace('_'," "),fontsize = 11)
    plt.xticks(fontsize=11, rotation=45)
plt.tight_layout()
plt.show()

# demographic data
plt.figure(figsize=(8,5))
for x,title in enumerate(df.columns[4:9]):
    plt.subplot(2,3,x+1) 
    plt.hist(df.iloc[:,x+4],bins = 50,color = 'blue')
    plt.ylabel('Count',fontsize = 10)
    plt.title(title.title().replace('_'," "),fontsize = 11)
    plt.xticks(fontsize=14, rotation=45)
plt.tight_layout()
plt.show()

# School Climate
plt.figure(figsize=(8,5))
for j,title in enumerate(df.columns[9:15]):
    plt.subplot(2,3,j+1) 
    plt.hist(df.iloc[:,j+9],bins = 50,color = 'green')
    plt.ylabel('Count',fontsize = 10)
    plt.title(title.title().replace('_'," "),fontsize = 11)
    plt.xticks(fontsize=14, rotation=45)
plt.tight_layout()
plt.show()

# School size and measures of achievement
plt.figure(figsize=(8,5))
for j,title in enumerate(df.columns[18:]):
    plt.subplot(2,3,j+1) 
    plt.hist(df.iloc[:,j+18],bins = 50,color = 'purple')
    plt.ylabel('Count',fontsize = 10)
    plt.title(title.title().replace('_'," "),fontsize = 11)
    plt.xticks(fontsize=14, rotation=45)
plt.tight_layout()
plt.show()

#%%     Question 1

# apply a log transformation since the raw data is highly skewed
temp = np.log(df.iloc[:,:2]) # first two columns of the data frame, number of applications and admissions to HSPHS
temp[~np.isfinite(temp)] = np.nan # convert -inf to nan

# histograms to show effects of log transformation
plt.subplot(2,2,1)
plt.hist(df.iloc[:,0],bins = 50,color = 'purple') #before
plt.title('Raw Applications',fontsize = 10)
plt.subplot(2,2,2)
plt.hist(temp.iloc[:,0],bins = 50,color = 'purple') #after
plt.title('Log(Applications',fontsize = 10)
plt.subplot(2,2,3)
plt.hist(df.iloc[:,1],bins = 50,color = 'purple') #before
plt.title('Raw Admissions',fontsize = 10)
plt.subplot(2,2,4)
plt.hist(temp.iloc[:,1],bins = 50,color = 'purple') #after
plt.title('Log(Admissions)',fontsize = 10)
plt.tight_layout()
plt.show()

# Doesnt make much sense to do log transformation
# Stick to raw data

# Clean up the data in a temporary array
stacked_data = np.stack((df['applications'],df['acceptances']),axis = 1) # stack the required columns
cleaned_data =  stacked_data[~np.isnan(stacked_data).any(axis=1)] # remove nans across cols to get equal amount

# correlation between the number of applications and admissions to HSPHS?
r_applications_admissions = np.corrcoef(cleaned_data[:,0],cleaned_data[:,1])
print(f"Correlation between number of applications and number of admissions is: {r_applications_admissions[0,1]:.4f}")

# set up the data for linear regression object
X = np.transpose([cleaned_data[:,0]]) # number of applications
Y = cleaned_data[:,1] # number of admissions

regr = linear_model.LinearRegression() # linearRegression function from linear_model
temp2 = regr.fit(X,Y) # use fit method 

rSquared = regr.score(X,Y) #0.6427
betas = regr.coef_ # slopes
yInt = regr.intercept_  # y-intercept
yHat = yInt + betas[0]*cleaned_data[:,0] #predicted values


RMSE =  mean_squared_error(Y, yHat, squared=False)

# Scatter plot to show the relationship
plt.style.use('ggplot')
plt.scatter(cleaned_data[:,0],cleaned_data[:,1],marker = 'o')
plt.xlabel('Number of applications')
plt.ylabel('Number of acceptances')
plt.title(f"Correlation = {r_applications_admissions[0,1]:.3f}  RMSE = {RMSE:.2f}")
plt.plot(cleaned_data[:,0],yHat,color='blue',linewidth=0.7) # OLS fitted line
plt.show()

# to check statistical significance of the betas
# regression summary to check p values of coefficients
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary()) 

#%%     Question 2

# find the applications rate
df['applications_rate'] = (df['applications'] / df['school_size']) * 100

# clean the data in a temporary array
stacked_data = np.stack((df['applications_rate'],df['acceptances']),axis = 1)
cleaned_data =  stacked_data[~np.isnan(stacked_data).any(axis=1)]

# compute correlation
r_applicationsRate_admissions = np.corrcoef(cleaned_data[:,0],cleaned_data[:,1])
print(f"Correlation between applications rate and number of admissions is: {r_applicationsRate_admissions[0,1]:.4f}")

# set up the dat for linear regression object
X = np.transpose([cleaned_data[:,0]]) # applications_rate
Y = cleaned_data[:,1] # nubmer of admissions

regr = linear_model.LinearRegression() # linearRegression function from linear_model
temp = regr.fit(X,Y) # use fit method 

rSquared = regr.score(X,Y) # 0.5695005889825516
betas = regr.coef_ # slopes
yInt = regr.intercept_  # y-intercept
yHat = yInt + betas[0]*cleaned_data[:,0] #predicted values
RMSE =  mean_squared_error(Y, yHat, squared=False)

# Scatter plot to show the relationship
plt.scatter(cleaned_data[:,0],cleaned_data[:,1],marker = 'o',color='black')
plt.xlabel('Application rate')
plt.ylabel('Number of acceptances')
plt.title(f"Correlation = {r_applicationsRate_admissions[0,1]:.3f}  RMSE = {RMSE:.2f}")
plt.plot(cleaned_data[:,0],yHat,color='blue',linewidth=0.7) # OLS fitted line

# regression summary to check p values of coefficients
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary()) 

#%%     Question 3 

# Calculating per student odds
temp = pd.read_csv ('middleSchoolData.csv')
temp = temp.replace('^\s*$', np.nan, regex=True) # change whitespaces to nans if any

temp["acceptanceOdds"] = (temp['acceptances']/temp['school_size']) / (1 - (temp['acceptances']/temp['school_size']))
temp["acceptanceOdds"].replace(np.nan,0,inplace = True)
temp = temp.sort_values("acceptanceOdds",ascending=False,ignore_index = True)
print(temp.loc[0])

# Bar graph for decreasing rank of highest odds per student
x = list(temp.dbn.loc[:5])
y = list(temp.acceptanceOdds.loc[:5])
plt.figure(figsize=(10,5))
plt.bar(x, y,width=0.5)
plt.xticks(fontsize=10, rotation=65)
plt.ylabel('Odds')
plt.xlabel('School Code')
plt.title('Top 6 Per Student Odds of Admission to HSPS')
for i in range(len(x)):
    plt.text(i,np.round(y[i],decimals = 2),np.round(y[i],decimals = 2),ha="center",va="bottom")
plt.show()

#%%         Question 4

# using PCA and Correlation 

# extract a temporary data frame for the target variables
temp = df[['rigorous_instruction',
                    'collaborative_teachers', 
                    'supportive_environment',
                    'effective_school_leadership',
                    'strong_family_community_ties', 
                    'trust',
                    'student_achievement',
                    'reading_scores_exceed',
                    'math_scores_exceed']]

temp = temp[~np.isnan(temp).any(axis=1)] # drop nans row wise

# PCA for school climate indicators
school_climate = temp[['rigorous_instruction',
                    'collaborative_teachers', 
                    'supportive_environment',
                    'effective_school_leadership',
                    'strong_family_community_ties', 
                    'trust']].to_numpy()

r = np.corrcoef(school_climate,rowvar=False)
plt.imshow(r) 
plt.colorbar()
plt.show()

zscoredData = stats.zscore(school_climate)
pca = PCA()
pca.fit(zscoredData)
eigValues = pca.explained_variance_ 
covarExplained = eigValues/sum(eigValues)*100
loadings = pca.components_
origDataNewCoordinates = pca.fit_transform(zscoredData)

# Scree Plot 
numPredictors = school_climate.shape[1]
plt.style.use('ggplot')
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.plot([0,numPredictors],[1,1],color='black',linewidth=1) # Kaiser criterion line
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show() # one PC greater than 1

# Plotting values from the loadings matrix
# factor 1
plt.figure(figsize=(10,5))
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:],
        tick_label =['rigorous_instruction',
                    'collaborative_teachers', 
                    'supportive_environment',
                    'effective_school_leadership',
                    'strong_family_community_ties', 
                    'trust',]
        )
plt.title('Factor 1')
plt.xticks(fontsize=10, rotation=60)
plt.ylabel('Loadings')
plt.show() # looks like it points to trust, effective school leadership, and collaborative teachers

# old data in the PCA space
X = origDataNewCoordinates[:,0] # trust, effective school leadership, and collaborative teachers 

#%%     PCA for achivement indicators

achievement = temp[['student_achievement','reading_scores_exceed','math_scores_exceed']].to_numpy()

r = np.corrcoef(achievement,rowvar=False)
plt.imshow(r) 
plt.colorbar()
plt.show()

zscoredData = stats.zscore(achievement)
pca = PCA()
pca.fit(zscoredData)
eigValues = pca.explained_variance_ 
covarExplained = eigValues/sum(eigValues)*100
loadings = pca.components_
origDataNewCoordinates = pca.fit_transform(zscoredData)

# Scree plot
numPredictors = achievement.shape[1]
plt.style.use('ggplot')
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.plot([0,numPredictors],[1,1],color='black',linewidth=1) # Kaiser criterion line
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show() # only PC 1 greater than 1

# factor 1
plt.figure(figsize=(10,5))
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:],
        tick_label =['student_achievement','reading_scores_exceed','math_scores_exceed']
        )
plt.title('Factor 1')
plt.xticks(fontsize=10, rotation=60)
plt.ylabel('Loadings')
plt.show() # looks like it points to reading and math scores exceeded

# old data in the PCA space
Y =  origDataNewCoordinates[:,0] # reading and math scores exceeded

# compute the correlation between the two principle components
corr_XY = np.corrcoef(X,Y)

# scatter plot to show the relationship
plt.scatter(X,Y,marker = 'o',color='black')
plt.xlabel('Quality of Teachers')
plt.ylabel('Exceeding Math and Reading Scores')
plt.title(f"Correlation = {corr_XY[0,1]:.3f}")

# actual relationship between data
# correlation matrix
actual_correlation = temp.corr()
actual_correlation.to_excel('Q4.xlsx')


#%%     Question 5 

# test our hypothesis that rich schools are more likely to get into HSPS since they spend more on students and 
# have access to better resources, and better teachers. 
# Likewise, the middle school average student might also come from an affluent neighborhood meaning that
# they have a better learning environment, thus better chance of being accepted to a HSPS

median_spending = np.nanmedian(df['per_pupil_spending']) # median spending per student = $20147.0

# Rich and Large schools
df['rich_school'] = np.nan # preallocate with nans  23

# if over median spending per student , then rich school
df.loc[(df['per_pupil_spending'] > median_spending),'rich_school'] = 1
df.loc[(df['per_pupil_spending'] < median_spending),'rich_school'] = 0

temp = np.array([df['acceptances'],df['rich_school']]).T

temp = temp[~np.isnan(temp).any(axis=1)] # drop nans row wise

richSchools = np.array([]) # rich schools
poorSchools = np.array([]) # poor schools

# testing rich vs poor schools on average student achievement on a state-wide standardized test

for i in range(len(temp)): 
    if temp[i,1] == 0: # if poor school
        poorSchools  = np.append(poorSchools ,temp[i,0]) # acceptance value if poor
    else:
        richSchools = np.append(richSchools,temp[i,0]) # acceptance value if rich
        
# since we have unequal arrays
combinedData = np.transpose(np.array([richSchools,poorSchools])) # array of arrays

t1,p1 = stats.ttest_ind(combinedData[0],combinedData[1]) # t = -1.304978168054853, p = 0.19245099323340648
print("Difference in sample means:",np.mean(richSchools)- np.mean(poorSchools))
print(f"T-stat = {t1:.2f} and P-value = {p1:.2f}")


descriptivesContainer = np.empty([numMovies,4])
descriptivesContainer[:] = np.NaN 

x = ['Rich Schools', 'Poor Schools', 'Matrix 3'] # labels for the bars
xPos = np.array([1,2]) # x-values for the bars
plt.bar(xPos,descriptivesContainer[:,0],width=0.5,yerr=descriptivesContainer[:,3]) # bars + error
plt.xticks(xPos, x) # label the x_pos with the labels
plt.ylabel('Mean Acceptances') # add y-label
plt.title('t = {:.3f}'.format(t1) + ', p = {:.3f}'.format(p1)) # title is the test stat and p-value


#%%     Question 6

# one way
stacked_data = np.stack((df['per_pupil_spending'],df['acceptances']),axis = 1) # stack the required columns
cleaned_data =  stacked_data[~np.isnan(stacked_data).any(axis=1)] # remove nans across cols to get equal amount
cleaned_data =  cleaned_data[np.isfinite(cleaned_data).any(axis =1)]
 
# correlation between per pupil spending and number of acceptances
r_spending_admissions = stats.spearmanr(cleaned_data[:,0],cleaned_data[:,1])

print(f"Correlation between availability of material resources and number of admissions is: {r_spending_admissions [0]:.4f}")

X = np.transpose([cleaned_data[:,0]]) # Spending
Y = cleaned_data[:,1] # applications

# Scatter plot to show the relationship
plt.style.use('ggplot')
plt.scatter(cleaned_data[:,0],cleaned_data[:,1],marker = 'o')
plt.xlabel('Per Pupil Spending')
plt.ylabel('Number of acceptances')
plt.title(f"Correlation = {r_spending_admissions[0]:.3f}")
plt.show()

# another way
stacked_data = np.stack((df['avg_class_size'],df['acceptances']),axis = 1) # stack the required columns
cleaned_data =  stacked_data[~np.isnan(stacked_data).any(axis=1)] # remove nans across cols to get equal amount
cleaned_data =  cleaned_data[np.isfinite(cleaned_data).any(axis =1)] 

# correlation between average class size and admissions to HSPHS?
r_classSize_admissions = stats.spearmanr(cleaned_data[:,0],cleaned_data[:,1])

print(f"Correlation between Average Class Size and number of admissions is: {r_spending_admissions [0]:.4f}")

X = np.transpose([cleaned_data[:,0]]) # Average Class Size
Y = cleaned_data[:,1] # applications

# Scatter plot to show the relationship
plt.style.use('ggplot')
plt.scatter(cleaned_data[:,0],cleaned_data[:,1],marker = 'o',color = 'purple')
plt.xlabel('Average Class Size')
plt.ylabel('Number of acceptances')
plt.title(f"Correlation = {r_classSize_admissions[0]:.3f}")
plt.show()

temp = df[['acceptances','per_pupil_spending','avg_class_size','student_achievement','reading_scores_exceed','math_scores_exceed']]
temp = temp[~np.isnan(temp).any(axis=1)] # drop nans row wise
test = temp.corr()

#%% 

# extract a temporary data frame for the target variables
temp = df[['per_pupil_spending','avg_class_size','student_achievement','reading_scores_exceed','math_scores_exceed']]
temp = temp[~np.isnan(temp).any(axis=1)] # drop nans row wise

targets = temp[['student_achievement','reading_scores_exceed','math_scores_exceed']]

zscoredData = stats.zscore(targets)
pca = PCA()
pca.fit(zscoredData)
eigValues = pca.explained_variance_ 
covarExplained = eigValues/sum(eigValues)*100
loadings = pca.components_
origDataNewCoordinates = pca.fit_transform(zscoredData)

# Scree Plot 
numPredictors = targets.shape[1]
plt.style.use('ggplot')
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.plot([0,numPredictors],[1,1],color='black',linewidth=1) # Kaiser criterion line
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show() # one PC greater than 1

# Plotting values from the loadings matrix
# factor 1
plt.figure(figsize=(10,5))
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:],
        tick_label =['student_achievement','reading_scores_exceed','math_scores_exceed']
        )
plt.title('Factor 1')
plt.xticks(fontsize=10, rotation=60)
plt.ylabel('Loadings')
plt.show() # math and reading scores exceeded

stacked_data = np.stack((temp['per_pupil_spending'],origDataNewCoordinates[:,0]),axis = 1) # stack the required columns
 
# correlation between per pupil spending and achievement indicators
r_spending_achievement = stats.spearmanr(stacked_data[:,0],stacked_data[:,1])
print(f"Correlation between per_pupil_spending and achievement indicators is: {r_spending_achievement [0]:.4f}")

# Scatter plot to show the relationship
plt.style.use('ggplot')
plt.scatter(stacked_data[:,0],stacked_data[:,1],marker = 'o')
plt.xlabel('Per Pupil Spending')
plt.ylabel('Exceed Reading & Math Scores')
plt.title(f"Correlation = {r_spending_achievement[0]:.3f}")
plt.show()

# another way
stacked_data = np.stack((temp['avg_class_size'],origDataNewCoordinates[:,0]),axis = 1) # stack the required columns

# correlation between average class size and admissions to HSPHS?
r_classSize_achievement = stats.spearmanr(stacked_data[:,0],stacked_data[:,1])

print(f"Correlation between Average Class Size and number of admissions is: {r_classSize_achievement [0]:.4f}")

# Scatter plot to show the relationship
plt.style.use('ggplot')
plt.scatter(stacked_data[:,0],stacked_data[:,1],marker = 'o',color = 'purple')
plt.xlabel('Average Class Size')
plt.ylabel('Exceed Reading & Math Scores')
plt.title(f"Correlation = {r_classSize_achievement[0]:.3f}")
plt.show()

#%%     Question 7

temp = pd.read_csv ('middleSchoolData.csv')
temp = temp.sort_values("acceptances",ascending=False,ignore_index = True)
temp['prop_accepted'] = (temp['acceptances'] / sum(temp['acceptances']))* 100

temp = temp[['school_name','acceptances','prop_accepted']] # only work the variables we need

prop_accepted = 0
test = temp.copy()
test[:] = 0
for i in range(temp.shape[0]):
    if prop_accepted <= 90:
        test.school_name.loc[i] = temp.school_name.loc[i]
        test.acceptances.loc[i] = temp.acceptances.loc[i]
        test.prop_accepted.loc[i] = temp.prop_accepted.loc[i]
        prop_accepted += temp.prop_accepted.loc[i]
        # print(prop_accepted)
    else:
        break
test = test.loc[~(test==0).all(axis=1)] # remove all zeros

proportionOfSchools = (len(test['school_name']) / len(temp['school_name'])) * 100
print(proportionOfSchools)
numOfSchools = len(test['school_name'])
print(numOfSchools)

plt.figure(figsize=(13,7))
plt.bar(test.school_name.loc[:5], test.acceptances.loc[:5],width=0.5)
plt.xticks(fontsize=10, rotation=60)
for i in range(len(temp.acceptances.loc[:5])):
    plt.text(i,temp.acceptances.loc[i],temp.acceptances.loc[i],ha="center",va="bottom")
plt.show()


#%%     Question 8

temp = df.copy()
# remove nans row-wise
temp =  temp[~np.isnan(temp).any(axis=1)] 

# Big multicollinearity problem --> can't do a multiple linear regression without dropping some features or doing a PCA

# Let's do a PCA
# Building a prediction model

# store the dependent varaible
acceptances = temp.acceptances

# store the one of the predictors
applications = temp.applications

temp.drop(['acceptances'],axis = 1,inplace = True)
temp.drop(['applications_rate'],axis = 1,inplace = True)
temp.drop(['rich_school'],axis = 1,inplace = True)
temp.drop(['applications','asian_percent','black_percent', 'hispanic_percent', 'multiple_percent',
           'white_percent','disability_percent','poverty_percent', 'ESL_percent', 'school_size'],axis = 1,inplace = True)

# store all the predictors
predictors = temp.to_numpy()

r = np.corrcoef(predictors,rowvar=False)
plt.style.use('ggplot')
plt.imshow(r) 
plt.colorbar()
plt.show()

corrMatrix = temp.corr()

zscoredData = stats.zscore(predictors)
pca = PCA()
pca.fit(zscoredData)
eigValues = pca.explained_variance_ 
loadings = pca.components_
covarExplained = eigValues/sum(eigValues)*100
origDataNewCoordinates = pca.fit_transform(zscoredData)

numPredictors = predictors.shape[1]
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.plot([0,numPredictors],[1,1],color='black',linewidth=1) # Kaiser criterion line
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show() # Two eigenvaleus greater than 1

# Look at loadings of the first factor
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:],
        tick_label = ['per_pupil_spending', 'avg_class_size', 'rigorous_instruction',
       'collaborative_teachers', 'supportive_environment',
       'effective_school_leadership', 'strong_family_community_ties', 'trust',
       'student_achievement', 'reading_scores_exceed', 'math_scores_exceed']
        )
plt.title('Factor 1')
plt.xticks(fontsize=10, rotation=90)
plt.show() # Rigourous instruction, Collaborative Teachers, Supportive Environment

# Look at loadigns of the second factor
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:],
        tick_label = ['per_pupil_spending', 'avg_class_size', 'rigorous_instruction',
       'collaborative_teachers', 'supportive_environment',
       'effective_school_leadership', 'strong_family_community_ties', 'trust',
       'student_achievement', 'reading_scores_exceed', 'math_scores_exceed'])
plt.title('Factor 2')
plt.xticks(fontsize=10, rotation=90)
plt.show() # Availability of resources, trust, and exceeding math and reading scores

#%% 
X = np.array([applications,origDataNewCoordinates[:,0],origDataNewCoordinates[:,1]]).T
# X = pd.DataFrame(data = X, columns = ['PC1','PC2'])

X2 = sm.add_constant(X)
est = sm.OLS(acceptances, X2)
est2 = est.fit(cov_type='HC1')
print(est2.summary()) 

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

f = open('myreg.tex', 'w')
f.write(beginningtex)
f.write(est2.summary().as_latex())
f.write(endtex)
f.close()

#%% 
#   Question 8.b) Achieving high scores on objective measures of achievement?

# Building our clustering and classification model
temp = df.copy()
temp.drop(['applications_rate'],axis = 1,inplace = True)
temp.drop(['rich_school'],axis = 1,inplace = True)

# remove nans row-wise
temp =  temp[~np.isnan(temp).any(axis=1)] 

# median scores
medianStudentAchievement = np.nanmedian(temp['student_achievement'])
medianMath = np.nanmedian(temp['math_scores_exceed'])
medianReading = np.nanmedian(temp['reading_scores_exceed'])

# setting up our outcome variable
temp.loc[(temp['student_achievement'] > medianStudentAchievement) | (temp['math_scores_exceed'] > medianMath) &
       (temp['reading_scores_exceed'] > medianReading),
       'high_performer'] = 1 # high performer

temp.loc[(temp['student_achievement'] < medianStudentAchievement) | ((temp['math_scores_exceed'] < medianMath)) &
       (temp['reading_scores_exceed'] < medianReading),
       'high_performer'] = 0 # low performer

# store outcomes in an array
outcomes = temp.high_performer.to_numpy()

# remove unwanted variables
temp.drop(['applications','asian_percent', 'black_percent', 'hispanic_percent',
          'multiple_percent', 'white_percent','disability_percent', 
          'poverty_percent', 'ESL_percent','student_achievement',
          'reading_scores_exceed', 'math_scores_exceed', 'high_performer'],axis = 1,inplace = True)

# store predictors in an array
predictors = temp.to_numpy()

# check correlation heatmap if PCA is needed
r = np.corrcoef(predictors,rowvar=False)
plt.imshow(r) 
plt.colorbar()
plt.show()

# Start the PCA process on the predictors
zscoredData = stats.zscore(predictors)
pca = PCA()
pca.fit(zscoredData)
eigValues = pca.explained_variance_ 
loadings = pca.components_
origDataNewCoordinates = pca.fit_transform(zscoredData)

# Inspect the Screeplot
numPredictors = predictors.shape[1]
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues)
plt.plot([0,numPredictors],[1,1],color='black',linewidth=1) # Kaiser criterion line
plt.title('Scree plot')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues') # We get two significant PCs
plt.show()

# Look at loadings of the first factor
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[0,:],
        tick_label = ['acceptances', 'per_pupil_spending', 'avg_class_size',
       'rigorous_instruction', 'collaborative_teachers',
       'supportive_environment', 'effective_school_leadership',
       'strong_family_community_ties', 'trust', 'school_size']
        )
plt.title('Factor 1')
plt.xticks(fontsize=10, rotation=90)
plt.show() # Overall School Climate

# Look at loadigns of the second factor
plt.bar(np.linspace(1,numPredictors,numPredictors),loadings[1,:],
        tick_label = ['acceptances', 'per_pupil_spending', 'avg_class_size',
       'rigorous_instruction', 'collaborative_teachers',
       'supportive_environment', 'effective_school_leadership',
       'strong_family_community_ties', 'trust', 'school_size']
        )
plt.title('Factor 2')
plt.xticks(fontsize=10, rotation=90)
plt.show() # school size, per pupil spending, acceptances, average class size --> Availability of material resources

# Old data in the PCA space
plt.plot(origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],'.',markersize=3,color='black')
plt.xlabel('Overall School Climate')
plt.ylabel('Availability of material resources')
plt.show()

#%% Clustering Procedure
X = np.transpose(np.array([origDataNewCoordinates[:,0],origDataNewCoordinates[:,1],origDataNewCoordinates[:,2]]))

numClusters = 4 # how many clusters are we looping over? (from 2 to 4)
Q = np.empty([numClusters,1]) # init container to store sums
Q[:] = np.NaN # convert to NaN

# Compute kMeans:
for ii in range(2,numClusters + 1): # Loop through each cluster
    kMeans = KMeans(n_clusters = int(ii)).fit(X)
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(X,cId) # compute the mean silhouette coefficient of all samples
    Q[ii-2] = sum(s) # take sum

plt.plot(np.linspace(2,numClusters,numClusters),Q)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores') # 2 clusters has the highest silhouette score

#%% Redo Clustering with correct number of clusters according the silhouette algorithm

numClusters = 2 # correct clusters
Q = np.empty([numClusters,1]) 
Q[:] = np.NaN 

# Compute kMeans:
for ii in range(2,numClusters + 1): 
    kMeans = KMeans(n_clusters = int(ii)).fit(X)
    cId = kMeans.labels_
    cCoords = kMeans.cluster_centers_ 

# plot and color code the data
indexVector = np.linspace(1,len(np.unique(cId)),len(np.unique(cId))) 
for ii in indexVector:
    plotIndex = np.argwhere(cId == int(ii-1))
    plt.plot(origDataNewCoordinates[plotIndex,0],origDataNewCoordinates[plotIndex,1],'o',markersize=3.5)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=6,color='black')  
    plt.xlabel('Overall School Climate')
    plt.ylabel('Availability of material resources')
    plt.title('Clustering')

#%% 
# Classification:
svmModel = svm.SVC()
svmModel.fit(X,outcomes)
sV = svmModel.support_vectors_ # Retrieve the support vectors from the model

plt.plot(X[np.argwhere(outcomes==0),0],X[np.argwhere(outcomes==0),1],'o',markersize=3,color='green', label="Low Performer") # low performing school
plt.plot(X[np.argwhere(outcomes==1),0],X[np.argwhere(outcomes==1),1],'o',markersize=3,color='blue', label="High Performer") # high performing school
plt.plot(sV[:,0],sV[:,1],'o',markersize=3,label="Support Vectors", color='red')
plt.legend() 
plt.xlabel('Overall School Climate')
plt.ylabel('Availability of material resources')
plt.title('Classification using SVM')
plt.show()

decision = svmModel.predict(X) # Decision reflects who the model thinks will be depressed

# Step 4: Assess model accuracy by comparing predictions with reality
comp = np.transpose(np.array([decision,outcomes])) 
modelAccuracy = sum(comp[:,0] == comp[:,1])/len(comp)
print(modelAccuracy)

## -----------------------------End of Code -----------------------------------------------