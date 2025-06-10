# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 18:28:55 2025
@author: Simon
"""
#%% The cell to import modules, load datasets, and set the seed for RNG.
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler#to normalize data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score
from scipy.special import expit
from sklearn.decomposition import PCA
seed = 13517345#my N number


dataNum = np.genfromtxt('rmpCapstoneNum.csv',delimiter=',') #load the 'rmpCapstoneNum.csv' file
dataQual = pd.read_csv('rmpCapstoneQual.csv', header = None).to_numpy()
RNG = np.random.default_rng(seed)
alpha = 0.005#we have sufficient power (great effect size and sample size), so low alpha ok.
dataNum_colNames = np.array(["average rating", "average difficulty", "number of ratings", "pepper", "again proportion", "number of ratings online", "male", "female"])
dataQual_colNames = np.array(["major", "university", "state"])
#%% Q1:
#EDA (three graphs)
#1. See the underlying distribution of number of ratings
mask_numRate = np.all(~np.isnan(dataNum[:, [2]]), axis=1)
dataNum_numRate = dataNum[mask_numRate]

plt.hist(dataNum_numRate[:, 2], bins=200, alpha = 0.5)
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Ratings')
plt.xlim(1, 60)
plt.xticks(np.arange(1, 61, 5))
plt.axvline(x=7, color='red', linestyle='--', label='Threshold = 7')
plt.legend()
plt.show()

thresh = 7

#Pre-processing: Dropping rows with number of ratings less than the threshold (7)
dataNum_thresh = dataNum[dataNum[:, 2] >= thresh]

#%%
#2. See the underlying distribution of average ratings
plt.clf()
plt.hist(dataNum_thresh[:, 0], bins=30, edgecolor='black')
plt.xlabel('Average Ratings')
plt.ylabel('Frequency')
plt.title('Distribution of Average Ratings, considering threshold')
plt.xticks([1, 2, 3, 4, 5])
plt.show()#The distribution is not normal.
#%%
#3. Plot the average rating of male and nonmale.
mask_avg_male = np.all(~np.isnan(dataNum_thresh[:, [0, 6]]), axis=1)
dataNum_avg_male = dataNum_thresh[mask_avg_male]

male = dataNum_avg_male[dataNum_avg_male[:, 6] == 1][:, 0]
nonmale = dataNum_avg_male[dataNum_avg_male[:, 6] != 1][:, 0]

plt.clf()
plt.hist(male, alpha = 0.5, color = "blue", label="Male's avg")
plt.hist(nonmale, alpha = 0.5, color = "red", label="Nonmale's avg")
plt.xlabel('Average Ratings')
plt.ylabel('Frequency')
plt.title('Distribution of Average Ratings of Male and Non-male')
plt.legend()
plt.show()


#Mann-Whitney U test to check if gender -> ratings. 
u_avg_male, pu_avg_male = stats.mannwhitneyu(male, nonmale, alternative = 'greater')
print("Q1:", u_avg_male, pu_avg_male)
#u is so big and p-value is so small -> drop null hypo, so ratings are gender-biased.
#%% Q2:
#EDA
mask_avg_num = np.all(~np.isnan(dataNum_thresh[:, [0, 2]]), axis=1)
dataNum_avg_num = dataNum_thresh[mask_avg_num]

#split number of ratings to two labels, low and high.
#1. See the underlying distributions of average ratings with low and high number of ratings
dataNum_avg_num_low = dataNum_avg_num[dataNum_avg_num[:, 2] < np.median(dataNum_avg_num[:, 2])]
dataNum_avg_num_high = dataNum_avg_num[dataNum_avg_num[:, 2] >= np.median(dataNum_avg_num[:, 2])]
print("The median of number of ratings is:", np.median(dataNum_avg_num[:, 2]))
plt.clf()
plt.hist(dataNum_avg_num_low[:, 0], bins=30, alpha=0.5, color='blue', label="Low Num's avg")
plt.hist(dataNum_avg_num_high[:, 0], bins=30, alpha=0.5, color='red', label="High Num's avg")
plt.title('Distributions of average ratings with low and high number of ratings')
plt.xlabel('Average Ratings')
plt.ylabel('Frequency')
plt.legend()

plt.show()
#%%
#Permutation test to check if experience -> quality of teaching
def ourTestStatistic(x,y):
    return np.median(x) - np.median(y)
dataToUse = (dataNum_avg_num_high[:, 0], dataNum_avg_num_low[:, 0])
pTest = stats.permutation_test(dataToUse, ourTestStatistic, n_resamples=2e4, alternative='two-sided', random_state = seed) 
print('Test statistic:', pTest.statistic)#~0.10
print('exact p-value:',pTest.pvalue)#~0.50, but I'm not sure about my test statistics

#Draw null distribution. (not normal)
plt.hist(pTest.null_distribution, alpha=1, color='black', label="Null Distribution")
plt.show()
#%% Q3:
#Scatterplot of Average Rating vs. Average Difficulty
mask_avg_diff = np.all(~np.isnan(dataNum_thresh[:, [0, 1]]), axis=1)
dataNum_avg_diff = dataNum_thresh[mask_avg_diff]

plt.clf()
plt.scatter(dataNum_avg_diff[:, 1], dataNum_avg_diff[:, 0], c = "blue", alpha = 0.2, s = 15)
plt.title('Scatterplot of Average Rating vs. Average Difficulty')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')

plt.show()
#%% Spearman's rho
spearcorr_avg_diff = stats.spearmanr(dataNum_avg_diff[:, 1], dataNum_avg_diff[:, 0])
print("The Spearmans' correlation between Avg rating and Avg difficulty:", spearcorr_avg_diff.statistic)
#%% Q4:
mask_online = np.all(~np.isnan(dataNum_thresh[:, [0, 2, 5]]), axis=1)
dataNum_online = dataNum_thresh[mask_online]
prop_online = dataNum_online[:, 5] / dataNum_online[:, 2]

plt.clf()
plt.hist(prop_online, bins=30, alpha=0.5, color='blue')
plt.title('Distributions of Proportion of Number of Ratings from Online Classes')
plt.xlabel('Proportion of Online')
plt.ylabel('Frequency')
plt.show()

#%% SPlit and label.
if_online_alot = prop_online > np.median(prop_online)
online_alot = dataNum_online[if_online_alot]
online_little = dataNum_online[if_online_alot == False]

plt.clf()
plt.hist(online_alot[:, 0], bins=30, alpha=0.5, color='blue', label="Avg Rating of A lot of Online")
plt.hist(online_little[:, 0], bins=30, alpha=0.5, color='red', label="Avg Rating of Little Online")
plt.title('Distributions of average ratings with low and high proportion of online ratings')
plt.xlabel('Average Ratings')
plt.ylabel('Frequency')
plt.legend()

plt.show()
#%% Permutation test: online affects average rating?
def weighted_median_diff(x, y):
    w_x = len(x) / (len(x) + len(y))
    w_y = len(y) / (len(x) + len(y))
    return w_x * np.median(x) - w_y * np.median(y)
dataToUse_online = (online_alot[:, 0], online_little[:, 0])
pTest_online = stats.permutation_test(dataToUse_online, weighted_median_diff, n_resamples=2e4, alternative='two-sided', random_state=seed) 
print('Test statistic:', pTest_online.statistic)
print('Exact p-value:', pTest_online.pvalue)

#Draw null distribution. (not normal)
plt.hist(pTest_online.null_distribution, alpha=1, color='black', label="Null Distribution")
plt.show()
#%% Q5:
#Scatterplot of Average Rating vs. Proportion of Taking Again
mask_avg_again = np.all(~np.isnan(dataNum_thresh[:, [0, 4]]), axis=1)
dataNum_avg_again = dataNum_thresh[mask_avg_again]

plt.scatter(dataNum_avg_again[:, 4], dataNum_avg_again[:, 0], c = "blue", alpha = 0.2, s = 15)
plt.title('Scatterplot of Average Rating vs. Proportion of Taking Again')
plt.xlabel('Taking Again')
plt.ylabel('Average Rating')

plt.show()

#%% Spearman's rho
spearcorr_avg_diff = stats.spearmanr(dataNum_avg_again[:, 4], dataNum_avg_again[:, 0])
print("The Spearmans' correlation between Avg and Proportion of Taking Again:", round(spearcorr_avg_diff.statistic, 2))

#%% Q6:
mask_avg_hot = np.all(~np.isnan(dataNum_thresh[:, [0, 3]]), axis=1)
dataNum_avg_hot = dataNum_thresh[mask_avg_hot]

dataNum_hot = dataNum_avg_hot[dataNum_avg_hot[:, 3] == 1]
dataNum_nothot = dataNum_avg_hot[dataNum_avg_hot[:, 3] == 0]

plt.clf()
plt.hist(dataNum_hot[:, 0], bins=30, alpha=0.5, color='blue', label="Hottie's avg rating")
plt.hist(dataNum_nothot[:, 0], bins=30, alpha=0.5, color='red', label="Meh's avg rating")
plt.title("""Distributions of average ratings of professors received "pepper" and those who didn't""")
plt.xlabel('Average Ratings')
plt.ylabel('Frequency')
plt.legend()
plt.show()
#%% Permutation test to see if hotties get higher ratings.
dataToUse_hot = (dataNum_hot[:, 0], dataNum_nothot[:, 0])
#outTestStatistic was defined befored.
pTest_hot = stats.permutation_test(dataToUse_hot, ourTestStatistic, n_resamples=1e4, alternative='greater', random_state=seed) 
print('Test statistic:', pTest_hot.statistic)
print('Exact p-value:',pTest_hot.pvalue)

#Draw null distribution.
plt.hist(pTest_hot.null_distribution, alpha=1, color='black', label="Null Distribution, Average Rating of Hot/notHot")
plt.show()

#%% Q7: Regression: predict Average rating from Average difficulty. R^2 and RMSE?
x_avgDiff = dataNum_avg_diff[:, 1].reshape(len(dataNum_avg_diff),1)
y_avgRate = dataNum_avg_diff[:, 0]

#the question asks build a model to *predict*, so I think I need to predict.
#split to train and test sets.

train_xDiff, test_xDiff, train_yRate, test_yRate = train_test_split(x_avgDiff, y_avgRate, test_size = 0.2, random_state = seed)
diff_rating_LRModel = LinearRegression().fit(train_xDiff, train_yRate)

diff_rating_predicted = diff_rating_LRModel.predict(test_xDiff)

print("Coeff:", diff_rating_LRModel.coef_)
print("Intercept:", diff_rating_LRModel.intercept_)

print("R^2:", r2_score(test_yRate, diff_rating_predicted))
print("RMSE", (mean_squared_error(test_yRate, diff_rating_predicted))**(1/2))

#%% Plot the Distribution of Residuals of Avg Rating vs Avg difficulty Linear Regression Model
residuals_diff_Rate = test_yRate - diff_rating_predicted
plt.clf()
plt.hist(residuals_diff_Rate, alpha=1, color='blue')
plt.xlabel("Residuals of Average Rating")
plt.ylabel("Frequency")
plt.title("The Distribution of Residuals of Avg Rating vs Avg difficulty Linear Regression Model")
plt.show()
#%% Q8: Regression model to predict average rating from all factors.
#leering correlation matrix
corrMatrix = np.corrcoef(dataNum_avg_again,rowvar=False)

plt.imshow(corrMatrix) 
plt.xlabel('Question')
plt.ylabel('Question')
plt.colorbar()
plt.show()
#%% Cross validation first to pick lambda
x_allFact = dataNum_avg_again[:, 1:]
y_avgRate = dataNum_avg_again[:, 0]
xTrain_allFact, xTest_allFact, yTrain_avgRate, yTest_avgRate = train_test_split(x_allFact, y_avgRate, test_size=0.5, random_state=seed)

def crossVali_lambdas(xTrain, xTest, yTrain, yTest):
    lambdas = [0.01, 0.1, 1, 5, 10, 50, 100]
    rmse_list = []
    scaler = StandardScaler()
    xTrain_scaled = scaler.fit_transform(xTrain)
    xTest_scaled = scaler.transform(xTest)
    for l in lambdas:
        model = Ridge(alpha=l)
        model.fit(xTrain_scaled, yTrain)
        yPred = model.predict(xTest_scaled)
        rmse = mean_squared_error(yTest, yPred, squared=False)
        rmse_list.append(rmse)
    return rmse_list, lambdas[np.argmin(rmse_list)]

#ridge model
rmse_list_ridge, best_lambda_ridge = crossVali_lambdas(xTrain_allFact, xTest_allFact, yTrain_avgRate, yTest_avgRate)
print(rmse_list_ridge)
print("Best lambda (based on test RMSE) in Ridge:", best_lambda_ridge)

#%% Multiple Linear Regression using Ridge regularization, predicting average rating from all factors
xTrain_allFact, xTest_allFact, yTrain_avgRate, yTest_avgRate = train_test_split(x_allFact, y_avgRate, test_size=0.2, random_state=seed)
scale = StandardScaler()
xTrain_allFact_scaled = scale.fit_transform(xTrain_allFact)
xTest_allFact_scaled = scale.fit_transform(xTest_allFact)

ridge_avg_all = Ridge(alpha = best_lambda_ridge)
#this *alpha* here is lambda in regularization, not the alpha in statistical testing
ridge_avg_all.fit(xTrain_allFact_scaled, yTrain_avgRate) #Fit the model
yPred_avgRate = ridge_avg_all.predict(xTest_allFact_scaled)

print("R square:", r2_score(yTest_avgRate, yPred_avgRate))
print("RMSE:", (mean_squared_error(yTest_avgRate, yPred_avgRate))**(1/2))

betas_avg_all = ridge_avg_all.coef_
print("The coefficient for avg Difficulty", betas_avg_all[0])
#%% Q9: Classification predicting whether prof is hot from average rating.
#data_hot = dataNum_thresh[dataNum_thresh[:, 3] == 1]
#data_cold = dataNum_thresh[dataNum_thresh[:, 3] == 0]#cold LOL

plt.clf()
plt.hist(dataNum[:, 3])
plt.title("Hot and not Hot Group Size Comparison, Using unprocessed dataset")
plt.xticks([0, 1])
plt.xlabel('Hot or "Cold"')
plt.ylabel("Frequency")
plt.show()
#%%
plt.clf()
plt.hist(dataNum_thresh[:, 3])
plt.title("Hot and not Hot Group Size Comparison, Using dataNum_thresh")
plt.xticks([0, 1])
plt.xlabel('Hot or "Cold"')
plt.ylabel("Frequency")
plt.show()

#%% Plot average rating vs hot?
plt.clf()
plt.scatter(dataNum_thresh[:,0], dataNum_thresh[:,3], alpha = 0.05, color='orangered')
plt.xlabel('Average Rating')
plt.xlim([1, 5])
plt.ylabel('Hot?')
plt.yticks(np.array([0,1]))
plt.show()
#%% Actually fit the logistic regression.
xTrain_avgRate, xTest_avgRate, yTrain_hot, yTest_hot = train_test_split(dataNum_thresh[:, 0], dataNum_thresh[:, 3], test_size=0.2, random_state=seed)
xTrain_avgRate = xTrain_avgRate.reshape(len(xTrain_avgRate),1)
xTest_avgRate = xTest_avgRate.reshape(len(xTest_avgRate),1)

# Fit model:
avgHot_LogModel = LogisticRegression().fit(xTrain_avgRate, yTrain_hot)

#%% Plot the logistic model
x1 = np.linspace(1, 5)
y1 = x1 * avgHot_LogModel.coef_ + avgHot_LogModel.intercept_
sigmoid = expit(y1)

# Plot:
plt.clf()
plt.plot(x1,sigmoid.ravel(),color='black',linewidth=3)
plt.scatter(dataNum_thresh[:,0], dataNum_thresh[:,3], alpha = 0.05, color='orangered')

plt.xlabel('Average Rating')
plt.xlim([1, 5])
plt.hlines(0.5, 1, 5 ,colors='gray',linestyles='dotted')
plt.ylabel('Probability of being Hot')
plt.yticks(np.array([0,1]))
plt.show()
#%% Predict hot or not. Plot ROC and Calculate AUROC
yProb_hot = avgHot_LogModel.predict_proba(xTest_avgRate)[:, 1]#predicted prob to be hot.

fpr_avgHot, tpr_avgHot, thresholds_avgHot = roc_curve(yTest_hot, yProb_hot)
auroc_avgHot = roc_auc_score(yTest_hot, yProb_hot)
#print("fpr, tpr, and thresholds", fpr_avgHot, tpr_avgHot, thresholds_avgHot)
print("AUROC is:", auroc_avgHot)

plt.figure(figsize=(8, 6))
RocCurveDisplay(fpr = fpr_avgHot, tpr = tpr_avgHot, roc_auc = auroc_avgHot).plot()
plt.scatter(fpr_avgHot, tpr_avgHot)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random (AUROC = 0.5)')
plt.title('ROC Curve with AUROC of ' + str(round(auroc_avgHot, 2)))
plt.legend()
plt.show()
#%% Quality metrices of this model (avg Rating -> hot?)
yBinary_hot = (yProb_hot >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(yTest_hot, yBinary_hot).ravel()

precision = tp/(tp+fp)
precision_test = precision_score(yTest_hot, yBinary_hot)

sensitivity = tp/(tp + fn)
specificity = tn/(tn+fp)
npv = tn/(tn+fn)
accuracy = (tp+tn)/(tn+fp+fn+tp)

print("Precision:", round(precision, 2))
print("Sensitivity:", round(sensitivity, 2))
print("Specificity:", round(specificity, 2))
print("NPV (Negative Predictive Value):", round(npv, 2))
print("General Accuracy:", round(accuracy, 2))
#%% Q10: all factors -> hot?
#PCA first
#z-score the dataset first
dataNum_allfac = dataNum_avg_again[:, [0, 1, 2, 4, 5, 6, 7]]# excluding pepper column
dataNum_hot = dataNum_avg_again[:, 3]

xTrain_allfac, xTest_allfac, yTrain_hot, yTest_hot = train_test_split(dataNum_allfac, dataNum_hot, test_size=0.2, random_state=seed)

zscored_dataNum_allfac = stats.zscore(xTrain_allfac)

zscored_testX_allfac = stats.zscore(xTest_allfac)#this is prepared for logreg prediction.

pca_allfac = PCA().fit(zscored_dataNum_allfac)
loadings_allfac = pca_allfac.components_ 
eigVals_allfac = pca_allfac.explained_variance_

varExplained_allfac = eigVals_allfac/sum(eigVals_allfac)*100
print("Variance Explained in a sequence:")
for ii in range(len(varExplained_allfac)):
    print(varExplained_allfac[ii].round(3))
    
#Make a scree plot
x_scree = np.linspace(1, 7, 7)
plt.bar(x_scree, eigVals_allfac, color='cyan')
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()
    
# 90% eigensum criterion:
# Number of factors that account for 90% of the variance (Eigenvalues that 
# add up to 90% of the Eigensum. To account for at least 90% of the variability 
eigSum = np.cumsum(varExplained_allfac)
n_components = np.count_nonzero(eigSum < 90) + 1
print('Number of factors to account for at least 90% variance:', n_components)
#%%
pca_scores_allfac = pca_allfac.transform(zscored_dataNum_allfac)
pca_testScores_allfac = pca_allfac.transform(zscored_testX_allfac)

x_newTrainAll = pca_scores_allfac[:, :n_components]
x_newTestAll = pca_testScores_allfac[:, :n_components]

allHot_LogModel = LogisticRegression().fit(x_newTrainAll, yTrain_hot)

y_allPredHot = allHot_LogModel.predict_proba(x_newTestAll)[:, 1]

yBi_allHot = (y_allPredHot >= 0.5).astype(int)

# Evaluate the model
print("AUROC:", roc_auc_score(yTest_hot, y_allPredHot))#should use yTest and yProbability, not yBinary
#%%

tn, fp, fn, tp = confusion_matrix(yTest_hot, yBi_allHot).ravel()

precision = tp/(tp+fp)
sensitivity = tp/(tp + fn)
specificity = tn/(tn+fp)
npv = tn/(tn+fn)
accuracy = (tp+tn)/(tn+fp+fn+tp)

print("Precision:", round(precision, 2))
print("Sensitivity:", round(sensitivity, 2))
print("Specificity:", round(specificity, 2))
print("NPV (Negative Predictive Value):", round(npv, 2))
print("General Accuracy:", round(accuracy, 2))

#%% Extra Credit. West Coast hot or East Coast hot?
west_coast = ["CA", "OR", "WA", "AK", "HI"]
east_coast = ["ME", "NH", "MA", "RI", "CT", "NY", "NJ", "PA", "DE", "MD", "VA", "NC", "SC", "GA", "FL"]

west_mask = np.isin(dataQual[:, 2], west_coast)
east_mask = np.isin(dataQual[:, 2], east_coast)

west = dataNum[west_mask]
east = dataNum[east_mask]
west_maskHot = np.all(~np.isnan(west[:, [3]]), axis=1)
east_maskHot = np.all(~np.isnan(east[:, [3]]), axis=1)

west = west[west_maskHot]
east = east[east_maskHot]

west_hot = west[west[:, 3] == 1]
west_notHot = west[west[:, 3] == 0]
east_hot = east[east[:, 3] == 1]
east_notHot = east[east[:, 3] == 0]


print(len(east_notHot))

plt.clf()
plt.hist(west[:, 3], bins=30, alpha=0.5, color='blue', label="West Hot")
plt.hist(east[:, 3], bins=30, alpha=0.5, color='red', label="East Hot")
plt.title("""Distributions of Hot Professors in the West Coast and the East Coast""")
plt.xlabel('Hot')
plt.ylabel('Frequency')
plt.legend()
plt.xticks([0, 1])
plt.show()
#%%
table = np.array([[len(west_hot), len(west_notHot)], [len(east_hot), len(east_notHot)]])
print(len(west_hot), len(west_notHot), len(east_hot), len(east_notHot))

chi2, p_chi, dof, expected = stats.chi2_contingency(table)
print("Chi-square p-value:", p_chi)

print(expected)

p_west = len(west_hot) / (len(west))
p_east = len(east_hot) / (len(east))

print(f"West Coast hot %: {p_west*100:.1f}%")
print(f"East Coast hot %: {p_east*100:.1f}%")
print(f"Difference: {p_west - p_east:.3f} (+{((p_west/p_east)-1)*100:.1f}% higher)")

