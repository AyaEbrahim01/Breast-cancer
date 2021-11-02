import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#This contains tools for Ml and statistical modeling including classification, regression, clustering.
!pip install sklearn
import sklearn
#Converting raw data to clean dataset
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl_lm
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
sns.set(style="whitegrid", color_codes=True, font_scale=1.3)
%matplotlib inline
import io
from google.colab import files
uploaded = files.upload()
data_df = pd.read_csv(io.BytesIO(uploaded['data.csv']))
data_df.head()
data_df.info()
It looks like no data is  missing except the "Unnamed"
data_df= data_df.drop('Unnamed: 32' , axis=1)
data_df.info()
#Checking data type of each column
data_df.dtypes 
#We found that "diagnonsis" is categorial which has 2 classes => M or B
#Visualizing the disrtibution of classes
plt.figure(figsize=(8,4))
#countplot is like a histogram of a catagorial 
sns.countplot(data_df['diagnosis'],palette='Reds')
benign , malignant = data_df['diagnosis'].value_counts()
print('Number of cells labeled Benign: ',benign)
print('Number of cells labeled Malignant: ', malignant)
print('')
print('% of cells labeled Benign',round(benign / len(data_df) * 100,2),'%' )
print('% of cells labeled Malignant',round(malignant / len(data_df) * 100,2),'%' )
Number of cells labeled Benign:  357
Number of cells labeled Malignant:  212
% of cells labeled Benign 62.74 %
% of cells labeled Malignant 37.26 %
features = ['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']
sns.pairplot(data= data_df[features] ,hue="diagnosis",palette="RdBu")
.corr() find the relation between each and every numeric coulumn
https://www.geeksforgeeks.org/python-pandas-dataframe-corr/

.round() aproximate to the digit given
# Generate and visualize the correlation matrix
corr = data_df.corr().round(2)

#makes a mask "a virtual shape of the data" as a bolean array
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set figure size
f, ax = plt.subplots(figsize=(20, 20))

# Define custom colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap https://seaborn.pydata.org/generated/seaborn.heatmap.html
#V is value
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

#Adjust the padding between and around subplots.
plt.tight_layout()
# drop all "worst" columns
cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']
data_df = data_df.drop(cols, axis=1)

# drop all columns related to the "perimeter" and "area" attributes
# since we use radius as a measurement unit 
cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
data_df = data_df.drop(cols, axis=1)

# drop all columns related to the "concavity" and "concave points" attributes as we will use compaction instead

cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
data_df = data_df.drop(cols, axis=1)
data_df.columns
# Draw the heatmap again

corr = data_df.corr().round(2) # 0.09
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.tight_layout()
#splitting the data

X = data_df
y = data_df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
#define the formula to be used in the model

cols = data_df.columns.drop('diagnosis')
formula = 'diagnosis ~ ' + ' + '.join(cols)
print(formula, '\n')
#training the model and printing a summary

model = smf.glm(formula=formula, data=X_train, family=sm.families.Binomial())

print(model.fit().summary())
predictions = model.fit().predict(X_test)
predictions[1:6]
# Convert these probabilities into nominal values and check the first 5 predictions again.
predictions_nominal = [ "M" if x < 0.5 else "B" for x in predictions]
predictions_nominal[1:6]
print(classification_report(y_test, predictions_nominal, digits=3))

cfm = confusion_matrix(y_test, predictions_nominal)

true_negative = cfm[0][0]
false_positive = cfm[0][1]
false_negative = cfm[1][0]
true_positive = cfm[1][1]

print('Confusion Matrix: \n', cfm, '\n')

print('True Negative:', true_negative)
print('False Positive:', false_positive)
print('False Negative:', false_negative)
print('True Positive:', true_positive)
print('Correct Predictions', 
      round((true_negative + true_positive) / len(predictions_nominal) * 100, 1), '%')
