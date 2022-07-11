# https://www.kaggle.com/competitions/titanic/overview # Data Source

##########
# Import #
##########

import pandas as pd
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent

# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html  # How to select best model for your data

########################
# Train and Test Files #
########################

train = pd.read_csv('../Titanic/Data/train.csv')  # 891 total data
test = pd.read_csv('../Titanic/Data/test.csv') # 418 total data

####################
# Data Preparation #
####################

# Check null values in train and test
    # scikit algorithms generally cannot be powered by missing data, 
    # so I’ll be looking at the columns to see if there are any that contain missing data
    
train.isnull().sum() 
    # Age: 177
    # Cabin: 687
    # Embarked: 2
test.isnull().sum()
    # Age: 86
    # Cabin: 327
    # Fare: 1

#* I'll be keep original dataframes if i need them.

train_with_fake_data = train # Copy train dataframe
test_with_fake_data = test # Copy test dataframe

fake_data = train['Age'].mean() # Create fake data for missing values in train

train_with_fake_data['Age'] = train_with_fake_data['Age'].fillna(fake_data) # Fill null values with fake data

fake_data = test['Age'].mean() # Create fake data for missing values in train

test_with_fake_data['Age'] = test_with_fake_data['Age'].fillna(fake_data) # Fill null values with fake data

del fake_data # Delete no longer used data

#* I'll be creating a new column for the "Sex" column.
#* If "Sex" is "male" then "1"
#* Else "0"

train_with_fake_data['IsMale'] = (train_with_fake_data['Sex'] == 'male').astype(int) 
test_with_fake_data['IsMale'] = (test_with_fake_data['Sex'] == 'male').astype(int)


#* I'll be creating a new column for the "SibSp" column.
#* range of SibSp ->  0 < SibSp < 8
#* If SibSp < 4 then SS = 0
#* Else SS = 1

train_with_fake_data['SS'] = 0
test_with_fake_data['SS'] = 0

for index, row in train_with_fake_data.iterrows():
    train_with_fake_data['SS'][index] = 0 if row['SibSp'] < 4 else 1

for index, row in test_with_fake_data.iterrows():
    test_with_fake_data['SS'][index] = 0 if row['SibSp'] < 4 else 1

predictors = ['Pclass', 'IsMale', 'Age', 'SS']

X_train_f = train_with_fake_data[predictors].values
X_test_f = test_with_fake_data[predictors].values
y_train_f = train_with_fake_data['Survived'].values


#################
# Create Models #
#################

model = SGDClassifier(loss="hinge", penalty="l2") # Create model

model.fit(X_train_f, y_train_f) # Fit model

y_predict_f = model.predict(X_test_f) # Predict

output_f = pd.DataFrame({'PassengerId': test_with_fake_data.PassengerId, 'Survived': y_predict_f})
output_f.to_csv('submission.csv', index=False)

compare_with_local_best = pd.read_csv('Submission_1\submission.csv') # Score = 0.75837

comp = output_f.compare(compare_with_local_best) # 52 different values, 12.44019 % difference

###########
# Spoiler #
###########

# Score: 0.74880
# Improvement vs Last: +1.95523 % 
# Improvement vs Best: -1.26192 % 

###########
# Spoiler #
###########
