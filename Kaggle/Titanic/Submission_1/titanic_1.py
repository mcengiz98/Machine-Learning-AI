# https://www.kaggle.com/competitions/titanic/overview # Data Source

##########
# Import #
##########

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV # I will use this because i saw it on Kaggle

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

predictors = ['Pclass', 'IsMale', 'Age']

X_train_f = train_with_fake_data[predictors].values
X_test_f = test_with_fake_data[predictors].values
y_train_f = train_with_fake_data['Survived'].values


#################
# Create Models #
#################

model = LogisticRegressionCV(10) # Create model

model.fit(X_train_f, y_train_f) # Fit model

y_predict_f = model.predict(X_test_f) # Predict

output_f = pd.DataFrame({'PassengerId': test_with_fake_data.PassengerId, 'Survived': y_predict_f})
output_f.to_csv('submission.csv', index=False)


###########
# Spoiler #
###########

# Score: 0.75837

###########
# Spoiler #
###########