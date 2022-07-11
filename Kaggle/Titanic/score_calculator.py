# this will be predict score before submission (i hope so (: )

import pandas as pd

scoreboard = pd.read_csv('scoreboard.csv')

# sub1 = pd.read_csv('Submission_1\submission.csv')
# sub2 = pd.read_csv('Submission_2\submission.csv')

# sub1.info()


# scoreboard = sub1

# scoreboard['Submission2'] = sub2['Survived']

# scoreboard.rename(columns = {'Survived':'Best_Sol'}, inplace = True)
# scoreboard['Prediction']=""

# for index, row in scoreboard.iterrows():
#     if row['Submission2'] == row['Best_Sol']:
#         scoreboard['Prediction'][index] = 'True'
#     else:
#         scoreboard['Prediction'][index] = 'False'

# scoreboard.info()

scoreboard.to_csv('scoreboard.csv', index = False)