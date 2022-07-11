# this will be predict score before submission (i hope so (: )

import pandas as pd
import matplotlib.pyplot as mpl
import pickle
import numpy as np

scoreboard = pd.read_csv('scoreboard.csv')

result = pd.read_pickle('result.pkl')

new_sub = pd.read_csv('submission.csv')

referance_df = scoreboard
referance_df['Sub'] = new_sub['Survived']

referance_df['Difference']=""

for index, row in referance_df.iterrows():
    if row['Survived'] == row['Sub']:
        referance_df['Difference'][index] = True
    else:
        referance_df['Difference'][index] = False

s = referance_df['Difference'].value_counts()

score = s[True] /(s[True] + s[False])

last = int(result.iloc[-1]['submission'])

imp = ((score / np.float64(result.iloc[-1]['score'])) * 100) -100

dict = {'submission': (last+1), 'score': score, 'improvement': imp}

result = result.append(dict, ignore_index = True)

result

result.to_pickle('result.pkl')

mpl.plot(result['submission'], result['score']*100)
mpl.show()