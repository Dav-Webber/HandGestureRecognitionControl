import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('CSV/train3.csv')
g0 = df.loc[df['class'] == 0]
g1 = df.loc[df['class'] == 1]


g0 = g0.drop(columns=['class'])
g1 = g1.drop(columns=['class'])

# Create plot
fig = plt.figure()
ax1 = fig.add_subplot(111)

for i in range(round(g0.shape[0]/15.0)):
    for j in range(0,21,2):
        marker = 'o'
        if j==0:
            marker = 'x'
        ax1.scatter(x=g0.iloc[i][j],y=g0.iloc[i][j+1], c='red', marker=marker)
for i in range(round(g1.shape[0]/15)):
    marker = 'o'
    if j == 0:
        marker = 'x'
    for j in range(0,21,2):
        ax1.scatter(x=g1.iloc[i][j],y=g1.iloc[i][j+1], c='green', marker=marker)

plt.show()


