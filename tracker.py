# import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

df = pd.read_csv('./dataset/detections-5.csv')

# print(df.shape)
# print(df.head(5))
# print(df.describe())
#* df['x1'] = df['x1'].str.replace(r'\D', '', regex=True) # caught some extra 'n' and 'q' chars from terminal (fix later, what if it was a number instead...)

# print(df.T)
# sns.scatterplot(x=df['x1'], y=df['y1'], hue=df['tracker_id'], palette='Reds')
# plt.show()

# sns.boxplot(x=df['x1'], y=df['y1'])

# sns.lineplot(x=df['x1'], y=df['y1'])
# sns.heatmap(data = df[['x1', 'y1', 'tracker_id']])
# plt.show()
# corr = df.corr(numeric=True)
# sns.heatmap(data=corr, cmap='coolwarm')
# plt.show()
# print(df.groupby('recording_name'))

pca = PCA(n_components=1)
df['xy_pca'] = pca.fit_transform(df[['x1','x2','y1', 'y2']])
# fig, axes = plt.subplots(3,3, figsize=(10,5))

# i = 0
# for name, group in df.groupby('recording_name'):
#     if i > 5: break
#     i+=1
#     print(f'name: {name}')
    # sns.scatterplot(x=group['frame_num'], y=group['xy_pca'], hue=group['tracker_id'], palette='Reds')

    # sns.scatterplot(x=group['x1'], y=group['y1'], hue=group['tracker_id'], palette='Reds')
    # plt.show()


# model = KNeighborsClassifier()
X = df[['x1','x2','y1', 'y2']]
y = df[['tracker_id']]
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# model.fit(X_train, y_train)
# train_score = model.score(X_train,y_train)
# test_score = model.score(X_test,y_test)
# print(f'train: {train_score:.2f} test: {test_score:.2f} report:\n')

estimator = KNeighborsClassifier()
param_grid={
    'n_neighbors':(2,4,5,8,10),
    'weights':('uniform', 'distance'),
    'leaf_size':(1,20,30),
    'p':(1,2),
    'metric':('minkowski','chebyshev')
}
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
}
grid = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    scoring='accuracy',
    n_jobs=-1,
    cv=5
)

# grid.fit(X_train, y_train)
# print('best params',grid.best_params_)
# print('best score ',grid.best_score_)

best_params = {'leaf_size': 1, 'metric': 'minkowski', 'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
model = KNeighborsClassifier(**best_params)
model.fit(X_train,y_train)

train_score = model.score(X_train,y_train)
test_score = model.score(X_test,y_test)
print(f'train: {train_score:.2f} test: {test_score:.2f}')
