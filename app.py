import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report

from tensorflow import keras
from imblearn.over_sampling import SMOTE
df = pd.read_csv('FAKE PROFILE.csv')
df.shape
df.info()

print(df.head())

print(df.describe())

print(df['has_channel'].sum())

df.drop('has_channel',axis=1,inplace=True)
print(df.shape)


df['is_fake'].value_counts().plot(kind='pie',autopct="%.2f")
plt.show()

binary_cols = ['username_has_number','full_name_has_number','is_private','is_joined_recently','is_business_account','has_guides','has_external_url']


for b_col in binary_cols:
    plt.figure(figsize=(7,7))
    sns.countplot(data=df[[b_col,'is_fake']],x=b_col,hue='is_fake')
    plt.title(b_col)
    plt.show()
    print('\n')


other_cols = ['edge_followed_by','edge_follow','username_length','full_name_length']


for col in other_cols:
    plt.figure(figsize=(7,7))
    sns.histplot(data=df[[col,'is_fake']],x=col,hue='is_fake',element='poly')
    plt.title(col)
    plt.show()
    print('\n')

plt.figure(figsize=(7,7))
sns.pairplot(df[other_cols+['is_fake']],hue='is_fake')
plt.show()

df.describe()

cols_to_scale = ['username_length','full_name_length']

scale = MinMaxScaler()
scalled = scale.fit_transform(df[cols_to_scale])

df['username_length'],df['full_name_length'] = scalled[:,0],scalled[:,1]

print(df.head())

X,Y = df.drop('is_fake',axis=1),df['is_fake']
X.shape,Y.shape


smote = SMOTE(sampling_strategy='minority')
X,Y = smote.fit_resample(X,Y)
X.shape,Y.shape


xydf = pd.concat([X,Y],axis=1)
xydf.shape


plt.figure(figsize=(7,7))
sns.pairplot(xydf[other_cols+['is_fake']],hue='is_fake')
plt.show()

Y.value_counts().plot(kind='pie',autopct="%.2f")
plt.show()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=1)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
classifier = MLPClassifier(random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(x_test)
#print(classification_report(y_test, y_pred))
filename = 'fake-profile-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
print('Creating a pickle file for the classifier Compeleted')





model = keras.Sequential([
    keras.layers.Dense(11, input_shape=(11,), activation='relu'),
    keras.layers.Dense(2, activation='softmax'),

])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train,epochs=100)

model.evaluate(x_test,y_test)


y_test_predict = [np.argmax(i) for i in model.predict(x_test)]
y_test_cm = confusion_matrix(y_test,y_test_predict)

plt.figure(figsize=(7,7))
sns.heatmap(y_test_cm,annot=True,fmt='g',xticklabels=['Not Fake','Fake Account'],yticklabels=['Not Fake','Fake Account'])
plt.show()

print(classification_report(y_test,y_test_predict))


