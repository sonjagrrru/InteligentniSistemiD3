import numpy as np
import pandas as pd
import seaborn as sb

#podesavanje ispisa tabele
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

pd.set_option('display.max_columns', 20) #maks broj atributa
pd.set_option('display.width', None) #svi atributi u jednom redu


#CITANJE PODATAKA I NJIHOVA ANALIZA
data = pd.read_csv('datasets/car_state.csv') #dataframe tip vraca
print(data.head(10)) #prvih 10 podataka iz tabele
print('-------------------------------------------')
print(data.tail(10)) #poslednjih 10 podataka iz tabele
print('-------------------------------------------')
print(data.info()) #informacije o kolonama
print('-------------------------------------------')
print(data.describe())
print('-------------------------------------------')
print(data.describe(include=[object]))#informacije o atributima

#NEMA NaN VREDNOSTI NI ZA JEDAN ATRIBUT!
#svi atributi su tipa string i to treba da se promeni

#Graficki prikaz zavisnosti izlaza od svakog atributa pojedinacno
sb.displot(data[data['buying_price'].notna()], x='buying_price', hue='status',
           multiple='fill', bins=16)
plt.show()

sb.displot(data[data['maintenance'].notna()], x='maintenance', hue='status',
           multiple='fill', bins=16)
plt.show()

sb.displot(data[data['doors'].notna()], x='doors', hue='status',
           multiple='fill', bins=16)
plt.show()

sb.displot(data[data['seats'].notna()], x='seats', hue='status',
           multiple='fill', bins=16)
plt.show()

sb.displot(data[data['safety'].notna()], x='safety', hue='status',
           multiple='fill', bins=16)
plt.show()

sb.displot(data[data['trunk_size'].notna()], x='trunk_size', hue='status',
           multiple='fill', bins=16)
plt.show()

#IZBACITI BROJ VRATA? ILI SAMO STAVITI 2 I 2 ILI VISE

ohe = OneHotEncoder(dtype=int, sparse=False)
# fit_transform zahteva promenu oblika
buying_price = ohe.fit_transform(data['buying_price'].to_numpy().reshape(-1, 1))
data.drop(columns=['buying_price'], inplace=True)
data = data.join(pd.DataFrame(data=buying_price,
columns=ohe.get_feature_names_out(['buying_price'])))

safety = ohe.fit_transform(data['safety'].to_numpy().reshape(-1, 1))
data.drop(columns=['safety'], inplace=True)
data = data.join(pd.DataFrame(data=safety,
columns=ohe.get_feature_names_out(['safety'])))

maintenance = ohe.fit_transform(data['maintenance'].to_numpy().reshape(-1, 1))
data.drop(columns=['maintenance'], inplace=True)
data = data.join(pd.DataFrame(data=maintenance,
columns=ohe.get_feature_names_out(['maintenance'])))

#menjamo vrednosti za vrata
data['doors'] = np.where((data.doors != '2'), '3 or more', data.doors)
#menjamo vrednosti za gepek
data['trunk_size'] = np.where((data.trunk_size != 'small'), 'regular/big', data.trunk_size)
#menjamo vrednosti za broj sedista
data['seats'] = np.where((data.seats != '2'), '4 or more', data.seats)
#kada dobijes podatak za procenu ovo treba da mu izmenjas!

le = LabelEncoder()
data['doors'] = le.fit_transform(data['doors'])
data['seats'] = le.fit_transform(data['seats'])
data['trunk_size'] = le.fit_transform(data['trunk_size'])


# Create feature and target arrays
X = data.loc[:, ['doors', 'seats', 'trunk_size', 'buying_price_high', 'buying_price_low', 'buying_price_medium',
                 'buying_price_very high', 'safety_high', 'safety_low', 'safety_medium', 'maintenance_high',
                 'maintenance_low', 'maintenance_medium', 'maintenance_very high']]
y = data.loc[:, 'status']

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.7, random_state=1)

knn = KNeighborsClassifier(n_neighbors=36)
knn.fit(X_train, y_train) #ubaci u model
prediction = pd.Series(knn.predict(X_test))

print('Preciznost testa: ', knn.score(X_test, y_test)*100, '%')

from KNNmoj import KNN

y = data.loc[:, ['status']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

#velicina testa je 20% od 1398

moj = KNN(X_train, y_train)
predikcija = y_test.copy(deep=True)
tacni = 0
netacni = 0
for iter in range(len(X_test)):
    predikcija.iloc[iter] = moj.predikcija(X_test.iloc[iter])
    # print(predikcija.iloc[iter])
    # print(y_test.iloc[iter])
    # print(predikcija.iloc[iter]==y_test.iloc[iter])
    # print('-----------------------------')
    if predikcija.iloc[iter].equals(y_test.iloc[iter]):
        tacni = tacni + 1
    else:
        netacni = netacni + 1
print('-------------------------------------------')
preciznost = tacni * 100.0/(tacni + netacni)
print('Preciznost moje KNN implementacije iznosi: ', preciznost)


