import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

#podesavanje ispisa tabele
pd.set_option('display.max_columns', 15) #maks broj atributa
pd.set_option('display.width', None) #svi atributi u jednom redu


#CITANJE PODATAKA I NJIHOVA ANALIZA
data = pd.read_csv('datasets/car_purchase.csv') #dataframe tip vraca
print(data.head(10)) #prvih 10 podataka iz tabele
print('-------------------------------------------')
print(data.tail(10)) #poslednjih 10 podataka iz tabele
print('-------------------------------------------')
print(data.info()) #informacije o kolonama
print('-------------------------------------------')
print(data.describe()) #informacije o atributima
print(data.describe(include=[object]))

#Graficki prikaz zavisnosti maksimalne cene od svakog pojedinacnog atributa
usefullData = data.loc[:, ['annual_salary', 'max_purchase_amount']]
usefullData = usefullData.sort_values(by='annual_salary')
plt.scatter(usefullData['annual_salary'], usefullData['max_purchase_amount'],
            color="green", s=10, label='Max purchase dependency')
plt.legend()
plt.xlabel('annual_salary')
plt.ylabel('max_purchase_amount')
plt.show()

usefullData = data.loc[:, ['age', 'max_purchase_amount']]
usefullData = usefullData.sort_values(by='age')
plt.scatter(usefullData['age'], usefullData['max_purchase_amount'],
            color="blue", s=10, label='Max purchase dependency')
plt.legend()
plt.xlabel('age')
plt.ylabel('max_purchase_amount')
plt.show()

usefullData = data.loc[:, ['credit_card_debt', 'max_purchase_amount']]
usefullData = usefullData.sort_values(by='credit_card_debt')
plt.scatter(usefullData['credit_card_debt'], usefullData['max_purchase_amount'],
            color="red", s=10, label='Max purchase dependency')
plt.legend()
plt.xlabel('credit_card_debt')
plt.ylabel('max_purchase_amount')
plt.show()

usefullData = data.loc[:, ['net_worth', 'max_purchase_amount']]
usefullData = usefullData.sort_values(by='net_worth')
plt.scatter(usefullData['net_worth'], usefullData['max_purchase_amount'],
            color="purple", s=10, label='Max purchase dependency')
plt.legend()
plt.xlabel('net_worth')
plt.ylabel('max_purchase_amount')
plt.show()



#NEMA NaN VREDNOSTI NI ZA JEDAN ATRIBUT!

#nebitni atributi: customer_id, gender
usefullData = data.loc[:, ['age', 'annual_salary', 'credit_card_debt', 'net_worth']]
finals = data.loc[:, 'max_purchase_amount'] #kada nemaju [[]] onda je ovo tipa series odnosno jedna kolona
le = LabelEncoder()
#usefullData['gender'] = le.fit_transform(usefullData['gender']) #strung pol prebacuje u 0 ili 1

#delimo podatke na dva dela, train i test, gde se za train uzima 60% podataka
#random_state koja je razlicita od 0 znaci da ce svaki put kad se pokrene program, iste vrste biti odabrane
#za test/train. Ako svaki put zelis drugi izbor podesi ovo na 0
X_train, X_test, y_train, y_test = train_test_split(usefullData, finals, test_size=0.4,
                                                    random_state=1)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print('-------------------------------------------')
print('Koeficijenti: ', reg.coef_)

#odstupanje(variance score) = 1 – Var{y – y’}/Var{y}, u idealnom slucaju je 1
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# grafik za odstupanje linearnog modela od stvarne vrednosti (err) - Residual error
# Y = L(X) + err
plt.style.use('fivethirtyeight')

# graficko predstavljanje greske za training podatke
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

# graficko predstavljanje greske za test podatke
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

# x osa
plt.hlines(y=0, xmin=0, xmax=80000, linewidth=2)

plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()

#Gradijentni spust

from LinearRegressionGradientDescent import LinearRegressionGradientDescent

usefullData = data.loc[:, ["age", "annual_salary", "credit_card_debt", "net_worth"]]
finals = data.loc[:, 'max_purchase_amount']

# Sve atribute skaliramo da im vrednosti budu reda velicine 10
usefullData["annual_salary"] = usefullData["annual_salary"]/1000.0
usefullData["net_worth"] = usefullData["net_worth"]/10000.0
usefullData["credit_card_debt"] = usefullData["credit_card_debt"]/1000.0
label = finals/1000.0

X_train, X_test, y_train, y_test = train_test_split(usefullData, label, train_size=0.7, random_state=123, shuffle=False)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

lrgd = LinearRegressionGradientDescent()
lrgd.fit(X_train, y_train)

#koraci ucenja za svaki atribut
learning_rates = np.array([[1], [0.0001], [0.0001], [0.0001], [0.0001]])
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, 200)
y_prediction = pd.DataFrame(data=lrgd.predict(X_test), columns=['LRGD_prediction'])

#sjedinjenje tabela
lrgd_output = X_test.join(y_prediction)
lrgd_output = lrgd_output.join(y_test)
print("Predikcije linearne regresije metodom gradijentnig spusta:\n", lrgd_output)



