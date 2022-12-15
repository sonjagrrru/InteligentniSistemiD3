import pandas as pd

#moja implementacija
class KNN():
    def __init__(self, X, y):
        self.baza = X
        self.cilj = y

    def predikcija(self, podatak):
        komsija = self.baza.copy(deep=True)
        j = 0

        #pravim DataFrame koji je velicine baze i u svakoj vrsti ima dupliran element za koji se
        #trazi K najblizih suseda
        for i in range(len(self.baza)):
            komsija.iloc[i] = podatak

        #racunanje rastojanja od svakog elementa baze za nas element koristeci DataFrame
        komsija=komsija.sub(self.baza)
        komsija = komsija.abs()
        rastojanja = komsija.sum(axis=1) #zbir po vrstama, menhetn rastojanje
        resenje = self.cilj.copy(deep=True)
        resenje.insert(loc=0, column='rastojanja', value=rastojanja)

        #biramo n najblizih suseda
        resenje = resenje.nlargest(n=40, columns='rastojanja', keep='first')
        predik = resenje.loc[:, ['status']]

        #vraca vrednost koja se najcesce ponavlja
        return predik.mode()



