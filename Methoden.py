import numpy as np
import matplotlib.pyplot as plt

#Plotting von Daten-----------------------------------------------------------------------------------------------------
plt.scatter(X[:, 0], X[:, 1], c=y) # Punkte plotten, c=y färbt die Punkte nach ihren Labels
plt.plot(np.array[0, 1], f(np.array[0, 1])) # die Funktion f plotten
plt.axis((0, 1, 0, 1))                      # Achsenbegrenzungen setzen
plt.show()
#Erstellung von h(x)----------------------------------------------------------------------------------------------------
h = lambda w, x: np.sign(np.dot(np.transpose(w), x))
#PLA Algorithmus--------------------------------------------------------------------------------------------------------
def pla(x: np.ndarray, y: np.ndarray, w: np.ndarray):   # optional Typisierung
    x_calc = np.column_stack((np.ones(len(x)), x))      # Bias hinzufügen als erste Spalte in dem x array

    for i, element in enumerate(x):     # für jedes Element in x
        if h(w, x_calc[i]) != y[i]:     # wenn die Vorhersage falsch ist
            w = w + y[i] * x_calc[i]    # Gewichte anpassen
            return pla(x, y, w)         # rekursiv aufrufen, bis alle Punkte korrekt klassifiziert sind

    return w                            # wenn alle Punkte korrekt klassifiziert sind, Gewichte zurückgeben
#Steigung der Geraden berechnen-----------------------------------------------------------------------------------------
def p(x, w):
    m = -(w[1] / w[2])
    b = -(w[0] / w[2])

    return m * x + b
#Dimension eines Arrays bekommen----------------------------------------------------------------------------------------
array = [[1, 2, 3], [4, 5, 6]]  # Beispielarray
array.shape # gibt die Dimension von dem Array "array" zurück
#-----------------------------------------------------------------------------------------------------------------------
def lin_reg(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(X.shape[0]), X])  # Bias hinzufügen als erste Spalte in dem X array
    X_dagger = np.linalg.pinv(X1)  # Pseudo-Inverse berechnen, die Methode macht schon alles alleine
    w = X_dagger @ y                         # Gewichte berechnen
    return w
#Lineare Regression-----------------------------------------------------------------------------------------------------
w_2_1 = lin_reg(X_2_1, y_2_1)         # Gewichtsvektor berechnen für Datensatz 2_1
def p_2_1(x):                         # Methode um die Gerade zu berechnen
    result = w_2_1[0] + w_2_1[1] * x  # Berechnung der Geraden
    return result