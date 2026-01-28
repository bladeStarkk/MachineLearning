# Notizen zu Machine Learning, um mich auf die Klausur vorbereiten zu können
## Nach den Uebungen sortiert
[Uebung 1](#uebung-1)
[Uebung 2](#uebung-2)


# Uebung 1:
Welches Thema?
- [Grundbegriffe](#grundbegriffe)
- [Lernarten](#lernarten)
- [Das Perzeptron](#das-perzeptron)

## Codebeispiele mit Notizen
Daten visualisieren:
```python
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=y) # Punkte plotten, c=y färbt die Punkte nach ihren Labels
plt.plot(np.array[0, 1], f(np.array[0, 1])) # die Funktion f plotten
plt.axis((0, 1, 0, 1)) # Achsenbegrenzungen setzen
plt.show()
```
PLA Algorithmus:
```python
def pla(x: np.ndarray, y: np.ndarray, w: np.ndarray):   # optional Typisierung
    x_calc = np.column_stack((np.ones(len(x)), x))      # Bias hinzufügen als erste Spalte in dem x array

    for i, element in enumerate(x):     # für jedes Element in x
        if h(w, x_calc[i]) != y[i]:     # wenn die Vorhersage falsch ist
            w = w + y[i] * x_calc[i]    # Gewichte anpassen
            return pla(x, y, w)         # rekursiv aufrufen, bis alle Punkte korrekt klassifiziert sind

    return w                # wenn alle Punkte korrekt klassifiziert sind, Gewichte zurückgeben
```
Gerade aus den Gewichten berechnen:
```python
def p(x, w):
    m = -(w[1] / w[2])
    b = -(w[0] / w[2])

    return m * x + b
```
Dimension eines Arrays herausfinden:
```python
array.shape # gibt die Dimension von dem Array "array" zurück
```

## Grundbegriffe
- **features oder Eigenschaften:** Merkmale, die einen Datensatz beschreiben (z.B. Alter, Einkommen)
- **feature vector:** Sammlung von features, die einem Datenpunkt zugeordnet sind
- **labels oder Zielvariable:** Das, was vorhergesagt werden soll (z.B. ob ein Kunde einen Kredit bekommt) 
wird mit einer +1 oder -1 codiert
- **funktion:** Die Beziehung zwischen Features und Labels, man gibt features (x) ein und bekommt labels (y) raus
- **trainingsdaten:** Datensatz, der zum Trainieren des Modells verwendet wird
- **testdaten:** Datensatz, der zum Testen des Modells verwendet wird   
- **Algorithmus:** ein Lernalgorithmus sucht sich mit den Daten eine Funktion, die die Daten gut beschreibt
### Notizen dazu um einen besseren Eindruck zu bekommen:
Ein Kunde ist ein Datenpunkt oder feature vector in dem Beispiel bei der Kreditvergabe.

## Lernarten
- **überwachtes Lernen (supervised learning):**
  - Ziel: eine Funktion zu finden, die Eingabedaten (features) auf Ausgabedaten (labels) abbildet
  - Beispiel: Klassifikation (z.B. Spam-Erkennung), Regression (z.B. Vorhersage von Hauspreisen)
  - Dazu eine Funktion g finden, die die perfekte Funktion f approximiert
- **unüberwachtes Lernen (unsupervised learning):**
  - Ziel: Muster oder Strukturen in den Eingabedaten zu finden, ohne dass es explizite Ausgabedaten gibt
  - Beispiel: Clustering (z.B. Kundensegmentierung), Dimensionsreduktion (z.B. PCA)
  - Also ein F finden, was die Daten die man bekommen hat gut beschreibt
- **bestärkendes Lernen (reinforcement learning):**
  - Ziel: eine Strategie zu lernen, um in einer Umgebung Belohnungen zu maximieren
  - Beispiel: Spiele spielen (z.B. Schach), Robotik (z.B. autonome Fahrzeuge)
  - das Ziel hier ist eine Funktion zu finden, die die unbekannte beste Strategie S approximiert
## Das Perzeptron
- **Idee:**
  - Komponenten des feature vectors(also die features selber) werden anders gewichtet
  - dadurch wird der Vektor w erzeugt, der die Gewichte enthält
  - bei 2 Features, bildet der Vektor w eine Gerade im 2D Raum
  - die Gerade teilt den Raum in 2 Hälften
  - Diese Gerade soll die Datenpunkte so trennen, dass alle Punkte einer Klasse auf einer Seite der Gerade liegen
  - Die Gerade wird mit jedem Schritt angepasst, bis alle Punkte korrekt klassifiziert sind

# Uebung 2:
Welches Thema?
- [In-Sample Fehler und Out-of-Sample Fehler](#in-sample-fehler-und-out-of-sample-fehler)
- [Supervised Learning Arten (Lineare Modelle)](#supervised-learning-arten-lineare-modelle)
- [Nichtlineare Transformationen](#nichtlineare-transformationen)

## Codebeispiele mit Notizen
Pseudo-Inverse Methode zur linearen Regression:
```python
 def lin_reg(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(X.shape[0]), X])  # Bias hinzufügen als erste Spalte in dem X array
    X_dagger = np.linalg.pinv(X1)  # Pseudo-Inverse berechnen, die Methode macht schon alles alleine
    w = X_dagger @ y                         # Gewichte berechnen
    return w
```
Gerade berechnen und Anwendung von der Pseudo-Inverse Methode:
```python
w_2_1 = lin_reg(X_2_1, y_2_1)         # Gewichtsvektor berechnen für Datensatz 2_1
def p_2_1(x):                         # Methode um die Gerade zu berechnen
    result = w_2_1[0] + w_2_1[1] * x  # Berechnung der Geraden
    return result
```
```python
# Perzeptron-Entscheidungsfunktion
def h(x: np.ndarray, w: np.ndarray) -> int:     # findet h(x)
    return np.sign(np.dot(w, x))


# Berechnet den Fehler E_in
def e_in(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:     #berechnet den E_in Fehler
    samples = sum((h(x_i, w) != y_i for x_i, y_i in zip(X, y)))
    return (1 / X.shape[0]) * samples


def pla_pocket(X, y, w, T, pocket):                                 # PLA Algorithmus
    if T == 0:                                                      # ruft sich selbst auf, bis T = 0
        return pocket
    X_stacked = np.column_stack([np.ones(X.shape[0]), X])           # Bias hinzufügen als erste Spalte in dem X array
    for x_i, y_i in zip(X_stacked, y):                              # für jedes Element in X und y
        h_i = h(x_i, w)                                             # Vorhersage berechnen
        if h_i == y_i:                                              # wenn die Vorhersage korrekt ist
            continue

        # neues w ausrechnen, weiteriterieren und eventuell neues pocket speichern
        w_next = w + x_i * y_i
        # wenn kein pocket existiert oder der neue Fehler kleiner ist:
        if pocket is None or e_in(w_next, X_stacked, y) < e_in(pocket, X_stacked, y):   
            pocket = np.copy(w_next)                                # neues pocket speichern

        return pla_pocket(X, y, w_next, T - 1, pocket)              # rekursiv aufrufen mit neuem w und T-1
    return w
```


## In-Sample Fehler und Out-of-Sample Fehler
- **In-Sample Fehler (Einpassungsfehler):**
  - Fehler, der auf den Trainingsdaten gemessen wird
  - Man kann den Fehler schon schon erkennen, weil der Fehler auf den Trainingsdaten berechnet wird
  - Berechnung: 
  - $( E_{in} = \frac{1}{N} \sum_{i=1}^{N} L(h(x_i), y_i) )$
    - $N$: Anzahl der Trainingsdatenpunkte
    - $L$: Verlustfunktion (z.B. 0-1 Verlust für Klassifikation)
    - $h(x_i)$: Vorhersage des Modells für den Datenpunkt $x_i$
    - $y_i$: tatsächliches Label des Datenpunktes $x_i$
  - Fehler ist also die Anzahl der Fehler geteilt durch die Anzahl der Datenpunkte
- **Out-of-Sample Fehler (Allgemeinerungsfehler):**
  - kann nicht genau bestimmt werden, da die Testdaten unbekannt sind
  - wird geschätzt auf Testdaten, die wir nicht zum Trainieren verwendet haben

## Supervised Learning Arten (Lineare Modelle)
- **[Klassifikation](#lineare-klassifikation):**
  - Bild von f: diskrete Klasen
  - Beispiel: Kreditvergabe (ja oder nein)
  - Algorithmus: PLA, Pocket, etc.
- **[Regression](#lineare-regression):**
  - Bild von f: reelle Zahlen
  - Beispiel: Hoehe des zu vergebenden Kredits
  - Algorithmus: Pseudo-Inverse
- **[Logistische Regression](#logistische-regression):**
  - Bild von f: Wahrscheinlichkeiten
  - Beispiel: Wahrscheinlichkeit fuer Zahlungsausfall
  - Algorithmus: Gradientenabstieg
### Lineare Klassifikation
- Lineare Modelle kombinieren Feautures linear miteinander
- Linear separierbare Daten: PLA 
- Nicht linear separierbare Daten (Rauschkontamination): Pocket Algorithmus 
- Nicht linear separierbare Daten (nichtlineare Target Function): Nicht-lineare Transformation oder andere Modelle
### Lineare Regression
- Beispiel eine Bank will die Festlegung von kreditrahmen automatisieren
- Bank hat einen Datensatz, von Kunden die schon ein Kredit bekommen haben und deren features
- Verschieden Angestellte entscheiden aber anders -> Rauschen im Datensatz
- Annahme: Eine lineare Funktion kann y approximieren, mit rauschen
- Ziel: finde ein w, dass den in-sample Fehler minimiert
- $ \mathbf{w} = (X^T X)^{-1} X^T \mathbf{y} $
### Logistische Regression
wird erst spaeter behandelt

## Nichtlineare Transformationen
- **Nichtlineare Transformation der Features:**
  - $ h(\mathbf{x}) = \text{sign}(-0.6 + x_1^2 + x_2^2) $
  - $ h(\mathbf{x}) = \text{sign} \left( \underbrace{(-0.6)}_{\tilde{w}_0} \cdot \underbrace{1}_{z_0} + \underbrace{1}_{\tilde{w}_1} \cdot \underbrace{x_1^2}_{z_1} + \underbrace{1}_{\tilde{w}_2} \cdot \underbrace{x_2^2}_{z_2} \right) $
  - $ = \text{sign} \left( \underbrace{ \begin{bmatrix} \tilde{w}_0 & \tilde{w}_1 & \tilde{w}_2 \end{bmatrix} }_{\tilde{\mathbf{w}}^T} \underbrace{ \begin{bmatrix} 1 \\ z_1 \\ z_2 \end{bmatrix} }_{\mathbf{z}} \right) $
  - Algorithmen bleiben gleich, nur die Daten wurden transformiert

# Uebung 3:
Welches Thema?

## Codebeispiele mit Notizen

## 