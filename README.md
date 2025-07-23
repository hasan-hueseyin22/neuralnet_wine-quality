# Wine Quality Prediction using a Neural Network

## 📝 Projektbeschreibung

Dieses Projekt implementiert ein neuronales Netz zur Vorhersage der Qualität von Weißwein. Basierend auf 11 chemischen Merkmalen (z.B. Säuregehalt, Zucker, Alkohol) wird das Modell trainiert, die Weinqualität auf einer Skala von 3 bis 9 zu klassifizieren.

Dieses Projekt demonstriert den Aufbau eines Deep-Learning-Modells von Grund auf und den Einsatz moderner Werkzeuge zur automatischen Leistungsoptimierung.

**Besonderheiten des Projekts:**
-   Implementierung eines Feedforward-Neuronalen-Netzes mit **TensorFlow/Keras**.
-   Systematische **Hyperparameter-Optimierung** mit **KerasTuner** zur Findung der optimalen Modellarchitektur (Anzahl der Schichten, Neuronen, Dropout-Rate etc.) und des besten Lernalgorithmus.
-   Anwendung von Best Practices wie Daten-Skalierung und Early Stopping zur Vermeidung von Overfitting.

**Datensatz:** [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality) von der UCI Machine Learning Repository.

## 🛠️ Tech Stack

-   **Python**
-   **Pandas** für die Datenmanipulation
-   **Scikit-learn** für Datenvorverarbeitung (Skalierung, Splitting)
-   **TensorFlow/Keras** für den Aufbau und das Training des neuronalen Netzes
-   **KerasTuner** für die Hyperparameter-Optimierung

## 🚀 Installation und Ausführung

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/hasan-hueseyin22/neuralnet_wine-quality.git](https://github.com/hasan-hueseyin22/neuralnet_wine-quality.git)
    cd neuralnet_wine-quality
    ```

2.  **Virtuelle Umgebung erstellen und aktivieren (empfohlen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Auf Windows: venv\Scripts\activate
    ```

3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Modell optimieren und trainieren:**
    Das Skript startet den gesamten Prozess: Daten-Download, Hyperparameter-Suche mit KerasTuner, Training des finalen Modells mit den besten gefundenen Parametern und anschließende Evaluierung.
    ```bash
    python src/train.py
    ```
    *Hinweis: Die Hyperparameter-Suche kann je nach `MAX_TRIALS` und deiner Hardware einige Zeit in Anspruch nehmen.*

## 📊 Ergebnisse

Nach Abschluss der Suche gibt das Skript eine Zusammenfassung der besten Hyperparameter aus. Anschließend wird das finale Modell mit diesen Parametern trainiert und auf dem Test-Set evaluiert. Die finale Genauigkeit (`accuracy`) wird in der Konsole angezeigt. Das trainierte Modell wird als TensorFlow/Keras-Modell im Ordner `models/best_wine_quality_model` gespeichert.

## 📂 Repository-Struktur
 ```
neuralnet_wine-quality/
├── data/
├── models/
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```
