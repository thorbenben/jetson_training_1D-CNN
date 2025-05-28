# Edge AI Projekt: jetson_training_1D-CNN
Maschinelles Lernen für Sensorüberwachung: Training auf Jetson, Inferenz auf Pi

## 📝 Überblick

Dieses Projekt ist ein kleines Experiment im Bereich des Maschinellen Lernens. Ziel ist es, ein Modell zur Überwachung von Sensordaten zu erstellen. Das Besondere daran:

* Das **Training** und **Fine-Tuning** des Modells soll auf einem leistungsstärkeren System wie einem NVIDIA Jetson Orin (Nano) stattfinden.
* Die **Inferenz** (also die Anwendung des trainierten Modells) läuft auf einem Raspberry Pi (3B / 02W).
* Es wird versucht, das Modell gegenüber **Data Drift** robuster zu gestalten, indem es auf veränderten ("gedrifteten") Daten nachtrainiert (fine-getuned) wird.

Das Projekt simuliert verschiedene Betriebszustände einer Maschine (Normal, Lagerschaden, Antriebsverschleiß, Pumpendefekt) anhand von vier Sensorwerten (Temperatur, Vibration, Druck, Leistung).

## Szenario

   Mit diesem Projekt soll **OnDeviceTraining** auf einem EdgeAI Server / Trainings System dargestellt werden.  
   Somit wird ein training/finetuning ohne Verbindung zum Internet oder externen Servern ermöglicht.  
   Dabei kann jedoch immernoch ein **effizientes Training** durchgeführt werden, da die GPU des (im vergleich zum PI) leistungsstarken Nvidia Jetson Orin Nano Super verwendet wird.  

Der **Nvidia Orin Nano Super** wurde für den Test in den **"MAXNSuper" Mode** geschaltet, das Training kann jedoch auch mit jeder anderen Energievariante durchgeführt werden.  

Der Raspberry Pi 3B steht hierbei **stellvertretend für kleine, teils bereits integrierte Systeme** in der Maschine. Diese sind oft veraltet oder verfügen über zu wenig Rechenleistung, um ein eigenes Training durchzuführen,  eine Inferenz ist jedoch möglich. 
Der Nvidia Jetson Orin Nano Super hingegen (in diesem Fall auf dem Developer Dev Board) kann in verschiedenste Szenarien integriert werden und als **zentraler, lernfähiger Edge-Server** in abgeschotteten Systemen eingesetzt werden.



## 📂 Projektstruktur
```bash
.
└── jetson_training_1D-CNN/
    ├── Jetson/
    │   ├── convert_to_tflite.py
    │   ├── generate_data.py
    │   ├── train_or_finetune.py
    │   ├── data/
    │   │   └── (generierte Daten von generate_data.py)
    │   └── models/
    │       ├── (Modell und Statisiken von train_or_finetune.py / convert_to_tflite.py)
    │       └── scaler.pkl
    ├── Raspberry Pi/
    │   ├── inference_on_pi.py
    │   ├── data/
    │   │   └── (generierte Daten von generate_data.py)
    │   └── models/
    │       ├── (mindestens die erstellten tflite modells)
    │       └── (Statistiken der inferenz)
    ├── Dokumentation_run_23.05.2025/
    │   ├── finetuning_23_05_2025/
    │   │   ├── (Screenshots von JPG und jtop + csv von JPG)
    │   │   └── (confusion_matrix.png)
    │   ├── initial_training_23_05_2025/
    │   │   ├── (Screenshots von JPG und jtop + csv von JPG)
    │   │   └── (confusion_matrix.png)
    │   └── pi_inferenz_23_05_2025/
    │       └── (confusion_matrix.png + inference results)
    └── README.md

```

## Workflow

**Empfehlung:**
Nutze Venv

1.  **Daten generieren**:
    Führe `generate_data.py` aus. Dieses Skript erstellt verschiedene Datensätze für Training, Inferenz und Simulation von Data Drift und speichert sie im `data/` Ordner.
    ```bash
    python3 generate_data.py
    ```

2.  **Modell trainieren oder fine-tunen**:
    Führe `train_or_finetune.py` aus. Das Skript fragt dich, ob du ein neues Modell trainieren oder ein bestehendes Modell fine-tunen möchtest.
    ```bash
    python3 train_or_finetune.py
    ```
    * Beim **initialen Training** wird ein neues Modell erstellt, trainiert und als `model.keras` im `models/` Ordner gespeichert. Der verwendete `StandardScaler` wird als `scaler.pkl` gespeichert.
    * Beim **Fine-Tuning** wird das `model.keras` geladen, auf Drift-Daten weiter trainiert und als `model_finetuned.keras` gespeichert.

3.  **Modell zu TFLite konvertieren**:
    Führe `convert_to_tflite.py` aus. Das Skript fragt dich, welches Keras-Modell (`model.keras` oder `model_finetuned.keras`) du konvertieren möchtest. Das Ergebnis wird als `.tflite`-Datei im `models/` Ordner gespeichert.
    ```bash
    python3 convert_to_tflite.py
    ```

4.  **Inferenz auf dem Raspberry Pi**:
    Führe `inference_on_pi.py` aus. Dieses Skript ist für den Raspberry Pi gedacht, kann aber auch auf einem normalen PC mit den entsprechenden Bibliotheken laufen. Es fragt dich:
    * Welches TFLite-Modell verwendet werden soll (initial oder fine-tuned).
    * Welcher Datensatz für die Inferenz genutzt werden soll (Standard-Inferenzdaten oder Drift-Inferenzdaten).
    ```bash
    python3 inference_on_pi.py
    ```
    Die Ergebnisse und eine Konfusionsmatrix werden im `models/` Ordner gespeichert.

5.  **Scritt 2-4 mit den drfited Datensätzen wiederholen**


## 🐍 Skript-Details

### `generate_data.py`

* Erzeugt simulierte Zeitreihen-Sensordaten für vier Klassen: "Normal", "Lagerschaden", "Antriebsverschleiß", "Pumpendefekt".
* Jeder Datensatz besteht aus Sequenzen von Sensorwerten (Temperatur, Vibration, Druck, Leistung).
* Folgende Datensätze werden erstellt und im `data/` Ordner als `.npy`-Dateien gespeichert:
    * `train_data.npy`, `train_labels.npy`: Für das initiale Training.
    * `infer_data.npy`, `infer_labels.npy`: Für die Evaluation/Inferenz (leichte Abweichungen vom Trainingsdatensatz).
    * `drift_train_data.npy`, `drift_train_labels.npy`: Für das Fine-Tuning des Modells (simuliert Data Drift).
    * `drift_infer_data.npy`, `drift_infer_labels.npy`: Für die Evaluation/Inferenz unter Drift-Bedingungen.

### `train_or_finetune.py`

* **Interaktive Auswahl**: Fragt beim Start, ob ein initiales Training (Modus 1) oder ein Fine-Tuning (Modus 2) durchgeführt werden soll.
* **Initiales Training (Modus 1)**:
    * Lädt die `train_data.npy`.
    * Erstellt ein 1D Convolutional Neural Network (CNN).
    * Trainiert das Modell.
    * Speichert das trainierte Modell als `model.keras` und den `StandardScaler` als `scaler.pkl` im `models/` Ordner.
    * Fragt optional nach einer Evaluation auf Standard-Inferenzdaten und Drift-Inferenzdaten.
* **Fine-Tuning (Modus 2)**:
    * Lädt das existierende `model.keras` und `scaler.pkl`.
    * Lädt die `drift_train_data.npy`.
    * Trainiert das Modell mit einer niedrigeren Lernrate weiter.
    * Speichert das fine-getunete Modell als `model_finetuned.keras`.
    * Fragt optional nach einer Evaluation auf Standard-Inferenzdaten und Drift-Inferenzdaten.
* **Callbacks**: Nutzt EarlyStopping und ReduceLROnPlateau während des Trainings.
* **Plots**: Speichert Trainingsverlaufs-Graphen und Konfusionsmatrizen der Evaluationen.

### `convert_to_tflite.py`

* **Interaktive Auswahl**: Fragt, ob `model.keras` oder `model_finetuned.keras` konvertiert werden soll.
* **Konvertierung**: Wandelt das ausgewählte Keras-Modell in ein TensorFlow Lite (`.tflite`) Modell um.
    * Verwendet Standardoptimierungen (kann Float16-Quantisierung beinhalten).
    * Nutzt optional einen repräsentativen Datensatz (aus den initialen Trainingsdaten), um die Quantisierungsqualität potenziell zu verbessern.
* **Speichern**: Das konvertierte Modell wird im `models/` Ordner als `model.tflite` oder `model_finetuned.tflite` gespeichert.
* **Verifizierung**: Führt einen kurzen Test mit dem konvertierten TFLite-Modell durch.

### `inference_on_pi.py`

* **Interaktive Auswahl**:
    * Fragt, welches `.tflite`-Modell verwendet werden soll (initiales oder fine-getuntes).
    * Fragt, welcher Inferenzdatensatz verwendet werden soll (`infer_data.npy` oder `drift_infer_data.npy`).
* **Inferenz**:
    * Lädt das ausgewählte TFLite-Modell und den `scaler.pkl`.
    * Lädt die ausgewählten Inferenzdaten.
    * Führt Vorhersagen für die Inferenzdaten durch.
    * Misst die durchschnittliche Inferenzzeit pro Sample.
* **Auswertung**:
    * Berechnet Metriken wie Genauigkeit, F1-Score, ROC-AUC.
    * Gibt einen Klassifikationsbericht aus.
    * Speichert eine Konfusionsmatrix als Bild (`.png`) und die gesamten Ergebnisse als Textdatei (`.txt`) im `models/` Ordner. Die Dateinamen spiegeln das verwendete Modell und den Datensatz wider.

## 🛠️ Setup und Abhängigkeiten

Stelle sicher, dass du Python (z.B. Version 3.8+) installiert hast. Die benötigten Bibliotheken können mit `pip` installiert werden:

* Vorbereitung (Empfehlung)

```bash
sudo apt update
sudo apt install python3-venv -y
python3 -m venv .venv  ODER (.env)
source .venv/bin/activate
pip install --upgrade pip

```

* **Nvidia Jetson Orin Nano Super**

```bash
pip install ...

	• tensorflow==2.16.1+nv24.8
# (pip install https://developer.download.nvidia.com/compute/redist/jp/v61/tensorflow/tensorflow-2.16.1+nv24.08-cp310-cp310-linux_aarch64.whl) -> Abhängig von der JP version, kann die verwendung der GPU beeinflussen !
	• "numpy<2.0" -> 1.26.4
	• scikit-learn==1.6.1
	• matplotlib
	• Pandas==2.2.3
	• seaborn

# Nvidia Jetson Orin Nano Super JetPack v6.1

```

* **Raspberry PI**
```bash
pip install ...

	• tflite-runtime==2.14.0 (
	• "Numpy<2.0" -> 1.26.4
	• scikit-learn==1.6.1
	• matplotlib
	• pandas==2.2.3
	• seaborn

# Beispiel für Raspberry Pi 3B Bookworm

```
