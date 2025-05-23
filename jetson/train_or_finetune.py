import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import pandas as pd # für Confusion Matrix Darstellung
import seaborn as sns # schönere Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix as sk_confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


# --- Globale Konfiguration ---
DATA_DIR = "data"
MODEL_DIR = "models"

# Dateinamen vereinfacht
INITIAL_MODEL_SAVE_NAME = "model.keras"
FINETUNED_MODEL_SAVE_NAME = "model_finetuned.keras"
SCALER_NAME = "scaler.pkl"

# Daten für Training/Finetuning
INITIAL_TRAIN_DATA_PREFIX = "train"         # -> train_data.npy, train_labels.npy
FINETUNE_DRIFT_DATA_PREFIX = "drift_train"  # -> drift_train_data.npy, drift_train_labels.npy

# Daten für Evaluation (optional, nach Training/Finetuning)
INFERENCE_EVAL_DATA_PREFIX = "infer"        # -> infer_data.npy, infer_labels.npy
DRIFT_INFER_EVAL_DATA_PREFIX = "drift_infer" # -> drift_infer_data.npy, drift_infer_labels.npy


SEQUENCE_LENGTH = 60 # Muss mit generate_data.py übereinstimmen
N_FEATURES = 4       # Muss mit generate_data.py übereinstimmen
N_CLASSES = 4
CLASS_NAMES = ["Normal", "Lagerschaden", "Antriebsverschleiss", "Pumpendefekt"] # Namen für Reports

BATCH_SIZE = 64
# Epochs und Learning Rate werden je nach Modus (train/finetune) angepasst
INITIAL_EPOCHS = 40 # Weniger für schnelleres Demo-Training
FINETUNE_EPOCHS = 25
INITIAL_LEARNING_RATE = 0.0008
FINETUNE_LEARNING_RATE = 0.00005 # Sehr klein fürs Finetuning

# --- GPU-Check ---
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\n" + "*"*40)
            print("      !!!! GPU(S) GEFUNDEN !!!!")
            print(f"         Verfügbare GPUs: {gpus}")
            print("      TensorFlow kann mit der GPU genutzt werden!")
            print("*"*40 + "\n")
            return True
        except RuntimeError as e:
            # Dieser Fehler kann bei manchen Konfigurationen auftreten,
            # obwohl die GPU trotzdem genutzt werden kann.
            print("\n" + "*"*40)
            print("      ! GPU GEFUNDEN, jedoch mit Vorbehalt !")
            print(f"         Fehlerdetails: {e}")
            print("      Versuche trotzdem, die GPU zu nutzen!")
            print("*"*40 + "\n")
            return bool(tf.config.list_physical_devices('GPU')) # Erneute Prüfung
    else:
        print("\n" + "-"*40)
        print("       Keine GPU gefunden. Training läuft auf der CPU. ")
        print("      Das kann etwas länger dauern...")
        print("-"*40 + "\n")
        return False

# --- Daten laden und vorbereiten ---
def load_and_preprocess_data(data_prefix, sequence_length, n_features, num_classes, fit_scaler_mode=False, existing_scaler=None):
    data_path = os.path.join(DATA_DIR, f"{data_prefix}_data.npy")
    labels_path = os.path.join(DATA_DIR, f"{data_prefix}_labels.npy")

    X_raw = np.load(data_path)
    y_original_labels = np.load(labels_path) # Die originalen integer Labels

    # Merkmale normalisieren/standardisieren
    X_flattened = X_raw.reshape(-1, n_features) # Alle Samples und Features in 2D

    current_scaler = None
    if fit_scaler_mode: # Nur im initialen Trainings-Modus
        print("Passe Scaler an Trainingsdaten an...")
        scaler = StandardScaler()
        X_scaled_flattened = scaler.fit_transform(X_flattened)
        current_scaler = scaler # Den neuen Scaler merken
        # Scaler speichern
        scaler_save_path = os.path.join(MODEL_DIR, SCALER_NAME)
        with open(scaler_save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler gespeichert unter {scaler_save_path}")
    elif existing_scaler: # Für Finetuning oder Evaluation
        print("Verwende existierenden Scaler...")
        X_scaled_flattened = existing_scaler.transform(X_flattened)
        current_scaler = existing_scaler
    else:
        raise ValueError("Scaler muss entweder angepasst (fit_scaler_mode=True) oder übergeben werden (existing_scaler)!")

    X_scaled_reshaped = X_scaled_flattened.reshape(-1, sequence_length, n_features) # Wieder in 3D (Samples, Sequenz, Features)
    y_categorical_labels = keras.utils.to_categorical(y_original_labels, num_classes=num_classes)

    return X_scaled_reshaped, y_categorical_labels, y_original_labels, current_scaler


# --- 1D-CNN-Modell Definition ---
def create_1d_cnn_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# --- Hilfsfunktionen für Plotting und Evaluation ---
def plot_training_history(history, mode_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Genauigkeit')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validierung Genauigkeit')
    plt.title(f'Modell Genauigkeit ({mode_name})')
    plt.xlabel('Epoche')
    plt.ylabel('Genauigkeit')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Verlust')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validierung Verlust')
    plt.title(f'Modell Verlust ({mode_name})')
    plt.xlabel('Epoche')
    plt.ylabel('Verlust')
    plt.legend()
    plt.tight_layout()
    plot_filename = f"{mode_name}_training_history.png"
    plt.savefig(os.path.join(MODEL_DIR, plot_filename))
    print(f"Trainingsverlauf gespeichert: {os.path.join(MODEL_DIR, plot_filename)}")
    plt.close()

def plot_confusion_matrix_eval(cm, class_names_list, filename, plot_title_prefix=""):
    full_title = f"{plot_title_prefix} Konfusionsmatrix"
    df_cm = pd.DataFrame(cm, index=class_names_list, columns=[f"Vorh. {name}" for name in class_names_list])
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="viridis", annot_kws={"size":10})
    plt.title(full_title, fontsize=14)
    plt.ylabel('Wahre Klasse', fontsize=12)
    plt.xlabel('Vorhergesagte Klasse', fontsize=12)
    plt.tight_layout()
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    plt.savefig(os.path.join(MODEL_DIR, filename))
    print(f"Konfusionsmatrix gespeichert: {os.path.join(MODEL_DIR, filename)}")
    plt.close()

def evaluate_model_performance(model, X_eval_data, y_eval_cat_labels, y_eval_orig_labels, dataset_name_str="Evaluations-Satz"):
    print(f"\n--- Evaluiere Modell auf: {dataset_name_str} ---")
    loss, accuracy = model.evaluate(X_eval_data, y_eval_cat_labels, verbose=0)
    print(f"{dataset_name_str} - Verlust: {loss:.4f}")
    print(f"{dataset_name_str} - Genauigkeit: {accuracy:.4f}")

    y_pred_probabilities = model.predict(X_eval_data)
    y_pred_class_labels = np.argmax(y_pred_probabilities, axis=1)

    f1_macro = f1_score(y_eval_orig_labels, y_pred_class_labels, average='macro', zero_division=0)
    f1_weighted = f1_score(y_eval_orig_labels, y_pred_class_labels, average='weighted', zero_division=0)
    print(f"{dataset_name_str} - F1-Score (Macro): {f1_macro:.4f}")
    print(f"{dataset_name_str} - F1-Score (Weighted): {f1_weighted:.4f}")

    roc_auc_val = "N/A"
    try:
        y_eval_binarized = label_binarize(y_eval_orig_labels, classes=list(range(N_CLASSES)))
        if y_eval_binarized.shape[1] == N_CLASSES and N_CLASSES > 1:
            roc_auc_val = roc_auc_score(y_eval_binarized, y_pred_probabilities, multi_class='ovr', average='macro')
            print(f"{dataset_name_str} - ROC-AUC (Macro OVR): {roc_auc_val:.4f}")
        else:
            print(f"{dataset_name_str} - ROC-AUC: Nicht berechnet (Formproblem oder zu wenige Klassen).")
    except ValueError as e:
        print(f"{dataset_name_str} - ROC-AUC: Konnte nicht berechnet werden ({e}).")

    print(f"\n{dataset_name_str} - Klassifikationsbericht:")
    report_str = classification_report(y_eval_orig_labels, y_pred_class_labels, target_names=CLASS_NAMES, zero_division=0)
    print(report_str)

    cm = sk_confusion_matrix(y_eval_orig_labels, y_pred_class_labels, labels=list(range(N_CLASSES)))
    cm_filename = f"cm_{dataset_name_str.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plot_confusion_matrix_eval(cm, CLASS_NAMES, cm_filename, plot_title_prefix=dataset_name_str)
    print("--- Evaluation abgeschlossen ---")


# --- Hauptlogik ---
if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    
    # GPU-Check direkt am Anfang
    gpu_available = check_gpu()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Modus abfragen: Initiales Training oder Fine-Tuning?
    mode = ""
    while mode not in ["1", "2"]:
        mode = input("Was moechtest du tun?\n1: Initiales Modell trainieren\n2: Vorhandenes Modell finetunen\nAuswahl: ")

    model = None
    current_scaler = None
    epochs_to_run = 0
    learning_rate_to_use = 0.0
    model_save_path = ""
    training_data_prefix = ""
    history_plot_name = ""

    # --- Modus: Initiales Training ---
    if mode == "1":
        print("\n--- Modus: Initiales Training gewaehlt ---")
        training_data_prefix = INITIAL_TRAIN_DATA_PREFIX
        history_plot_name = "initial_training"
        model_save_path = os.path.join(MODEL_DIR, INITIAL_MODEL_SAVE_NAME)
        epochs_to_run = INITIAL_EPOCHS
        learning_rate_to_use = INITIAL_LEARNING_RATE

        # Daten laden (Scaler wird hier angepasst und gespeichert)
        X, y_cat, y_orig, scaler_instance = load_and_preprocess_data(
            training_data_prefix, SEQUENCE_LENGTH, N_FEATURES, N_CLASSES, fit_scaler_mode=True
        )
        current_scaler = scaler_instance # Wichtig für spaetere Evaluation

        # Train/Validation Split
        X_train, X_val, y_train_cat, y_val_cat = sklearn.model_selection.train_test_split(
            X, y_cat, test_size=0.2, random_state=42, stratify=np.argmax(y_cat, axis=1) # stratify anhand der Original-Label-Verteilung
        )
        print(f"Trainingsdaten: {X_train.shape}, Validierungsdaten: {X_val.shape}")

        # Modell erstellen
        input_shape = (SEQUENCE_LENGTH, N_FEATURES)
        model = create_1d_cnn_model(input_shape, N_CLASSES)
        model.summary()

    # --- Modus: Fine-Tuning ---
    elif mode == "2":
        print("\n--- Modus: Fine-Tuning gewaehlt ---")
        original_model_path = os.path.join(MODEL_DIR, INITIAL_MODEL_SAVE_NAME)
        if not os.path.exists(original_model_path):
            print(f"FEHLER: Initiales Modell '{INITIAL_MODEL_SAVE_NAME}' nicht im Ordner '{MODEL_DIR}' gefunden. Bitte zuerst trainieren.")
            exit()
        if not os.path.exists(os.path.join(MODEL_DIR, SCALER_NAME)):
            print(f"FEHLER: Scaler '{SCALER_NAME}' nicht im Ordner '{MODEL_DIR}' gefunden. Bitte zuerst initial trainieren.")
            exit()

        training_data_prefix = FINETUNE_DRIFT_DATA_PREFIX
        history_plot_name = "finetuning"
        model_save_path = os.path.join(MODEL_DIR, FINETUNED_MODEL_SAVE_NAME)
        epochs_to_run = FINETUNE_EPOCHS
        learning_rate_to_use = FINETUNE_LEARNING_RATE

        # Existierenden Scaler laden
        with open(os.path.join(MODEL_DIR, SCALER_NAME), 'rb') as f:
            current_scaler = pickle.load(f)
        print("Existierender Scaler geladen.")

        # Daten für Fine-Tuning laden (mit existierendem Scaler)
        X_drift, y_drift_cat, y_drift_orig, _ = load_and_preprocess_data(
            training_data_prefix, SEQUENCE_LENGTH, N_FEATURES, N_CLASSES, existing_scaler=current_scaler
        )

        # Train/Validation Split für Fine-Tuning Daten
        X_train, X_val, y_train_cat, y_val_cat = sklearn.model_selection.train_test_split(
            X_drift, y_drift_cat, test_size=0.25, random_state=42, stratify=y_drift_orig
        )
        print(f"Fine-Tuning Trainingsdaten: {X_train.shape}, Validierungsdaten: {X_val.shape}")

        # Ursprüngliches Modell laden
        model = keras.models.load_model(original_model_path)
        print(f"Modell '{INITIAL_MODEL_SAVE_NAME}' fuer Fine-Tuning geladen.")
        # Alle Layer trainierbar machen
        for layer in model.layers:
            layer.trainable = True
        model.summary()

    # --- Gemeinsame Schritte für Training/Fine-Tuning ---
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_to_use)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(f"Modell kompiliert mit Lernrate: {learning_rate_to_use}")

    print(f"Starte Training/Fine-Tuning für {epochs_to_run} Epochen...")
    # Mit EarlyStopping um overfitting zu vermeiden
    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001, verbose=1)
    ]

    history = model.fit(X_train, y_train_cat,
                        epochs=epochs_to_run,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val_cat),
                        callbacks=callbacks_list,
                        verbose=1)

    # Modell am Ende des Trainings speichern
    model.save(model_save_path)
    print(f"Modell gespeichert unter {model_save_path}")

    # Trainingsverlauf plotten
    plot_training_history(history, history_plot_name)

    # --- Optionale Evaluation nach Training/Fine-Tuning ---
    eval_choice = input(f"\nMoechtest du das {history_plot_name.replace('_', ' ')} Modell jetzt auf Standard-Inferenzdaten evaluieren? (j/n): ").lower()
    if eval_choice == 'j' and current_scaler:
        print("\nLade Standard-Inferenzdaten fuer Evaluation...")
        X_eval_infer, y_eval_infer_cat, y_eval_infer_orig, _ = load_and_preprocess_data(
            INFERENCE_EVAL_DATA_PREFIX, SEQUENCE_LENGTH, N_FEATURES, N_CLASSES, existing_scaler=current_scaler
        )
        evaluate_model_performance(model, X_eval_infer, y_eval_infer_cat, y_eval_infer_orig, dataset_name_str=f"Standard Inferenzdaten ({history_plot_name} Modell)")

    # Evaluation auf Drift-Daten, egal ob initiales Training oder Finetuning
    # Beim initialen Training sehen wir so direkt, wie gut es generalisiert auf Drift
    # Beim Finetuning sehen wir, ob das Finetuning auf Drift erfolgreich war.
    eval_drift_choice = input(f"Moechtest du das {history_plot_name.replace('_', ' ')} Modell auch auf Drift-Inferenzdaten evaluieren? (j/n): ").lower()
    if eval_drift_choice == 'j' and current_scaler:
        print("\nLade Drift-Inferenzdaten fuer Evaluation...")
        X_eval_drift, y_eval_drift_cat, y_eval_drift_orig, _ = load_and_preprocess_data(
            DRIFT_INFER_EVAL_DATA_PREFIX, SEQUENCE_LENGTH, N_FEATURES, N_CLASSES, existing_scaler=current_scaler
        )
        evaluate_model_performance(model, X_eval_drift, y_eval_drift_cat, y_eval_drift_orig, dataset_name_str=f"Drift Inferenzdaten ({history_plot_name} Modell)")


    print(f"\n{history_plot_name.capitalize().replace('_', ' ')} abgeschlossen!")
