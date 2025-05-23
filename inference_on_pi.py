import tflite_runtime.interpreter as tflite # Speziell für Pi
import numpy as np
import os
import time
import pickle
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konfiguration ---
MODEL_DIR = "models"
DATA_DIR = "data"

# Basisnamen der Modelle (ohne .tflite)
INITIAL_TFLITE_BASE_NAME = "model"
FINETUNED_TFLITE_BASE_NAME = "model_finetuned"

SCALER_NAME = "scaler.pkl" # Der Scaler ist immer derselbe

# Datensatz-Praefixe (fuer _data.npy und _labels.npy)
STANDARD_INFERENCE_DATA_PREFIX = "infer"
DRIFT_INFERENCE_DATA_PREFIX = "drift_infer" # Neuer Datensatz für Drift-Inferenz

# Parameter (müssen mit Training/Konvertierung übereinstimmen)
SEQUENCE_LENGTH = 60
N_FEATURES = 4
N_CLASSES = 4
CLASS_NAMES = ["Normal", "Lagerschaden", "Antriebsverschleiss", "Pumpendefekt"]


def load_scaler(scaler_path_full):
    logger.info(f"Lade Scaler von: {scaler_path_full}")
    with open(scaler_path_full, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler geladen.")
    return scaler

def preprocess_inference_data(data_raw_values, scaler_instance, sequence_len, num_features):
    logger.info(f"Vorverarbeite Inferenzdaten der Form: {data_raw_values.shape}")
    if data_raw_values.shape[-1] != num_features:
        logger.error(f"Ups! Anzahl Features in Rohdaten ({data_raw_values.shape[-1]}) passt nicht zu erwarteten Features ({num_features}).")
        raise ValueError("Fehler in Feature-Anzahl bei preprocess_inference_data")

    data_flattened_values = data_raw_values.reshape(-1, num_features)
    data_scaled_flattened = scaler_instance.transform(data_flattened_values) # Wichtig: nur transform, nicht fit_transform!
    data_scaled_reshaped = data_scaled_flattened.reshape(-1, sequence_len, num_features)
    logger.info(f"Form der vorverarbeiteten Daten: {data_scaled_reshaped.shape}")
    return data_scaled_reshaped.astype(np.float32) # TFLite mag oft float32

def plot_confusion_matrix_pi(cm_array, class_name_list, file_path_full, title_prefix_str=""):
    full_plot_title = f"{title_prefix_str} Konfusionsmatrix - Pi Inferenz"
    logger.info(f"Erstelle Konfusionsmatrix: {full_plot_title}")
    df_cm_object = pd.DataFrame(cm_array, index=class_name_list, columns=[f"Vorh. {name}" for name in class_name_list])
    plt.figure(figsize=(10, 8))
    try:
        heatmap = sns.heatmap(df_cm_object, annot=True, fmt="d", cmap="viridis", annot_kws={"size": 10})
    except ValueError: # Fallback fuer ältere Seaborn-Versionen
        heatmap = sns.heatmap(df_cm_object, annot=True, fmt="d") # Einfacher
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    plt.ylabel('Wahre Klasse', fontsize=12)
    plt.xlabel('Vorhergesagte Klasse', fontsize=12)
    plt.title(full_plot_title, fontsize=14)
    plt.tight_layout()

    # Ordner für Plot erstellen, falls nicht vorhanden
    plot_dir = os.path.dirname(file_path_full)
    if not os.path.exists(plot_dir):
        try:
            os.makedirs(plot_dir)
            logger.info(f"Ordner {plot_dir} erstellt.")
        except OSError as e:
            logger.error(f"Konnte Ordner {plot_dir} nicht erstellen: {e}")

    plt.savefig(file_path_full)
    logger.info(f"Konfusionsmatrix gespeichert: {file_path_full}")
    plt.close()


if __name__ == "__main__":
    logger.info("Starte Raspberry Pi Inferenz Skript...")

    # 1. Modell auswählen
    model_choice = ""
    while model_choice not in ["1", "2"]:
        model_choice = input(f"Welches TFLite-Modell verwenden?\n1: {INITIAL_TFLITE_BASE_NAME}.tflite (Initial)\n2: {FINETUNED_TFLITE_BASE_NAME}.tflite (Fine-tuned)\nAuswahl: ")
    
    selected_tflite_base_name = INITIAL_TFLITE_BASE_NAME if model_choice == "1" else FINETUNED_TFLITE_BASE_NAME
    tflite_model_filename = f"{selected_tflite_base_name}.tflite"
    model_type_for_report = "Initial" if model_choice == "1" else "Fine-tuned"
    logger.info(f"Verwende Modell: {tflite_model_filename}")

    # 2. Datensatz auswählen
    data_choice = ""
    while data_choice not in ["1", "2"]:
        data_choice = input(f"Welchen Datensatz fuer Inferenz verwenden?\n1: Standard Inferenzdaten ({STANDARD_INFERENCE_DATA_PREFIX}_data.npy)\n2: Drift Inferenzdaten ({DRIFT_INFERENCE_DATA_PREFIX}_data.npy)\nAuswahl: ")

    selected_data_prefix = STANDARD_INFERENCE_DATA_PREFIX if data_choice == "1" else DRIFT_INFERENCE_DATA_PREFIX
    dataset_type_for_report = "Standard" if data_choice == "1" else "Drift"
    logger.info(f"Verwende Datensatz: {selected_data_prefix}_data.npy")

    # Dateinamen für Ergebnisse und Plot anpassen
    results_file_suffix = f"{selected_tflite_base_name}_on_{selected_data_prefix}"
    results_file_name = f"inference_results_pi_{results_file_suffix}.txt"
    cm_plot_file_name = f"confusion_matrix_pi_{results_file_suffix}.png"


    model_full_path = os.path.join(MODEL_DIR, tflite_model_filename)
    if not os.path.exists(model_full_path):
        logger.error(f"FEHLER: TFLite-Modell nicht gefunden: {model_full_path}")
        exit(1)

    try:
        # num_threads kann auf dem Pi angepasst werden (z.B. Anzahl der CPU Kerne)
        interpreter = tflite.Interpreter(model_path=model_full_path, num_threads=4)
        interpreter.allocate_tensors() # Wichtig!
        logger.info("TFLite-Modell geladen und Tensoren zugewiesen.")
    except Exception as e:
        logger.error(f"FEHLER beim Laden des TFLite-Modells: {e}")
        exit(1)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"Eingabedetails des Modells: {input_details}")
    logger.info(f"Ausgabedetails des Modells: {output_details}")

    scaler_full_path = os.path.join(MODEL_DIR, SCALER_NAME)
    if not os.path.exists(scaler_full_path):
        logger.error(f"FEHLER: Scaler-Datei nicht gefunden: {scaler_full_path}")
        exit(1)
    scaler_instance = load_scaler(scaler_full_path)

    # Inferenzdaten und Labels laden
    inference_data_full_path = os.path.join(DATA_DIR, f"{selected_data_prefix}_data.npy")
    inference_labels_full_path = os.path.join(DATA_DIR, f"{selected_data_prefix}_labels.npy")

    if not os.path.exists(inference_data_full_path) or not os.path.exists(inference_labels_full_path):
        logger.error(f"FEHLER: Inferenzdaten ({selected_data_prefix}_data.npy) oder Labels nicht im Ordner {DATA_DIR} gefunden.")
        exit(1)

    X_inference_raw_values = np.load(inference_data_full_path)
    y_inference_true_labels = np.load(inference_labels_full_path) # Originale Integer-Labels
    logger.info(f"Inferenzdaten geladen: Rohform {X_inference_raw_values.shape}, Labelform {y_inference_true_labels.shape}")

    X_inference_processed = preprocess_inference_data(X_inference_raw_values, scaler_instance, SEQUENCE_LENGTH, N_FEATURES)

    num_samples_to_infer = X_inference_processed.shape[0]
    y_pred_probabilities_all = np.zeros((num_samples_to_infer, N_CLASSES), dtype=np.float32)
    y_pred_labels_all = np.zeros(num_samples_to_infer, dtype=np.int32)
    inference_times_list = []

    logger.info(f"Führe Inferenz für {num_samples_to_infer} Proben durch...")
    for i in range(num_samples_to_infer):
        # Eingabedaten für das Modell vorbereiten (oft Batch-Dimension hinzufügen)
        current_input_data = np.expand_dims(X_inference_processed[i], axis=0)

        # Form überpruefen (sollte durch expand_dims passen, wenn Modell Batch=1 erwartet)
        expected_input_shape = tuple(input_details[0]['shape'])
        if current_input_data.shape != expected_input_shape:
            logger.warning(f"Huch! Eingabeform {current_input_data.shape} passt nicht zu Modell-Input {expected_input_shape}.")
            # Hier keine automatische Anpassung, Datenaufbereitung sollte stimmen.
        
        # Datentyp-Check
        expected_input_dtype = input_details[0]['dtype']
        if current_input_data.dtype != expected_input_dtype:
            logger.warning(f"Datentyp der Eingabe ({current_input_data.dtype}) weicht von Erwartung ({expected_input_dtype}) ab. Konvertiere...")
            current_input_data = current_input_data.astype(expected_input_dtype)

        interpreter.set_tensor(input_details[0]['index'], current_input_data)
        start_time = time.perf_counter() # Genauer als time.time() für kurze Dauern
        interpreter.invoke() # Inferenz starten
        end_time = time.perf_counter()
        inference_times_list.append(end_time - start_time)

        output_from_model = interpreter.get_tensor(output_details[0]['index'])
        y_pred_probabilities_all[i] = output_from_model[0] # Annahme: Ausgabe ist (1, N_CLASSES)
        y_pred_labels_all[i] = np.argmax(output_from_model[0])

        if (i + 1) % 100 == 0 or (i+1) == num_samples_to_infer : # Alle 100 Samples mal was ausgeben
            logger.info(f"Verarbeitet {i+1}/{num_samples_to_infer} Proben...")

    avg_inference_time_ms = (sum(inference_times_list) / num_samples_to_infer) * 1000
    logger.info("Inferenz fertig.")
    logger.info(f"Durchschnittliche Inferenzzeit: {avg_inference_time_ms:.3f} ms pro Probe")

    # --- Metriken berechnen ---
    logger.info("Berechne Leistungsmetriken...")
    accuracy_val = accuracy_score(y_inference_true_labels, y_pred_labels_all)
    f1_macro_val = f1_score(y_inference_true_labels, y_pred_labels_all, average='macro', zero_division=0)
    f1_weighted_val = f1_score(y_inference_true_labels, y_pred_labels_all, average='weighted', zero_division=0)

    roc_auc_ovr_macro_val = "N/A" # Standardwert
    try:
        y_inference_true_binarized = label_binarize(y_inference_true_labels, classes=list(range(N_CLASSES)))
        if y_inference_true_binarized.shape[1] == N_CLASSES and N_CLASSES > 1:
             roc_auc_ovr_macro_val = roc_auc_score(y_inference_true_binarized, y_pred_probabilities_all, multi_class='ovr', average='macro')
    except ValueError as e:
        logger.warning(f"Konnte ROC-AUC (multi_class='ovr') nicht berechnen: {e}. Evtl. nicht alle Klassen in y_true oder y_pred_probs.")

    logger.info(f"Genauigkeit: {accuracy_val:.4f}")
    logger.info(f"F1-Score (Macro): {f1_macro_val:.4f}")
    logger.info(f"F1-Score (Weighted): {f1_weighted_val:.4f}")
    if isinstance(roc_auc_ovr_macro_val, float): # Nur ausgeben, wenn Zahl
        logger.info(f"ROC-AUC Score (Macro OVR): {roc_auc_ovr_macro_val:.4f}")
    else:
        logger.info(f"ROC-AUC Score (Macro OVR): {roc_auc_ovr_macro_val}")

    logger.info("Erstelle Klassifikationsbericht...")
    classification_report_str = classification_report(y_inference_true_labels, y_pred_labels_all, target_names=CLASS_NAMES, zero_division=0)
    print("\nKlassifikationsbericht:") # Direkte Ausgabe für Konsole
    print(classification_report_str)

    cm_array_values = confusion_matrix(y_inference_true_labels, y_pred_labels_all, labels=list(range(N_CLASSES)))
    cm_full_path = os.path.join(MODEL_DIR, cm_plot_file_name)
    plot_title = f"Modell: {model_type_for_report} auf Daten: {dataset_type_for_report}"
    plot_confusion_matrix_pi(cm_array_values, CLASS_NAMES, cm_full_path, title_prefix_str=plot_title)

    # --- Ergebnisse speichern ---
    results_summary_dict = {
        "model_name_used": tflite_model_filename,
        "dataset_name_used": f"{selected_data_prefix}_data.npy",
        "model_type_evaluated_on_pi": model_type_for_report,
        "dataset_type_evaluated_on_pi": dataset_type_for_report,
        "average_inference_time_ms": avg_inference_time_ms,
        "accuracy": accuracy_val,
        "f1_macro": f1_macro_val,
        "f1_weighted": f1_weighted_val,
        "roc_auc_macro_ovr": roc_auc_ovr_macro_val if isinstance(roc_auc_ovr_macro_val, float) else "N/A",
        "classification_report_string": classification_report_str,
        "confusion_matrix_as_list": cm_array_values.tolist()
    }

    results_full_path = os.path.join(MODEL_DIR, results_file_name)
    logger.info(f"Speichere Inferenzergebnisse nach: {results_full_path}")
    try:
        with open(results_full_path, 'w') as f:
            f.write(f"Raspberry Pi Inferenz Ergebnisse:\n")
            f.write("----------------------------------\n")
            for key, value in results_summary_dict.items():
                if key == "confusion_matrix_as_list":
                    f.write(f"\nKonfusionsmatrix (Werte):\n")
                    df_cm_results = pd.DataFrame(value, index=CLASS_NAMES, columns=[f"Vorh. {name}" for name in CLASS_NAMES])
                    f.write(df_cm_results.to_string())
                    f.write("\n")
                elif key == "classification_report_string":
                    f.write(f"\nKlassifikationsbericht:\n{value}\n")
                else:
                    f.write(f"{key}: {value}\n")
        logger.info("Inferenzergebnisse erfolgreich gespeichert.")
    except IOError as e:
        logger.error(f"FEHLER beim Schreiben der Ergebnisdatei: {e}")

    logger.info("\n--- Zur Interpretation der Ergebnisse ---")
    logger.info(f"Modell '{model_type_for_report}' wurde auf '{dataset_type_for_report}' Daten getestet.")
    logger.info("Vergleiche die Metriken (Genauigkeit, F1) und die Konfusionsmatrix.")
    logger.info("Besonders interessant ist der Vergleich:")
    logger.info("  - Initiales Modell auf Standard-Daten (Baseline)")
    logger.info("  - Initiales Modell auf Drift-Daten (zeigt, wie Drift die Leistung beeinflusst)")
    logger.info("  - Fine-getuntes Modell auf Drift-Daten (zeigt, ob Fine-Tuning geholfen hat)")
    logger.info("  - Fine-getuntes Modell auf Standard-Daten (zeigt, ob das Modell noch generalisieren kann -> 'Catastrophic Forgetting')")
