import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pickle
import logging

# --- Konfiguration ---
MODEL_DIR = "models"
DATA_DIR = "data" # Für repraesentativen Datensatz

# Mögliche Keras-Modellnamen (ohne ".keras")
INITIAL_KERAS_BASE_NAME = "model"
FINETUNED_KERAS_BASE_NAME = "model_finetuned"
SCALER_NAME = "scaler.pkl" # Bleibt gleich

# Repraesentativer Datensatz wird von den initialen Trainingsdaten genommen
REP_DATA_SOURCE_PREFIX = "train" # -> train_data.npy

SEQUENCE_LENGTH = 60
N_FEATURES = 4

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_model_to_tflite(keras_model_path, tflite_model_path, sequence_length, n_features, scaler_for_rep_data_path=None):
    logger.info(f"TensorFlow Version: {tf.__version__}, Keras Version: {keras.__version__}")

    if not os.path.exists(keras_model_path):
        logger.error(f"FEHLER: Keras-Modell nicht da: {keras_model_path}")
        return False

    try:
        logger.info(f"Lade Keras-Modell von: {keras_model_path}")
        # compile=False, da der Optimizer-Status für die Inferenz-Konvertierung nicht gebraucht wird
        model = keras.models.load_model(keras_model_path, compile=False)
        logger.info("Keras-Modell geladen.")
        model.summary(print_fn=logger.info) # Zeigt die Modellstruktur
    except Exception as e:
        logger.error(f"Fehler beim Laden des Keras-Modells: {e}")
        return False

    try:
        logger.info("Mache eine 'Concrete Function' fuer die Konvertierung...")
        run_model = tf.function(lambda x: model(x))
        # Feste Input-Signatur. 
        input_signature = tf.TensorSpec(shape=[1, sequence_length, n_features], dtype=model.inputs[0].dtype) # dtype vom Modell nehmen
        concrete_func = run_model.get_concrete_function(input_signature)
        logger.info(f"Concrete Function mit Input-Signatur {input_signature} erstellt.")
    except Exception as e:
        logger.error(f"Fehler bei Concrete Function: {e}")
        return False

# TFLiteConverter von concret_funktions statt from keras um Kompatibilitätsprobleme zu vermeiden
    logger.info("Initialisiere TFLiteConverter.")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    logger.info("Setze Konvertierungsoptionen (Standard Optimierungen).")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS] # Nur TFLite Ops, kein Flex Delegate nötig

    # (Optional) Repraesentativer Datensatz für bessere Quantisierung
    # Hier verwenden wir es für die DEFAULT Optimierung, was auch Float16 Quantisierung einschliessen kann.
    if scaler_for_rep_data_path and os.path.exists(scaler_for_rep_data_path):
        rep_data_raw_path = os.path.join(DATA_DIR, f"{REP_DATA_SOURCE_PREFIX}_data.npy")
        if os.path.exists(rep_data_raw_path):
            try:
                logger.info(f"Lade Scaler von {scaler_for_rep_data_path} fuer rep. Datensatz.")
                with open(scaler_for_rep_data_path, 'rb') as f:
                    scaler = pickle.load(f)

                logger.info(f"Lade Rohdaten fuer rep. Datensatz von {rep_data_raw_path} (ca. 300 Samples).")
                X_rep_raw = np.load(rep_data_raw_path)[:300]

                logger.info("Verarbeite rep. Datensatz (Skalierung).")
                X_rep_flattened = X_rep_raw.reshape(-1, n_features)
                X_rep_scaled_flattened = scaler.transform(X_rep_flattened)
                X_rep_scaled = X_rep_scaled_flattened.reshape(-1, sequence_length, n_features)
                X_rep_scaled = X_rep_scaled.astype(np.float32) # Korrekter Datentyp

                def representative_dataset_gen():
                    for i in range(X_rep_scaled.shape[0]):
                        yield [X_rep_scaled[i:i+1]] # Muss eine Liste von Tensoren sein

                converter.representative_dataset = representative_dataset_gen
                logger.info("Repraesentativer Datensatz fuer Konverter vorbereitet.")

            except Exception as e:
                logger.warning(f"Warnung: Rep. Datensatz konnte nicht erstellt werden: {e}. Mache ohne weiter.")
        else:
            logger.warning(f"Quelldatei fuer rep. Datensatz nicht gefunden: {rep_data_raw_path}")
    else:
        logger.info("Kein Scaler-Pfad fuer rep. Datensatz angegeben oder Datei nicht gefunden. Konvertierung ohne.")

    logger.info("Starte Konvertierung zu TFLite...")
    try:
        tflite_model_content = converter.convert()
        logger.info("Modell zu TFLite konvertiert.")
    except Exception as e:
        logger.error(f"FEHLER bei Konvertierung: {e}")
        import traceback
        traceback.print_exc() # Mehr Details zum Fehler
        return False

    try:
        logger.info(f"Speichere TFLite-Modell nach: {tflite_model_path}")
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model_content)
        logger.info(f"TFLite-Modell gespeichert. Groesse: {os.path.getsize(tflite_model_path) / 1024:.2f} KB")
    except Exception as e:
        logger.error(f"FEHLER beim Speichern der TFLite-Datei: {e}")
        return False

    # Kurzer Test des TFLite Modells
    try:
        logger.info("Teste das konvertierte TFLite-Modell kurz...")
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info(f"TFLite Input: {input_details}")
        logger.info(f"TFLite Output: {output_details}")

        # Dummy Input, muss evtl. skalieren
        dummy_input_data = np.random.rand(1, sequence_length, n_features).astype(np.float32)
        if scaler_for_rep_data_path and os.path.exists(scaler_for_rep_data_path):
             with open(scaler_for_rep_data_path, 'rb') as f:
                scaler_for_dummy = pickle.load(f)
             dummy_input_flat = dummy_input_data.reshape(-1, n_features)
             dummy_input_scaled_flat = scaler_for_dummy.transform(dummy_input_flat)
             dummy_input_data = dummy_input_scaled_flat.reshape(1, sequence_length, n_features).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], dummy_input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        logger.info(f"TFLite Dummy-Inferenz Output Form: {output_data.shape}. Test OK.")
    except Exception as e:
        logger.error(f"Fehler beim Testen des TFLite-Modells: {e}")
    return True

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    choice = ""
    while choice not in ["1", "2"]:
        choice = input(f"Welches Keras-Modell konvertieren?\n1: {INITIAL_KERAS_BASE_NAME}.keras (Initial)\n2: {FINETUNED_KERAS_BASE_NAME}.keras (Fine-tuned)\nAuswahl: ")

    keras_base_name = ""
    if choice == "1":
        keras_base_name = INITIAL_KERAS_BASE_NAME
    else:
        keras_base_name = FINETUNED_KERAS_BASE_NAME

    keras_model_full_path = os.path.join(MODEL_DIR, f"{keras_base_name}.keras")
    tflite_model_full_path = os.path.join(MODEL_DIR, f"{keras_base_name}.tflite") # TFLite kriegt gleichen Basisnamen
    scaler_full_path = os.path.join(MODEL_DIR, SCALER_NAME)

    logger.info(f"Konvertiere: {keras_model_full_path}  ->  {tflite_model_full_path}")
    if os.path.exists(scaler_full_path):
        logger.info(f"Verwende Scaler (fuer rep. Daten): {scaler_full_path}")
    else:
        logger.warning(f"Scaler {scaler_full_path} nicht gefunden. Rep. Daten werden nicht optimal sein.")
        scaler_full_path = None # Sicherstellen, dass es nicht verwendet wird, wenn nicht da

    success = convert_model_to_tflite(
        keras_model_full_path,
        tflite_model_full_path,
        SEQUENCE_LENGTH,
        N_FEATURES,
        scaler_full_path
    )

    if success:
        logger.info("Konvertierungsskript erfolgreich durchgelaufen.")
    else:
        logger.error("Konvertierungsskript mit Fehlern beendet.")
