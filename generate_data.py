import numpy as np
import os

# --- Konfiguration ---
N_SAMPLES_TRAIN = 5000      # Kleiner Datensatz
N_SAMPLES_INFERENCE = 1000
N_SAMPLES_FINETUNE_DRIFT = 2000 # Daten für Fine-Tuning mit Drift
N_SAMPLES_INFERENCE_DRIFT = 800  # Neuer Datensatz für Inferenz mit Drift-Annahme
SEQUENCE_LENGTH = 60
N_FEATURES = 4              # Temperatur, Vibration, Druck, Leistung

# --- Hilfsfunktionen zur Datengenerierung ---

def generate_normal_operation(n_samples, seq_length):
    data = np.zeros((n_samples, seq_length, N_FEATURES))
    for i in range(n_samples):
        # Temperatur: 40-60°C
        data[i, :, 0] = np.random.normal(loc=50, scale=5, size=seq_length)
        # Vibration: 0.1-0.5 mm/s
        data[i, :, 1] = np.random.normal(loc=0.3, scale=0.1, size=seq_length)
        # Druck: 10-12 bar
        data[i, :, 2] = np.random.normal(loc=11, scale=0.5, size=seq_length)
        # Leistung: 100-150 W
        data[i, :, 3] = np.random.normal(loc=125, scale=10, size=seq_length)
    labels = np.full(n_samples, 0) # 0 fuer Normal
    return data, labels

def generate_defective_bearing(n_samples, seq_length):
    data = np.zeros((n_samples, seq_length, N_FEATURES))
    for i in range(n_samples):
        data[i, :, 0] = np.random.normal(loc=65, scale=8, size=seq_length) # Temp höher
        data[i, :, 1] = np.random.normal(loc=1.0, scale=0.3, size=seq_length) + \
                        np.sin(np.linspace(0, np.random.uniform(5,10) * np.pi, seq_length)) * np.random.uniform(0.1, 0.3) # Vib hoeher
        data[i, :, 2] = np.random.normal(loc=11, scale=0.5, size=seq_length) # Druck normal
        data[i, :, 3] = np.random.normal(loc=140, scale=12, size=seq_length) # Leistung leicht höher
    labels = np.full(n_samples, 1) # 1 für Lagerschaden
    return data, labels

def generate_wear_drive_elements(n_samples, seq_length):
    data = np.zeros((n_samples, seq_length, N_FEATURES))
    for i in range(n_samples):
        data[i, :, 0] = np.random.normal(loc=55, scale=6, size=seq_length) # Temp leicht höher
        data[i, :, 1] = np.random.normal(loc=0.5, scale=0.15, size=seq_length) + \
                        np.random.normal(loc=0, scale=0.08, size=seq_length) # Vib anders
        data[i, :, 2] = np.random.normal(loc=11, scale=0.5, size=seq_length) # Druck normal
        data[i, :, 3] = np.random.normal(loc=150, scale=15, size=seq_length) # Leistung erhöht
    labels = np.full(n_samples, 2) # 2 für Antriebsverschleiss
    return data, labels

def generate_pump_defect(n_samples, seq_length):
    data = np.zeros((n_samples, seq_length, N_FEATURES))
    for i in range(n_samples):
        data[i, :, 0] = np.random.normal(loc=55, scale=10, size=seq_length) # Temp variabel
        data[i, :, 1] = np.random.normal(loc=0.35, scale=0.15, size=seq_length) # Vib variabel
        data[i, :, 2] = np.random.normal(loc=7, scale=1, size=seq_length) # Druckabfall!!
        data[i, :, 3] = np.random.normal(loc=110, scale=20, size=seq_length) # Leistung variabel
    labels = np.full(n_samples, 3) # 3 für Pumpendefekt
    return data, labels

def generate_dataset(n_total_samples, seq_length, dataset_type="train", base_params=None):
    # dataset_type ist hier eher zur Info beim Printen
    n_per_class = n_total_samples // 4 # Grob gleiche Anzahl pro Klasse
    data_list = []
    labels_list = []

    # Parameter für leichten Drift im Inferenzdatensatz (optional)
    params_normal = {"loc_temp": 50, "loc_vib": 0.3, "loc_press": 11, "loc_power": 125}
    params_bearing = {"loc_temp": 65, "loc_vib": 1.0, "loc_power": 140}
    params_wear = {"loc_temp": 55, "loc_vib": 0.5, "loc_power": 150}
    params_pump = {"loc_press": 7, "loc_power": 110}

    if base_params: # Wenn spezielle Parameter übergeben werden (für infer_data z.B.)
        params_normal.update(base_params.get("normal", {}))
        params_bearing.update(base_params.get("bearing", {}))
        params_wear.update(base_params.get("wear", {}))
        params_pump.update(base_params.get("pump", {}))

    # Normal
    d_normal, l_normal = generate_normal_operation(n_per_class, seq_length)

    for i in range(n_per_class): # Überschreibe mit potenziell geänderten Parametern
        d_normal[i, :, 0] = np.random.normal(loc=params_normal["loc_temp"], scale=5, size=seq_length)
        d_normal[i, :, 1] = np.random.normal(loc=params_normal["loc_vib"], scale=0.1, size=seq_length)
        d_normal[i, :, 2] = np.random.normal(loc=params_normal["loc_press"], scale=0.5, size=seq_length)
        d_normal[i, :, 3] = np.random.normal(loc=params_normal["loc_power"], scale=10, size=seq_length)
    data_list.append(d_normal)
    labels_list.append(l_normal)

    # Defektes Lager
    d_bearing, l_bearing = generate_defective_bearing(n_per_class, seq_length)
    for i in range(n_per_class):
        d_bearing[i, :, 0] = np.random.normal(loc=params_bearing["loc_temp"], scale=8, size=seq_length)
        d_bearing[i, :, 1] = np.random.normal(loc=params_bearing["loc_vib"], scale=0.3, size=seq_length) + \
                             np.sin(np.linspace(0, np.random.uniform(4,9) * np.pi, seq_length)) * np.random.uniform(0.1, 0.25)
        d_bearing[i, :, 3] = np.random.normal(loc=params_bearing["loc_power"], scale=12, size=seq_length)
    data_list.append(d_bearing)
    labels_list.append(l_bearing)

    # Verschleiss Antrieb
    d_wear, l_wear = generate_wear_drive_elements(n_per_class, seq_length)
    for i in range(n_per_class):
        d_wear[i, :, 0] = np.random.normal(loc=params_wear["loc_temp"], scale=6, size=seq_length)
        d_wear[i, :, 1] = np.random.normal(loc=params_wear["loc_vib"], scale=0.15, size=seq_length) + \
                          np.random.normal(loc=0, scale=0.07, size=seq_length)
        d_wear[i, :, 3] = np.random.normal(loc=params_wear["loc_power"], scale=15, size=seq_length)
    data_list.append(d_wear)
    labels_list.append(l_wear)

    # Pumpendefekt (restliche Samples, um auf n_total_samples zu kommen)
    remaining_samples = n_total_samples - (3 * n_per_class)
    d_pump, l_pump = generate_pump_defect(remaining_samples, seq_length)
    for i in range(remaining_samples):
        d_pump[i, :, 2] = np.random.normal(loc=params_pump["loc_press"], scale=1, size=seq_length)
        d_pump[i, :, 3] = np.random.normal(loc=params_pump["loc_power"], scale=20, size=seq_length)
    data_list.append(d_pump)
    labels_list.append(l_pump)

    all_data = np.concatenate(data_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # Daten mischen
    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    all_data = all_data[indices]
    all_labels = all_labels[indices]

    print(f"Generierter '{dataset_type}' Datensatz: Datenform {all_data.shape}, Labelform {all_labels.shape}")
    return all_data, all_labels

def generate_drifted_data(n_samples, seq_length, is_for_finetuning=True):
    # Diese Funktion erzeugt Daten mit deutlichem Drift.
    n_per_class = n_samples // 4
    data_list = []
    labels_list = []

    # Normalbetrieb (Label 0) - Basiswerte etwas höher
    d_normal_drift = np.zeros((n_per_class, seq_length, N_FEATURES))
    for i in range(n_per_class):
        d_normal_drift[i, :, 0] = np.random.normal(loc=58, scale=5, size=seq_length) # Temp etwas hoeher
        d_normal_drift[i, :, 1] = np.random.normal(loc=0.4, scale=0.1, size=seq_length) # Vib etwas hoeher
        d_normal_drift[i, :, 2] = np.random.normal(loc=11.8, scale=0.4, size=seq_length)
        d_normal_drift[i, :, 3] = np.random.normal(loc=135, scale=8, size=seq_length)
    labels_list.append(np.full(n_per_class, 0))
    data_list.append(d_normal_drift)

    # Defektes Lager (Label 1) - bekommt Charakteristiken von Pumpendefekt (niedriger Druck)
    d_bearing_drift = np.zeros((n_per_class, seq_length, N_FEATURES))
    for i in range(n_per_class):
        d_bearing_drift[i, :, 0] = np.random.normal(loc=70, scale=7, size=seq_length)
        d_bearing_drift[i, :, 1] = np.random.normal(loc=1.2, scale=0.25, size=seq_length) + \
                                   np.sin(np.linspace(0, np.random.uniform(6,11) * np.pi, seq_length)) * np.random.uniform(0.15, 0.35)
        d_bearing_drift[i, :, 2] = np.random.normal(loc=8.0, scale=1.0, size=seq_length) # !!! DRIFT: Niedriger Druck !!!
        d_bearing_drift[i, :, 3] = np.random.normal(loc=150, scale=18, size=seq_length)
    labels_list.append(np.full(n_per_class, 1))
    data_list.append(d_bearing_drift)

    # Verschleiss Antrieb (Label 2) - aggressiver Drift (höhere Vib & Temp)
    d_wear_drift = np.zeros((n_per_class, seq_length, N_FEATURES))
    for i in range(n_per_class):
        d_wear_drift[i, :, 0] = np.random.normal(loc=75, scale=7, size=seq_length) # !!! DRIFT: Deutlich hoehere Temp !!!
        d_wear_drift[i, :, 1] = np.random.normal(loc=0.9, scale=0.2, size=seq_length) + \
                                 np.random.normal(loc=0, scale=0.1, size=seq_length) # !!! DRIFT: Staerkere Vib !!!
        d_wear_drift[i, :, 2] = np.random.normal(loc=10.5, scale=0.6, size=seq_length)
        d_wear_drift[i, :, 3] = np.random.normal(loc=170, scale=20, size=seq_length)
    labels_list.append(np.full(n_per_class, 2))
    data_list.append(d_wear_drift)

    # Pumpendefekt (Label 3) - noch ausgeprägter (sehr niedriger Druck)
    remaining_samples = n_samples - (3 * n_per_class)
    d_pump_drift = np.zeros((remaining_samples, seq_length, N_FEATURES))
    for i in range(remaining_samples):
        d_pump_drift[i, :, 0] = np.random.normal(loc=60, scale=12, size=seq_length)
        d_pump_drift[i, :, 1] = np.random.normal(loc=0.45, scale=0.25, size=seq_length)
        d_pump_drift[i, :, 2] = np.random.normal(loc=5.5, scale=0.8, size=seq_length) # !!! DRIFT: Sehr niedriger Druck !!!
        d_pump_drift[i, :, 3] = np.random.normal(loc=90, scale=28, size=seq_length)
    labels_list.append(np.full(remaining_samples, 3))
    data_list.append(d_pump_drift)

    all_data = np.concatenate(data_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    all_data = all_data[indices]
    all_labels = all_labels[indices]

    data_purpose = "Finetuning" if is_for_finetuning else "Drift-Inferenz"
    print(f"Generierter '{data_purpose}' Datensatz: Datenform {all_data.shape}, Labelform {all_labels.shape}")
    return all_data, all_labels


def save_data(data, labels, folder_path, file_prefix):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(os.path.join(folder_path, f"{file_prefix}_data.npy"), data)
    np.save(os.path.join(folder_path, f"{file_prefix}_labels.npy"), labels)
    print(f"Gespeichert: {file_prefix}_data.npy und {file_prefix}_labels.npy in {folder_path}")

if __name__ == "__main__":
    data_dir = "data" # Ordner für die Daten

    # 1. Initialen Trainingsdatensatz generieren
    train_data, train_labels = generate_dataset(N_SAMPLES_TRAIN, SEQUENCE_LENGTH, "train")
    save_data(train_data, train_labels, data_dir, "train") # Wird zu train_data.npy, train_labels.npy

    # 2. Standard Inferenzdatensatz generieren (mit leichten Unterschieden zum Training)
    inference_params = { # leichte Abweichungen für den Inferenzdatensatz
        "normal": {"loc_temp": 52, "loc_vib": 0.32},
        "wear": {"loc_temp": 57, "loc_vib": 0.55},
        "pump": {"loc_press": 6.8}
    }
    infer_data, infer_labels = generate_dataset(N_SAMPLES_INFERENCE, SEQUENCE_LENGTH, "infer", base_params=inference_params)
    save_data(infer_data, infer_labels, data_dir, "infer") # Wird zu infer_data.npy, infer_labels.npy

    # 3. Datensatz für Fine-Tuning generieren (mit deutlichem Drift)
    drift_train_data, drift_train_labels = generate_drifted_data(N_SAMPLES_FINETUNE_DRIFT, SEQUENCE_LENGTH, is_for_finetuning=True)
    save_data(drift_train_data, drift_train_labels, data_dir, "drift_train") # Wird zu drift_train_data.npy, drift_train_labels.npy

    # 4. Datensatz für Drift-Inferenz generieren (Daten, die aehnlichen Drift wie drift_train aufweisen)
    drift_infer_data, drift_infer_labels = generate_drifted_data(N_SAMPLES_INFERENCE_DRIFT, SEQUENCE_LENGTH, is_for_finetuning=False)
    save_data(drift_infer_data, drift_infer_labels, data_dir, "drift_infer") # Wird zu drift_infer_data.npy, drift_infer_labels.npy

    print("\nAlle Datensaetze generiert und gespeichert.")
    print(f"Daten liegen in: {os.path.abspath(data_dir)}")
