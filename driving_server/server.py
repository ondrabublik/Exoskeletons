import socket
import numpy as np
import tensorflow as tf
import joblib
import os
import sys

# Cesta k modelu a scaleru (relativně k umístění server.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WALK_MEASUREMENT_DIR = os.path.join(os.path.dirname(BASE_DIR), "walkMeasurementStand")

SCALER_PATH = os.path.join(WALK_MEASUREMENT_DIR, "scaler.save")
# Zkusíme načíst model s různými názvy
POSSIBLE_MODEL_PATHS = [
    os.path.join(WALK_MEASUREMENT_DIR, "lstm_model_v2.h5"),
    os.path.join(WALK_MEASUREMENT_DIR, "lstm_model.h5"),
    os.path.join(WALK_MEASUREMENT_DIR, "model.h5"),
]

# UDP server parametry
HOST = "0.0.0.0"  # Naslouchá na všech rozhraních
PORT = 8888

def get_local_ip():
    """Získá lokální IP adresu počítače"""
    try:
        # Vytvoříme dočasný socket pro zjištění IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Nepřipojujeme se, jen zjistíme IP pomocí připojení k externí adrese
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback: zkusíme získat IP z hostname
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception:
            return "127.0.0.1"

def get_all_ips():
    """Získá všechny IP adresy počítače"""
    ips = []
    try:
        hostname = socket.gethostname()
        # Získáme všechny IP adresy
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            # Filtrujeme pouze IPv4 adresy (ne IPv6)
            if ':' not in ip and ip != '127.0.0.1':
                if ip not in ips:
                    ips.append(ip)
    except Exception:
        pass
    return ips

def load_model_and_scaler():
    """Načte scaler a LSTM model"""
    print(f"Načítám scaler z: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    
    # Zkusíme načíst model z různých možných cest
    model = None
    for model_path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(model_path):
            print(f"Načítám model z: {model_path}")
            model = tf.keras.models.load_model(model_path)
            break
    
    if model is None:
        raise FileNotFoundError(
            f"Model nebyl nalezen. Zkoušel jsem: {POSSIBLE_MODEL_PATHS}"
        )
    
    print("Model a scaler úspěšně načteny!")
    return scaler, model

def process_data(data_line, scaler, model):
    """
    Zpracuje jeden řádek dat a vrátí predikci
    
    Args:
        data_line: řetězec s 7-8 hodnotami oddělenými čárkami
        scaler: načtený StandardScaler
        model: načtený LSTM model
    
    Returns:
        tuple: (predikce_binární, pravděpodobnost)
    """
    try:
        # Parsování dat (očekáváme 7 hodnot, ale může být i 8)
        values = [float(x.strip()) for x in data_line.split(",")]
        
        # Vezmeme pouze prvních 7 hodnot (vstupy)
        if len(values) < 7:
            raise ValueError(f"Očekáváno alespoň 7 hodnot, obdrženo {len(values)}")
        
        X = np.array([values[:7]])  # tvar (1, 7)
        
        # Normalizace
        X_scaled = scaler.transform(X)
        
        # Úprava tvaru pro LSTM: (samples, timesteps, features) = (1, 7, 1)
        X_lstm = X_scaled.reshape(-1, 7, 1)
        
        # Predikce
        y_pred_prob = model.predict(X_lstm, verbose=0).ravel()[0]
        y_pred = 1 if y_pred_prob >= 0.5 else 0
        
        return y_pred, y_pred_prob
        
    except Exception as e:
        print(f"Chyba při zpracování dat: {e}")
        raise

def main():
    """Hlavní funkce UDP serveru"""
    # Načtení modelu a scaleru
    try:
        scaler, model = load_model_and_scaler()
    except Exception as e:
        print(f"Chyba při načítání modelu nebo scaleru: {e}")
        sys.exit(1)
    
    # Získání IP adres
    local_ip = get_local_ip()
    all_ips = get_all_ips()
    
    # Vytvoření UDP socketu
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        sock.bind((HOST, PORT))
        print("=" * 60)
        print("UDP SERVER SPUŠTĚN")
        print("=" * 60)
        print(f"Server naslouchá na: {HOST}:{PORT}")
        print(f"\nHlavní IP adresa serveru: {local_ip}")
        if all_ips:
            print(f"Všechny dostupné IP adresy:")
            for ip in all_ips:
                print(f"  - {ip}:{PORT}")
        print(f"\nPro připojení Wemos použijte IP adresu: {local_ip}")
        print("=" * 60)
        print("Čekám na data...\n")
        
        while True:
            # Příjem dat
            data, addr = sock.recvfrom(1024)  # buffer 1024 bytů
            
            try:
                # Dekódování dat
                data_str = data.decode('utf-8').strip()
                print(f"Přijato od {addr}: {data_str}")
                
                # Zpracování dat neuronovou sítí
                y_pred, y_pred_prob = process_data(data_str, scaler, model)
                
                # Vytvoření odpovědi (binární predikce a pravděpodobnost)
                response = f"{y_pred},{y_pred_prob:.6f}"
                
                # Odeslání odpovědi zpět
                sock.sendto(response.encode('utf-8'), addr)
                print(f"Odeslána odpověď: {response}")
                
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                print(f"Chyba: {error_msg}")
                # Odeslání chybové zprávy
                sock.sendto(error_msg.encode('utf-8'), addr)
                
    except KeyboardInterrupt:
        print("\nServer ukončen uživatelem")
    except Exception as e:
        print(f"Chyba serveru: {e}")
    finally:
        sock.close()
        print("Socket uzavřen")

if __name__ == "__main__":
    main()
