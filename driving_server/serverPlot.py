import socket
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Použití non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import threading
import queue
import time
from flask import Flask, render_template_string, Response

# UDP server parametry
HOST = "0.0.0.0"  # Naslouchá na všech rozhraních
PORT = 8888  # UDP port

# Flask server parametry
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000  # HTTP port

# Parametry pro grafy
MAX_POINTS = 100  # Maximální počet bodů na grafu
NUM_SENSORS = 7  # Počet senzorů (hodnot)

# Fronta pro data mezi vlákny
data_queue = queue.Queue()

# Buffer pro data
data_buffer = []
data_lock = threading.Lock()  # Zámek pro synchronizaci přístupu k bufferu

# Událost pro SSE
update_event = threading.Event()

# Flask app
app = Flask(__name__)

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


def parse_sensor_packet(data):
    if isinstance(data, (bytes, bytearray)) and len(data) == 7 * 4:
        try:
            return list(struct.unpack('<7f', data))
        except struct.error:
            raise ValueError("Binární packet nelze rozparsovat")

    try:
        text = data.decode('utf-8').strip()
        values = [float(x.strip()) for x in text.split(',') if x.strip() != '']
        if len(values) >= 7:
            return values[:7]
        raise ValueError(f"Nedostatečný počet hodnot ve CSV: {len(values)}")
    except Exception as e:
        raise ValueError(f"Neplatný packet: {e}")


def udp_listener():
    """Funkce pro naslouchání UDP paketům"""
    # Vytvoření UDP socketu
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        sock.bind((HOST, PORT))
        print("=" * 60)
        print("UDP PLOT SERVER SPUŠTĚN")
        print("=" * 60)
        print(f"UDP Server naslouchá na: {HOST}:{PORT}")
        print(f"Web Server běží na: http://{FLASK_HOST}:{FLASK_PORT}")
        
        # Získání IP adres
        local_ip = get_local_ip()
        all_ips = get_all_ips()
        print(f"\nHlavní IP adresa serveru: {local_ip}")
        if all_ips:
            print(f"Všechny dostupné IP adresy:")
            for ip in all_ips:
                print(f"  - UDP: {ip}:{PORT}")
                print(f"  - Web: http://{ip}:{FLASK_PORT}")
        print(f"\nPro připojení Wemos použijte UDP IP adresu: {local_ip}:{PORT}")
        print(f"Pro zobrazení grafů otevřete: http://{local_ip}:{FLASK_PORT}")
        print("=" * 60)
        print("Čekám na data...\n")
        
        while True:
            # Příjem dat
            data, addr = sock.recvfrom(1024)  # buffer 1024 bytů
            
            try:
                values = parse_sensor_packet(data)
                print(f"Přijato od {addr}: {values}")

                sensor_data = values[:7]
                # Přidání timestamp
                timestamp = time.time()
                data_queue.put((timestamp, sensor_data))

                # Signalizace nových dat pro SSE
                update_event.set()

                # Odeslání odpovědi pro stav LED (druhá hodnota = náklon)
                pred = 0
                if len(values) >= 1 and abs(values[1]) > 20:
                    pred = 1
                response = bytes([pred])
                sock.sendto(response, addr)
                print(f"Odeslána odpověď: {pred}")
                    
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                print(f"Chyba: {error_msg}")
                # Odeslání chybové zprávy
                sock.sendto(error_msg.encode('utf-8'), addr)
                
    except KeyboardInterrupt:
        print("\nUDP Server ukončen uživatelem")
    except Exception as e:
        print(f"Chyba UDP serveru: {e}")
    finally:
        sock.close()
        print("UDP Socket uzavřen")

def generate_plot():
    """Generuje base64 kódovaný graf"""
    global data_buffer
    
    with data_lock:
        # Získání nových dat z fronty
        while not data_queue.empty():
            timestamp, sensor_data = data_queue.get()
            data_buffer.append((timestamp, sensor_data))
            
            # Omezení velikosti bufferu
            if len(data_buffer) > MAX_POINTS:
                data_buffer.pop(0)
    
    if not data_buffer:
        # Prázdný graf
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Čekám na data...', transform=ax.transAxes, ha='center', va='center', fontsize=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Příprava dat pro graf
        timestamps = [item[0] for item in data_buffer]
        sensor_values = np.array([item[1] for item in data_buffer])
        
        # Vytvoření grafu
        fig, axes = plt.subplots(NUM_SENSORS, 1, figsize=(12, 10))
        if NUM_SENSORS == 1:
            axes = [axes]  # Zajistíme, že axes je vždy seznam
        
        sensor_names = [f'Senzor {i+1}' for i in range(NUM_SENSORS)]
        
        for i in range(NUM_SENSORS):
            axes[i].plot(timestamps, sensor_values[:, i], 'b-', linewidth=2, marker='o', markersize=3)
            axes[i].set_title(f'{sensor_names[i]} - Reálný čas')
            axes[i].set_xlabel('Čas (s)')
            axes[i].set_ylabel('Hodnota')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(min(timestamps), max(timestamps))
        
        plt.tight_layout()
    
    # Převedení na base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64

def event_stream():
    """Server-Sent Events generátor"""
    while True:
        # Čekání na nová data
        update_event.wait()
        update_event.clear()
        
        # Generování nového grafu
        plot_image = generate_plot()
        
        # Odeslání dat klientovi
        yield f"data: {plot_image}\n\n"

@app.route('/stream')
def stream():
    """SSE endpoint"""
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/')
def index():
    """Hlavní stránka s grafem"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-time Sensor Data Plot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            h1 { color: #333; text-align: center; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            img { max-width: 100%; height: auto; }
            .status { text-align: center; margin: 20px 0; padding: 10px; background: #e8f5e8; border: 1px solid #4caf50; border-radius: 4px; }
            .loading { color: #666; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Real-time Sensor Data Visualization</h1>
            <div class="status">
                <strong>Status:</strong> Server běží | UDP Port: {{ udp_port }} | Poslední aktualizace: <span id="timestamp"></span>
            </div>
            <div id="plot-container">
                <p class="loading">Načítání grafu...</p>
            </div>
        </div>
        <script>
            const plotContainer = document.getElementById('plot-container');
            const timestampElement = document.getElementById('timestamp');
            
            // Připojení k Server-Sent Events
            const eventSource = new EventSource('/stream');
            
            eventSource.onmessage = function(event) {
                const plotImage = event.data;
                plotContainer.innerHTML = `<img src="data:image/png;base64,${plotImage}" alt="Sensor Data Plot">`;
                timestampElement.textContent = new Date().toLocaleString();
            };
            
            eventSource.onerror = function(event) {
                console.error('SSE chyba:', event);
                plotContainer.innerHTML = '<p style="color: red;">Chyba připojení k serveru</p>';
            };
            
            // Inicializace času
            timestampElement.textContent = new Date().toLocaleString();
        </script>
    </body>
    </html>
    """
    plot_image = generate_plot()
    return render_template_string(html_template, plot_image=plot_image, udp_port=PORT)

def main():
    """Hlavní funkce"""
    # Spuštění UDP listeneru v samostatném vlákně
    listener_thread = threading.Thread(target=udp_listener, daemon=True)
    listener_thread.start()
    
    # Spuštění Flask serveru
    print("Spouštím web server...")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()