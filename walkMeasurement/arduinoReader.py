import serial
import serial.tools.list_ports
import time
from datetime import datetime

PORT = "COM3"
BAUDRATE = 9600

def wait_for_port(port_name):
    """Čeká, dokud se neobjeví daný port."""
    print(f"Čekám na připojení {port_name}...")
    while True:
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if port_name in ports:
            print(f"✅ Zařízení na {port_name} nalezeno.")
            return
        time.sleep(1)

def record_data():
    """Čte data z Arduina a ukládá je do souboru, dokud se port neodpojí."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_{timestamp}.txt"
    print(f"📁 Ukládám data do souboru: {filename}")

    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        time.sleep(2)  # počkej na inicializaci Arduina
        with open(filename, "w", encoding="utf-8") as f:
            while True:
                try:
                    line = ser.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        print(line)
                        f.write(line + "\n")
                except serial.SerialException:
                    # Pokud se port odpojí během čtení
                    print("⚠️ Port byl odpojen. Ukládám data...")
                    break
    except serial.SerialException:
        # Port nelze otevřít (např. odpojené zařízení)
        pass
    finally:
        try:
            ser.close()
        except:
            pass

# --- Hlavní smyčka ---
print("Pro ukončení programu stiskni Ctrl+C.")
try:
    while True:
        wait_for_port(PORT)
        record_data()
except KeyboardInterrupt:
    print("\n👋 Ukončuji program.")
