#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

// ====== WiFi nastavení ======
const char* ssid = "hpwifi";
const char* password = "password";

// ====== UDP server nastavení ======
const char* serverIP = "172.29.6.137";  // IP adresa počítače se serverem
const int serverPort = 8888;
const int localPort = 8889;  // Lokální port pro příjem odpovědí

WiFiUDP Udp;

// ====== Simulace dat ze senzorů ======
// Nahraďte tuto funkci čtením z vašich skutečných senzorů
void readSensorData(float* data) {
  // Příklad: simulace 7 hodnot
  // V reálném použití zde načtete data z IMU, akcelerometru, gyroskopu, atd.
  data[0] = 15.76;
  data[1] = -0.9612;
  data[2] = 0.1213;
  data[3] = 0.0205;
  data[4] = -4.4122;
  data[5] = -1.5725;
  data[6] = -2.1527;
  
  // Případně můžete použít skutečné senzory:
  // data[0] = analogRead(A0) * 3.3 / 1024.0;
  // atd.
}

void setup() {
  Serial.begin(115200);
  delay(100);
  
  Serial.println();
  Serial.print("Připojování k WiFi: ");
  Serial.println(ssid);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.println("WiFi připojeno!");
  Serial.print("IP adresa: ");
  Serial.println(WiFi.localIP());
  
  // Spuštění UDP
  Udp.begin(localPort);
  Serial.print("UDP server spuštěn na portu: ");
  Serial.println(localPort);
  Serial.print("Odesílám data na: ");
  Serial.print(serverIP);
  Serial.print(":");
  Serial.println(serverPort);
}

void loop() {
  float sensorData[7];
  
  // Načtení dat ze senzorů
  readSensorData(sensorData);
  
  // Vytvoření řetězce ve formátu: "val1, val2, val3, val4, val5, val6, val7"
  String dataString = "";
  for (int i = 0; i < 7; i++) {
    if (i > 0) dataString += ", ";
    dataString += String(sensorData[i], 4);  // 4 desetinná místa
  }
  
  Serial.print("Odesílám data: ");
  Serial.println(dataString);
  
  // Odeslání dat přes UDP
  Udp.beginPacket(serverIP, serverPort);
  Udp.write(dataString.c_str());
  Udp.endPacket();
  
  // Čekání na odpověď (volitelné)
  delay(100);  // Krátká pauza pro příjem odpovědi
  
  int packetSize = Udp.parsePacket();
  if (packetSize) {
    char incomingPacket[255];
    int len = Udp.read(incomingPacket, 255);
    if (len > 0) {
      incomingPacket[len] = 0;
    }
    
    Serial.print("Odpověď ze serveru: ");
    Serial.println(incomingPacket);
    
    // Parsování odpovědi (formát: "predikce,pravděpodobnost")
    // Můžete použít výsledek pro další zpracování
  }
  
  // Čekání před dalším odesláním (např. 100ms = 10 Hz)
  delay(100);
}
