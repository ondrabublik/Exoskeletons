#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <MadgwickAHRS.h>

const char* ssid = "hpwifi";
const char* password = "password";

const char* serverIP = "172.29.5.235";
const int serverPort = 8888;
const int localPort = 8889;

// pin potenciometru
const int POT_PIN = A0;

WiFiUDP Udp;

// dva senzory
Adafruit_MPU6050 mpu1;
Adafruit_MPU6050 mpu2;

// dva filtry
Madgwick filter1;
Madgwick filter2;

// frekvence IMU
const float imuFreq = 250.0;
const unsigned long imuPeriod = 1000000 / imuFreq;

unsigned long lastIMUTime = 0;

// odesílání UDP
int readCount = 0;
const int sendDivider = 25;

void setup() {

  Serial.begin(115200);

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

  Wire.begin();

  // inicializace prvního senzoru (0x68)
  if (!mpu1.begin(0x68)) {
    Serial.println("MPU1 nenalezen!");
    //while (1);
  }

  // inicializace druhého senzoru (0x69)
  if (!mpu2.begin(0x69)) {
    Serial.println("MPU2 nenalezen!");
    //while (1);
  }

  Serial.println("MPU6050 senzory inicializovány");

  // Madgwick filtry
  filter1.begin(imuFreq);
  filter2.begin(imuFreq);

  Udp.begin(localPort);

  Serial.println("UDP inicializováno");
}

void loop() {

  unsigned long now = micros();

  if (now - lastIMUTime >= imuPeriod) {

    lastIMUTime += imuPeriod;

    // data ze senzorů
    sensors_event_t a1, g1, t1;
    sensors_event_t a2, g2, t2;

    mpu1.getEvent(&a1, &g1, &t1);
    mpu2.getEvent(&a2, &g2, &t2);

    // převod rad/s -> deg/s
    float gx1 = g1.gyro.x * 57.2958;
    float gy1 = g1.gyro.y * 57.2958;
    float gz1 = g1.gyro.z * 57.2958;

    float gx2 = g2.gyro.x * 57.2958;
    float gy2 = g2.gyro.y * 57.2958;
    float gz2 = g2.gyro.z * 57.2958;

    // aktualizace filtrů
    filter1.updateIMU(
      gx1,
      gy1,
      gz1,
      a1.acceleration.x,
      a1.acceleration.y,
      a1.acceleration.z
    );

    filter2.updateIMU(
      gx2,
      gy2,
      gz2,
      a2.acceleration.x,
      a2.acceleration.y,
      a2.acceleration.z
    );

    readCount++;

    if (readCount >= sendDivider) {

      readCount = 0;

      // čtení potenciometru
      float pot = analogRead(POT_PIN) / 1024.0;

      float roll1  = filter1.getRoll();
      float pitch1 = filter1.getPitch();

      float roll2  = filter2.getRoll();
      float pitch2 = filter2.getPitch();

      char dataString[200];

      snprintf(
        dataString,
        sizeof(dataString),
        "%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f",
        pot,
        roll1,
        pitch1,
        gz1,
        roll2,
        pitch2,
        gz2
      );

      Serial.print("Odesílám: ");
      Serial.println(dataString);

      Udp.beginPacket(serverIP, serverPort);
      Udp.write((uint8_t*)dataString, strlen(dataString));
      Udp.endPacket();

      int packetSize = Udp.parsePacket();

      if (packetSize) {

        char incomingPacket[255];
        int len = Udp.read(incomingPacket, 255);

        if (len > 0) incomingPacket[len] = 0;

        Serial.print("Odpověď: ");
        Serial.println(incomingPacket);
      }
    }
  }
}