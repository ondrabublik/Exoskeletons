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

WiFiUDP Udp;

Adafruit_MPU6050 mpu;
Madgwick filter;

// frekvence IMU
const float imuFreq = 250.0;
const unsigned long imuPeriod = 1000000 / imuFreq;

unsigned long lastIMUTime = 0;

// odesílání UDP
int readCount = 0;
const int sendDivider = 10;

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

  if (!mpu.begin()) {
    Serial.println("MPU6050 nenalezen!");
    while (1);
  }

  Serial.println("MPU6050 inicializován");

  // Madgwick filtr
  filter.begin(imuFreq);

  Udp.begin(localPort);

  Serial.println("UDP inicializováno");
}

void loop() {

  unsigned long now = micros();

  if (now - lastIMUTime >= imuPeriod) {

    lastIMUTime += imuPeriod;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // rad/s -> deg/s
    float gx = g.gyro.x * 57.2958;
    float gy = g.gyro.y * 57.2958;
    float gz = g.gyro.z * 57.2958;

    filter.updateIMU(
      gx,
      gy,
      gz,
      a.acceleration.x,
      a.acceleration.y,
      a.acceleration.z
    );

    readCount++;

    if (readCount >= sendDivider) {

      readCount = 0;

      float roll  = filter.getRoll();
      float pitch = filter.getPitch();
      float yaw   = filter.getYaw();

      char dataString[128];

      snprintf(
        dataString,
        sizeof(dataString),
        "%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f",
        roll,
        pitch,
        yaw,
        gx,
        gy,
        gz,
        0.0
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