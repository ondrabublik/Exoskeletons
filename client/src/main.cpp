#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <MadgwickAHRS.h>
#include <Arduino.h>
#include "model.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Forward declarations
void task1IMU(void *pvParameters);
void task2Logic(void *pvParameters);

// _________________ neural network__________________________________________
// nastavení paměti
constexpr int kTensorArenaSize = 10 * 1024;   // zmenšeno pro ESP32
uint8_t tensor_arena[kTensorArenaSize];

// tensorFlow objekty
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Příznak pro predikci
bool predictionEnabled = true;  // false = bez neuronové sítě
// _________________ neural network__________________________________________


// _________________ wifi, UDP ______________________________________________
// nastavenní WiFi a UDP
const char* ssid = "hpwifi";
const char* password = "password";

const char* serverIP = "192.168.0.141";
const int serverPort = 8888;
const int localPort = 8889;

WiFiUDP Udp;

// Příznak pro komunikaci
bool outCommunication = true;  // false = bez WiFi a bez odesílání
// _________________ wifi, UDP ______________________________________________

// __________ potenciometer, motor, button, servo __________________________
// pin potenciometru
const int POT_PIN = 35;
float angleValue = 0;
const float angleMin = 0.0f;
const float angleMax = 1.0f;

// muscle button pin
const int MUSCLE_BUTTON_PIN = 19;

// servo pin (PWM)
const int SERVO_PIN = 12;
const int lock = 180;
const int unlock = 0;
const int SERVO_PWM_FREQ = 50;
const int SERVO_PWM_RESOLUTION = 12;
const int SERVO_PWM_CHANNEL = 0;

// motor pin (PWM)
const int MOTOR_PIN = 25;
const int MOTOR_PWM_FREQ = 2000;
const int MOTOR_PWM_RESOLUTION = 10;
const int MOTOR_PWM_CHANNEL = 1;
// Nastavitelná intenzita motoru v rozsahu 0..1
const float MOTOR_INTENSITY = 1.0f;

// LED na D3 (GPIO0)
const int LED_PIN = 3;
int ledState = LOW;  // 0 nebo 1 příchozí binární packet
// __________ potenciometer, motor, button, servo __________________________

// __________ MPU6050 sensors _______________________________________________
// two sensors MPU6050 (0x68 a 0x69)
Adafruit_MPU6050 mpu1;
Adafruit_MPU6050 mpu2;

// two filters
Madgwick filter1;
Madgwick filter2;

float gx1, gy1, gz1;
float gx2, gy2, gz2;

// frekvence IMU
const float imuFreq = 50.0;
const unsigned long imuPeriod = 1000000 / imuFreq;  // 20 ms v mikrosekundách

unsigned long lastIMUTime = 0;
unsigned long lastLogicTime = 0;
const unsigned long logicPeriod = 100000;  // 100 ms v mikrosekundách
// __________ MPU6050 sensors _______________________________________________

// FreeRTOS task handles
TaskHandle_t task1Handle = NULL;
TaskHandle_t task2Handle = NULL;

static uint32_t servoAngleToDuty(float angle) {
  float clamped = constrain(angle, 0.0f, 180.0f);
  float pulseUs = 500.0f + (clamped / 180.0f) * 2000.0f;  // 500-2500 us
  float periodUs = 1000000.0f / SERVO_PWM_FREQ;            // 50 Hz -> 20000 us
  float maxDuty = (1 << SERVO_PWM_RESOLUTION) - 1;
  return (uint32_t)((pulseUs / periodUs) * maxDuty);
}

static uint32_t motorIntensityToDuty(float intensity) {
  float clamped = constrain(intensity, 0.0f, 1.0f);
  float maxDuty = (1 << MOTOR_PWM_RESOLUTION) - 1;
  return (uint32_t)(clamped * maxDuty);
}

void setup() {

  Serial.begin(115200);
  delay(1000);

  // _________________ wifi, UDP _________________________________________________
  if (outCommunication) {
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

    Udp.begin(localPort);
    Serial.println("UDP inicializováno");
  } else {
    Serial.println("WiFi komunikace zakázána (outCommunication = false)");
  }
  // _________________ wifi, UDP _________________________________________________

  // __________ MPU6050 sensors _______________________________________________
  Wire.begin();
  Wire.setClock(400000);   // rychlejší I2C
  Wire.setTimeout(3000);

  // inicializace prvního senzoru (0x68)
  if (!mpu1.begin(0x68)) {
    Serial.println("MPU1 nenalezen!");
  }

  // inicializace druhého senzoru (0x69)
  if (!mpu2.begin(0x69)) {
    Serial.println("MPU2 nenalezen!");
  }

  Serial.println("MPU6050 senzory inicializovány");

  Serial.println("Scanning...");

  for (byte address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    if (Wire.endTransmission() == 0) {
      Serial.print("Found at 0x");
      Serial.println(address, HEX);
    }
  }
  
  // Madgwick filtry
  filter1.begin(imuFreq);
  filter2.begin(imuFreq);
  // __________ MPU6050 sensors _______________________________________________

  // __________ potenciometer, motor, button __________________________________
  // LED pin
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  pinMode(MOTOR_PIN, OUTPUT);
  ledcSetup(MOTOR_PWM_CHANNEL, MOTOR_PWM_FREQ, MOTOR_PWM_RESOLUTION);
  ledcAttachPin(MOTOR_PIN, MOTOR_PWM_CHANNEL);
  ledcWrite(MOTOR_PWM_CHANNEL, 0);
  ledcSetup(SERVO_PWM_CHANNEL, SERVO_PWM_FREQ, SERVO_PWM_RESOLUTION);
  ledcAttachPin(SERVO_PIN, SERVO_PWM_CHANNEL);
  ledcWrite(SERVO_PWM_CHANNEL, servoAngleToDuty(unlock));

  // Muscle button pin
  pinMode(MUSCLE_BUTTON_PIN, INPUT_PULLUP);
  // __________ potenciometer, motor, button __________________________________

  // _________________ neural network__________________________________________
  Serial.println("Inicializace modelu...");

  // Načtení modelu z paměti
  model = tflite::GetModel(model_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Chyba: nekompatibilní verze modelu!");
    while (1);
  }

  // Registr všech operací (jednoduché řešení)
  static tflite::MicroMutableOpResolver<10> resolver;
  
  // Registrace operací potřebných pro model
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddLogistic();

  // Vytvoření interpreteru
  interpreter = new tflite::MicroInterpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize
  );

  // Alokace tensorů
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors selhalo!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model připraven.");
  // _________________ neural network__________________________________________

  // Vytvoření Task 1 (IMU - čtení a filtrování)
  xTaskCreatePinnedToCore(
    task1IMU,           // Funkce tasku
    "Task1_IMU",        // Jméno tasku
    4096,               // Stack size
    NULL,               // Parametry
    2,                  // Priorita
    &task1Handle,       // Task handle
    1                   // Core 0
  );

  // Vytvoření Task 2 (Logika - inference a UDP)
  xTaskCreatePinnedToCore(
    task2Logic,         // Funkce tasku
    "Task2_Logic",      // Jméno tasku
    8192,               // Stack size
    NULL,               // Parametry
    1,                  // Priorita
    &task2Handle,       // Task handle
    0                   // Core 1
  );

  Serial.println("Tasky vytvořeny.");
}

void task1IMU(void *pvParameters) {
  // Task 1: Čtení IMU senzorů a aktualizace filtrů
  unsigned long lastIMUTime = 0;
  
  while (1) {
    unsigned long now = micros();

    if (now - lastIMUTime >= imuPeriod) {
      lastIMUTime = now;

      // data ze senzorů
      sensors_event_t a1, g1, t1;
      sensors_event_t a2, g2, t2;

      mpu1.getEvent(&a1, &g1, &t1);
      mpu2.getEvent(&a2, &g2, &t2);

      // převod rad/s -> deg/s
      gx1 = g1.gyro.x * 57.2958;
      gy1 = g1.gyro.y * 57.2958;
      gz1 = g1.gyro.z * 57.2958;

      gx2 = g2.gyro.x * 57.2958;
      gy2 = g2.gyro.y * 57.2958;
      gz2 = g2.gyro.z * 57.2958;

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
    }
    vTaskDelay(1 / portTICK_PERIOD_MS);  // Krátká pauza
  }
}

void task2Logic(void *pvParameters) {
  // Task 2: Logika - čtení vstupů, inference a UDP odesílání
  unsigned long lastLogicTime = 0;
  
  while (1) {
    unsigned long now = micros();

    // Odesílání dat a inference každých 100 ms
    if (now - lastLogicTime >= logicPeriod) {
      lastLogicTime += logicPeriod;

      // čtení potenciometru
      angleValue = analogRead(POT_PIN) / 4096.0;  // normalizace na 0.0 - 1.0 (předpoklad, že potenciometr je zapojen jako dělič napětí)

      // čtení muscle button
      float muscleButton = digitalRead(MUSCLE_BUTTON_PIN) ? 1.0 : 0.0;  // 1.0 stisknuto, 0.0 nestisknuto

      float roll1  = filter1.getRoll();
      float roll2  = filter2.getRoll();

      // Spuštění inference (pouze pokud je povolena predikce)
      float prediction = 0.0;
      if (predictionEnabled) {
        // Vložení signálů do vstupního tensoru (první dva vstupy)
        if (input->bytes >= 5 * sizeof(float)) {
          input->data.f[0] = angleValue; // TODO - zvolit správné signály pro model
          input->data.f[1] = (roll1 + 90.0) / 180.0;  // normalizace na 0.0 - 1.0
          input->data.f[2] = (gx1 + 100.0) / 200.0;  // normalizace na 0.0 - 1.0 (předpoklad, že gyroskop má rozsah ±100 deg/s)
          input->data.f[3] = (roll2 + 90.0) / 180.0;  // normalizace na 0.0 - 1.0
          input->data.f[4] = (gx2 + 100.0) / 200.0;
        }

        // Spuštění inference
        if (interpreter->Invoke() != kTfLiteOk) {
          Serial.println("Chyba při Invoke()");
        } else {
          // Čtení výstupu
          prediction = output->data.f[0];

          if (prediction > 0.9f && angleValue >= angleMin && angleValue <= angleMax) {
            ledcWrite(MOTOR_PWM_CHANNEL, motorIntensityToDuty(MOTOR_INTENSITY));
            ledcWrite(SERVO_PWM_CHANNEL, servoAngleToDuty(lock));
            digitalWrite(LED_PIN, HIGH);
          } else {
            ledcWrite(MOTOR_PWM_CHANNEL, 0);
            ledcWrite(SERVO_PWM_CHANNEL, servoAngleToDuty(unlock));
            digitalWrite(LED_PIN, LOW);
          }
        }
      }

      // Odeslat binární data (pouze pokud je komunikace povolena)
      if (outCommunication) {
        // 7 hodnot pro binární packet float[7]
        float dataPayload[7] = {
          angleValue,
          roll1,
          gx1,
          roll2,
          gx2,
          muscleButton,
          prediction
        };
        Udp.beginPacket(serverIP, serverPort);
        Udp.write((uint8_t*)dataPayload, sizeof(dataPayload));
        Udp.endPacket();

        // Debug: vypíšeme data do seriálu
        // Serial.printf("Binární packet float[7]: %.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        //             dataPayload[0], dataPayload[1], dataPayload[2],
        //             dataPayload[3], dataPayload[4], dataPayload[5], dataPayload[6]);
      }
    }
    vTaskDelay(1 / portTICK_PERIOD_MS);  // Krátká pauza
  }
}

void loop() {
  // Loop zůstává prázdný - veškerá logika je v taskcích
  vTaskDelay(100 / portTICK_PERIOD_MS);
  //vTaskDelay(portMAX_DELAY);
}