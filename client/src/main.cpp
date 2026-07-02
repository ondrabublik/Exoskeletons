#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <ESP32Servo.h>
#include <Arduino.h>
#include <math.h>
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

// Buffer pro časovou historii vstupů do NN
constexpr int kNnChannels = 7;       // angle, c1, s1, gx1, c2, s2, gx2
constexpr int kNnBufferLength = 10;  // volitelná délka bufferu
float nnInputBuffer[kNnChannels][kNnBufferLength] = {0.0f};
int nnSamplesCollected = 0;
// _________________ neural network__________________________________________


// _________________ wifi, UDP ______________________________________________
// nastavenní WiFi a UDP
const char* ssid = "HONOR 200";
const char* password = "wifihonor";
//const char* ssid = "AimtecHackathon2026";
//const char* password = "Kdyzkodpomaha";

//const char* serverIP = "192.168.30.86";
const int localPort = 8889;

const char* serverIP = "10.255.57.209";
const int serverPort = 9999;

WiFiUDP Udp;

// Příznak pro komunikaci
bool outCommunication = true;  // false = bez WiFi a bez odesílání
float dataPayload[9];
// _________________ wifi, UDP ______________________________________________

// __________ potenciometer, motor, button, servo __________________________
// pin potenciometru
const int POT_PIN = 35;
float angleValue = 0;
float potStartupOffset = 0;
const float angleMin = 0.13f;
const float angleMax = 0.8f;

// muscle button pin
const int MUSCLE_BUTTON_PIN = 19;

// servo pin (PWM)
const int SERVO_PIN = 12;
const int lock = 0;
const int unlock = 90;
Servo doorServo;

// motor pin (PWM)
const int MOTOR_PIN = 25;
const int MOTOR_PWM_FREQ = 2000;
const int MOTOR_PWM_RESOLUTION = 10;
const int MOTOR_PWM_CHANNEL = 1;
// Nastavitelná intenzita motoru v rozsahu 0..1
const float MOTOR_INTENSITY = 1.0f;

// LED na D3 (GPIO0)
const int LED_PIN = 4;
int ledState = LOW;  // 0 nebo 1 příchozí binární packet
// __________ potenciometer, motor, button, servo __________________________

// __________ MPU6050 sensors _______________________________________________
// two sensors MPU6050 (0x68 a 0x69)
Adafruit_MPU6050 mpu1;
Adafruit_MPU6050 mpu2;
bool mpu1_ok = false;
bool mpu2_ok = false;

// frekvence IMU
const unsigned long imuPeriod = 100000;  // 100 ms v mikrosekundách

unsigned long lastIMUTime = 0;
unsigned long lastLogicTime = 0;
const unsigned long logicPeriod = 100000;  // 100 ms v mikrosekundách
// __________ MPU6050 sensors _______________________________________________

// FreeRTOS task handles
TaskHandle_t task1Handle = NULL;
TaskHandle_t task2Handle = NULL;

static bool initMpuWithRetry(Adafruit_MPU6050& mpu, uint8_t address, const char* name) {
  constexpr int kInitAttempts = 5;
  constexpr uint32_t kRetryDelayMs = 500;
  for (int attempt = 1; attempt <= kInitAttempts; ++attempt) {
    if (mpu.begin(address)) {
      Serial.printf("%s inicializovan na pokus %d\n", name, attempt);
      return true;
    }
    Serial.printf("%s init selhal (pokus %d/%d)\n", name, attempt, kInitAttempts);
    delay(kRetryDelayMs);
  }
  Serial.printf("%s nenalezen po %d pokusech.\n", name, kInitAttempts);
  return false;
}

static uint32_t motorIntensityToDuty(float intensity) {
  float clamped = constrain(intensity, 0.0f, 1.0f);
  float maxDuty = (1 << MOTOR_PWM_RESOLUTION) - 1;
  return (uint32_t)(clamped * maxDuty);
}

static float normalizeGyro(float gxDegPerSec) {
  return (gxDegPerSec + 100.0f) / 200.0f;
}

static float normalizeTrig01(float value) {
  float clamped = constrain(value, -1.0f, 1.0f);
  return (clamped + 1.0f) * 0.5f;
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

  // inicializace senzorů s retry (pomáhá při pomalejším náběhu po bootu)
  mpu1_ok = initMpuWithRetry(mpu1, 0x68, "MPU1");
  mpu2_ok = initMpuWithRetry(mpu2, 0x69, "MPU2");
  if (!mpu1_ok && !mpu2_ok) {
    Serial.println("Zadny MPU6050 nenalezen - IMU data budou nulova.");
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
  
  // __________ MPU6050 sensors _______________________________________________

  // __________ potenciometer, motor, button __________________________________
  // LED pin
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  // Kratka kalibrace potenciometru po startu (referencni offset)
  constexpr int kPotCalibrationSamples = 16;
  float potSum = 0.0f;
  for (int i = 0; i < kPotCalibrationSamples; ++i) {
    potSum += analogRead(POT_PIN) / 4096.0f;
    delay(5);
  }
  potStartupOffset = potSum / kPotCalibrationSamples;
  Serial.printf("Potenciometr offset pri startu: %.4f\n", potStartupOffset);

  pinMode(MOTOR_PIN, OUTPUT);
  ledcSetup(MOTOR_PWM_CHANNEL, MOTOR_PWM_FREQ, MOTOR_PWM_RESOLUTION);
  ledcAttachPin(MOTOR_PIN, MOTOR_PWM_CHANNEL);
  ledcWrite(MOTOR_PWM_CHANNEL, 0);
  doorServo.setPeriodHertz(50);
  doorServo.attach(SERVO_PIN, 500, 2500);
  doorServo.write(unlock);

  // Muscle button pin
  pinMode(MUSCLE_BUTTON_PIN, INPUT_PULLUP);
  // __________ potenciometer, motor, button __________________________________

  // _________________ neural network__________________________________________
  if (predictionEnabled) {
    Serial.println("Inicializace modelu...");

    // Načtení modelu z paměti
    model = tflite::GetModel(model_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("Chyba: nekompatibilní verze modelu!");
      while (1);
    }

    // Registr všech operací (jednoduché řešení)
    static tflite::MicroMutableOpResolver<20> resolver;
    
    // Registrace operací potřebných pro model
    resolver.AddConv2D();        // pro Conv1D !!!
    resolver.AddReshape();       // Flatten
    resolver.AddExpandDims();    // často u Conv1D / vstupního tvaru v Keras exportu
    resolver.AddMean();          // GlobalAveragePooling / reduce_mean v grafu
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
  } else {
    Serial.println("Predikce vypnuta - model se neinicializuje.");
  }
  // _________________ neural network__________________________________________

  // Vytvoření Task 1 (IMU - čtení a filtrování)
  xTaskCreatePinnedToCore(
    task1IMU,           // Funkce tasku
    "Task1_IMU",        // Jméno tasku
    4096,               // Stack size
    NULL,               // Parametry
    1,                  // Priorita
    &task1Handle,       // Task handle
    0                   // Core 0
  );

  // Vytvoření Task 2 (Logika - inference a UDP)
  xTaskCreatePinnedToCore(
    task2Logic,         // Funkce tasku
    "Task2_Logic",      // Jméno tasku
    8192,               // Stack size
    NULL,               // Parametry
    2,                  // Priorita
    &task2Handle,       // Task handle
    1                   // Core 1
  );

  Serial.println("Tasky vytvořeny.");
}

void task1IMU(void *pvParameters) {
  // Task 1: Odeslani sekundoveho batch
  unsigned long lastIMUTime = 0;
  
  while (1) {
    unsigned long now = micros();

    if (now - lastIMUTime >= imuPeriod) {
      lastIMUTime = now;
      // Odeslat binární data (pouze pokud je komunikace povolena)
      if (outCommunication) {
        Udp.beginPacket(serverIP, serverPort);
        Udp.write((uint8_t*)dataPayload, sizeof(dataPayload));
        Udp.endPacket();

        // // Debug: vypíšeme data do seriálu
        // Serial.printf("Binární packet float[7]: %.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
        //             dataPayload[0], dataPayload[1], dataPayload[2],
        //             dataPayload[3], dataPayload[4], dataPayload[5], dataPayload[6]);
}
      
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

      // data ze senzorů (bezpečný fallback na nuly, pokud IMU chybí)
      float gx1 = 0.0f;
      float gx2 = 0.0f;
      float ax1 = 0.0f;
      float ay1 = 0.0f;
      float ax2 = 0.0f;
      float ay2 = 0.0f;

      if (mpu1_ok) {
        sensors_event_t a1, g1, t1;
        mpu1.getEvent(&a1, &g1, &t1);
        gx1 = g1.gyro.x * 57.2958f;  // rad/s -> deg/s
        ax1 = a1.acceleration.x;
        ay1 = a1.acceleration.y;
      }

      if (mpu2_ok) {
        sensors_event_t a2, g2, t2;
        mpu2.getEvent(&a2, &g2, &t2);
        gx2 = g2.gyro.x * 57.2958f;  // rad/s -> deg/s
        ax2 = a2.acceleration.x;
        ay2 = a2.acceleration.y;
      }


      // čtení potenciometru
      angleValue = (analogRead(POT_PIN) / 4096.0f) - potStartupOffset;  // aktualni hodnota minus offset pri zapnuti

      // čtení muscle button
      float muscleButton = digitalRead(MUSCLE_BUTTON_PIN) ? 1.0 : 0.0;  // 1.0 stisknuto, 0.0 nestisknuto

      float norm1 = sqrtf(ax1 * ax1 + ay1 * ay1);
      float norm2 = sqrtf(ax2 * ax2 + ay2 * ay2);
      float c1 = 0.0f;
      float s1 = 0.0f;
      float c2 = 0.0f;
      float s2 = 0.0f;
      if (norm1 > 1e-6f) {
        c1 = ax1 / norm1;
        s1 = ay1 / norm1;
      }
      if (norm2 > 1e-6f) {
        c2 = ax2 / norm2;
        s2 = ay2 / norm2;
      }

      // normalizované vstupy (kanály)
      float nnChannels[kNnChannels];
      nnChannels[0] = angleValue;
      nnChannels[1] = normalizeTrig01(c1);
      nnChannels[2] = normalizeTrig01(s1);
      nnChannels[3] = normalizeGyro(gx1);
      nnChannels[4] = normalizeTrig01(c2);
      nnChannels[5] = normalizeTrig01(s2);
      nnChannels[6] = normalizeGyro(gx2);

      // posun bufferu doleva a vložení nového vzorku na konec
      for (int ch = 0; ch < kNnChannels; ch++) {
        for (int i = 0; i < kNnBufferLength - 1; i++) {
          nnInputBuffer[ch][i] = nnInputBuffer[ch][i + 1];
        }
        nnInputBuffer[ch][kNnBufferLength - 1] = nnChannels[ch];
      }
      if (nnSamplesCollected < kNnBufferLength) {
        nnSamplesCollected++;
      }

      // Spuštění inference (pouze pokud je povolena predikce)
      float prediction = 0.0;
      if (predictionEnabled) {
        bool readyForInference = false;
        // Do vstupu NN vlož všechny hodnoty bufferu pro každý kanál
        const int nnInputCount = kNnChannels * kNnBufferLength;
        if (input && input->bytes >= nnInputCount * (int)sizeof(float) && nnSamplesCollected >= kNnBufferLength) {
          int idx = 0;
          for (int i = 0; i < kNnBufferLength; i++) {
            for (int ch = 0; ch < kNnChannels; ch++) {
              input->data.f[idx++] = nnInputBuffer[ch][i];
            }
          }
          readyForInference = true;
        }

        // Spuštění inference
        if (readyForInference && interpreter->Invoke() != kTfLiteOk) {
          Serial.println("Chyba při Invoke()");
        } else if (readyForInference) {
          // Čtení výstupu
          prediction = output->data.f[0];

          if (prediction > 0.6f && angleValue >= angleMin && angleValue <= angleMax) {
            ledcWrite(MOTOR_PWM_CHANNEL, motorIntensityToDuty(MOTOR_INTENSITY));
            doorServo.write(lock);
            digitalWrite(LED_PIN, HIGH);
          } else {
            ledcWrite(MOTOR_PWM_CHANNEL, 0);
            doorServo.write(unlock);
            digitalWrite(LED_PIN, LOW);
          }
        }
      }

      if (outCommunication) {
        // 9 hodnot pro binární packet float[9]
        dataPayload[0] = angleValue;
        dataPayload[1] = normalizeTrig01(c1);
        dataPayload[2] = normalizeTrig01(s1);
        dataPayload[3] = normalizeGyro(gx1);
        dataPayload[4] = normalizeTrig01(c2);
        dataPayload[5] = normalizeTrig01(s2);
        dataPayload[6] = normalizeGyro(gx2);
        dataPayload[7] = muscleButton;
        dataPayload[8] = prediction;
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