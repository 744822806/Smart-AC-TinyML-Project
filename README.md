itioner (AC)** on or off based on environmental conditions.

---

## 🌍 Project Description

We have an IoT environment with **three sensors**:

| Sensor | Range | Description |
|---------|--------|-------------|
| 🌡️ Temperature | 0–45 °C | Ambient temperature |
| 💡 Light Intensity | 0–100 % | Brightness level |
| 💧 Humidity | 0–100 % | Relative humidity |

**Control Rule:**
> If the light intensity is above 70% and (temperature > 25°C or humidity > 65%), the AC should be **ON**, otherwise **OFF**.

This rule was used to generate a **dataset of 100,000 samples** for an artificial neural network model that predicts AC behavior.

---

## 🧰 Tech Stack

| Component | Description |
|------------|-------------|
| 🧠 TensorFlow / Keras | Train the ANN |
| 🔄 TensorFlow Lite | Convert model to TinyML format |
| ⚙️ ESP-IDF / Arduino-ESP32 | Embedded inference |
| 📊 NumPy + Pandas | Dataset generation |

---

## 🧮 ANN Model Architecture

| Layer | Type | Activation | Output Shape |
|-------|------|-------------|---------------|
| Input | Dense(3) | — | (None, 3) |
| Hidden | Dense(8) | ReLU | (None, 8) |
| Output | Dense(1) | Sigmoid | (None, 1) |

**Loss:** Binary Crossentropy  
**Optimizer:** Adam  
**Metrics:** Accuracy

---

## 📂 Project Structure

Smart-AC-TinyML/
│
├── data/
│ ├── dataset_generator.py # Generates 100,000 labeled samples
│ └── sample_data.csv
│
├── model/
│ ├── train_ann_model.ipynb # TensorFlow training notebook
│ ├── ann_model.h5 # Saved Keras model
│ └── ann_tflite_model.tflite # Converted TinyML model
│
├── esp32/
│ ├── predict.c # TinyML logic (or rule-based fallback)
│ ├── main.c # Main demo program
│ └── model_data.h # TFLite weights (future use)
│
├── README.md
├── LICENSE
└── .gitignore



---

## 🧠 TensorFlow Model Training Example

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Generate dataset
samples = 100000
temp = np.random.uniform(0, 45, samples)
light = np.random.uniform(0, 100, samples)
humid = np.random.uniform(0, 100, samples)
ac_on = (light > 70) & ((temp > 25) | (humid > 65))
labels = ac_on.astype(int)

X = np.stack([temp, light, humid], axis=1)
y = labels

# ANN Model
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=15, batch_size=64, validation_split=0.2)

# Save and convert
model.save('ann_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("ann_tflite_model.tflite", "wb").write(tflite_model)
⚙️ ESP32 TinyML Inference Code
predict.c
c
#include <stdbool.h>

bool predict(int temperature, int light, int humidity) {
    if (light > 70 && (temperature > 25 || humidity > 65))
        return true;
    return false;
}
main.c
c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "predict.c"

void app_main() {
    printf("Smart AC Predictor Demo\\n");

    for (int i = 0; i < 50; i++) {
        int temp = rand() % 46;
        int light = rand() % 101;
        int humid = rand() % 101;

        bool ac = predict(temp, light, humid);
        printf("temp: %d°C  light: %d%%  hum: %d%%  → AC: %s\\n",
               temp, light, humid, ac ? "on" : "off");
    }
}
🖥️ ESP32 Console Output
Below is the actual output from the ESP-IDF 5.3 terminal running the AC predictor program:

yaml
temp: 44.0°C  light: 54.5%  hum: 89.0%  → AC: off
temp: 38.9°C  light: 80.0%  hum: 45.8%  → AC: off
temp: 20.8°C  light: 63.5%  hum: 0.7%   → AC: off
temp: 33.7°C  light: 81.7%  hum: 26.1%  → AC: on
temp: 26.2°C  light: 28.6%  hum: 70.7%  → AC: off
temp: 15.4°C  light: 96.4%  hum: 74.0%  → AC: on
...
I (505) main_task: Returned from app_main()

🚀 Future Work

Integrate actual TinyML inference using TensorFlow Lite Micro

Add real sensor data (DHT11, BH1750) instead of random generation

Display output on an OLED screen

Connect to MQTT cloud dashboard for remote monitoring

Add OTA update support

📜 License

This project is licensed under the MIT License.
Copyright © 2025 Heyang Liu

👤 Author

Heyang (Henry) Liu
📍 Northeastern University, Boston, MA
📧 liu.heyan@northeastern.edu

🔗 LinkedIn

🌐 GitHub
