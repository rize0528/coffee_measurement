# coffee_measurement
This project is a ML/SL project to learn or to fit the CM-100 coffee colour analyser 
using cheap colour sensor to acquire similar measurement value. 

## Requirement 
To make the project able to cross-platform, all components will be ran in a docker image,
for how to install docker, please refer to [Get Docker](https://docs.docker.com/get-docker/).
 
## Installation
(TBA)

## Train the model
(TBA)



# Hardware design

## Arduino dependency

1. Matrix Math by Charlie Matlack

## Wemos D1 Mini

Dependency:
1. Adafruit SSD1306 Wemos Mini OLED

```
OLED -> Nano
 SDA -> A4
 SCL -> A5
 VCC -> 5V
 GND -> GND
```

## Arduino Nano

Dependency:
1. Adafruit SSD1306 OLED (128x64)

```
Nano D4 -> 2222A (Base)
Nano D9 -> Measure
Nano D10 -> Calibration
```


# License

TBD
