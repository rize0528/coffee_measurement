# coffee_measurement
This project is a ML/SL project to learn or to fit the CM-100 coffee colour analyser 
using cheap colour sensor to acquire similar measurement value. 

## Requirement 
To make the project able to cross-platform, all components will be ran in a docker image,
for how to install docker, please refer to [Get Docker](https://docs.docker.com/get-docker/).
 
## Installation
Open a terminal which capable to launch docker without any problem, then input the following commands:

```bash
cd <download_path>/coffee_measure/
```
Build SDK with dockerfile
```bash
sudo docker build -f dockerfile -t coffee_measure .
```
Launch the image and get trained model.
```bash
sudo docker run 
    -v <training_data_fullpath_if_have>:/opt/makerclub/res 
    -v <output_folder_full_path>:/opt/makerclub/output 
    --rm coffee_measure 
    /usr/local/bin/python /opt/makerclub/src/train.py --model mlp --input /opt/makerclub/res/ -l info
```




# Hardware design

## Arduino dependency

1. Matrix Math by Charlie Matlack, if you use MLP model.
   You don't need Matrix Math for linear regression model.

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
1. https://github.com/greiman/SSD1306Ascii in order to reduce memory footprint.

```
Nano D4 -> 2222A (Base)
Nano D9 -> Measure
Nano D10 -> Calibration
```


# License

TBD
