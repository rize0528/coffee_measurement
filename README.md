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

## Model trainer usage
This project has three built-in machine learning models, such as "Linear regression", "Multi-layer perceptron" and
"Polynomial regression".  
You can run the program inside the docker container or standalone run on your own environment with
properly installed all necessory python packages.
### Package dependency
```bash
sudo pip install -U pandas scikit-learn asciichartpy statsmodels
```
### Basic usage
```bash
  -h, --help            show this help message and exit
  -i or --input [REQUIRED], input filepath, if a folder path has been given, program will scan all *.csv file
                          that could be loaded as pandas dataframe. For the files which name are in form of "eval_*.csv"
                          will be treat as evaluation data, otherwise program will use training data to show the evaluation
                          report after model training.
  -e or --evaluation, Evaluation data-path, accept single file only.  
  -m or --model {regression,reg,mlp_regressor,mlp,poly,poly_reg} [REQUIRED]
  -o or --output, output folder path
  -l or --log-level {debug,info,warning,error}
  -p or --hyper-parameters, if you want to alter the hyper parameter of a ML model, passing the parameter with json-format like: -p "{\"degree\":3}" , all double quotes are required.                                

```
Once the model training has complete, program will dump the evaluation report in console with a pretty ascii chart:
```bash
Kernel type: ========MLP=========

+=======Legend=======+
| Green: Prediction  |
| Red: Ground Truth  |
| Blue: Error value  |
+====================+
+===[Performance report on evaluation data]===+
  128.00  ┤
  119.57  ┼
  111.13  ┤
  102.70  ┼╮            ╭──╮          ╭─╮
   94.26  ┤╮╮        ╭─╮│ ╰╰╮  ╭╮╭╮  ╭╯ ╰
   85.83  ┤╰─╮      ╭╯ ││   ╰─╮││╭╮  │
   77.39  ┤  ╰╮     │╯ ╰╯    ╰│││││ ╭╯
   68.96  ┤   ╰╮    │  ╰╯     ╰╯│││╭│
   60.52  ┤    ╰╮╭╮ │           ╰╯│╭╯
   52.09  ┤     ╰╯│ │             ╰╯
   43.65  ┤       ╰─╯
   35.22  ┤        ╰╯
   26.79  ┤
   18.35  ┤
    9.92  ┤        ╭─╮    ╭╮╭╮
    1.48  ┤╮╭──────╯ ╰────╯╰╯╰───────────
   -6.95  ┤╰╯

```
The ascii art in your prompt should be colored, you can easily compare the predict value and ground truth to evaluate the robustness of your model.

We also list some basic metrics for helping you to evaluate the model, such as the maximum or minimum distance among prediction and grundtruth values.
```
[2020-11-16 18:51:07,114][INFO][CoffeeMeasure.py] Linear regression evaluation report:
[2020-11-16 18:51:07,114][INFO][CoffeeMeasure.py]  |- Max error: 5.946456693912907
[2020-11-16 18:51:07,114][INFO][CoffeeMeasure.py]  |- Min error: 0.907086610894531
[2020-11-16 18:51:07,114][INFO][CoffeeMeasure.py]  |- Average error: 3.275590551151037
[2020-11-16 18:51:07,114][INFO][CoffeeMeasure.py]  |- Average absolute error: 3.275590551151037
```
To avoid log flooding, if you interested on the detail error value of each entry in evaluation data, you can specify the log level as well
```
--log-level debug
```
Detail metrics will like:
```
[2020-11-16 18:51:07,114][DEBUG][CoffeeMeasure.py]  |- Loss value between ground truth and prediction: [3.72913386 5.94645669 3.22519685 0.90708661 3.62834646 2.21732284]
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
