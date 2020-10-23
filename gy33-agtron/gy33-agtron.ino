// Copyright (C) 2020, Trend Micro Maker's Club.
// Licensed as BSD 2-clause
// Agtron measurement with GY-33 (TCS34725)
// Might be somewhat less accurate, compared with GY-31 (TCS3200), but we don't have data.

// Pin-outs for Wemos D1 Mini
// GY33 Vcc -> Vcc
// GY33 GND -> PN2222A Collector
// GY33 TX  -> RX
// GY33 RX  -> TX
// Measure button -> D5
// Calibration button -> D6
// PN2222A Emittor -> GND
// PN2222A Base -> 1K -> D7

// Pin-outs for Arduino Nano
// OLED SDA -> A4
// OLED SCL -> A5
// OLED Vcc -> 5V
// OLED GND -> GND
// PN2222 Base -> 1K -> D4
// Measure button -> D9
// Calibration button -> D10


#ifdef ARDUINO_AVR_NANO     // Arduino Nano
#define GY33_SW     4
#define MEASURE_BTN 9
#define CALIBRA_BTN 10
#else                       // Wemos D1 Mini
#define GY33_SW     D7
#define MEASURE_BTN D5
#define CALIBRA_BTN D6
#endif


#include "param.h"  // Model trained by ML algorithms in this file.
#include <math.h>




#ifdef ARDUINO_AVR_NANO
#include "SSD1306Ascii.h"
#include "SSD1306AsciiAvrI2c.h"
SSD1306AsciiAvrI2c display;

#else   // Not Arduino Nano, then Wemos D1 Mini
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
Adafruit_SSD1306 display(0);  // GPIO 0
// #if (SSD1306_LCDHEIGHT != 48)
// #error("Height incorrect, please fix Adafruit_SSD1306.h!");
// #endif
#endif

double offset = -28.559567914690287;

double fract(double x)
{
  return x - int(x);
}

double mix(double a, double b, double t)
{
  return a + (b - a) * t;
}

double step(double e, double x)
{
  return x < e ? 0.0 : 1.0;
}

// Adopted from https://gist.github.com/postspectacular/2a4a8db092011c6743a7
void rgb2hsv(double r, double g, double b, double* hsv)
{
  r = r / 255.0;
  g = g / 255.0;
  b = b / 255.0;
  double s = step(b, g);
  double px = mix(b, g, s);
  double py = mix(g, b, s);
  double pz = mix(-1.0, 0.0, s);
  double pw = mix(0.6666666, -0.3333333, s);
  s = step(px, r);
  double qx = mix(px, r, s);
  double qz = mix(pw, pz, s);
  double qw = mix(r, px, s);
  double d = qx - min(qw, py);
  hsv[0] = qz + (qw - py) / (6.0 * d + 1e-10);
  if (hsv[0] < 0) hsv[0] = -hsv[0];
  hsv[1] = d / (qx + 1e-10);
  hsv[2] = qx;
}

double tanh(double x)
{
  double x0 = exp(x);
  double x1 = 1.0 / x0;

  return ((x0 - x1) / (x0 + x1));
}

void matrix_tanh(int n, mtx_type *mtx)
{
  int i;
  for (i = 0; i < n; i++) {
    mtx[i] = (mtx_type) tanh((double)mtx[i]);
  }
}

bool calibrated = false;

// Model: 3 x 10 x 5 x 3 x 1

void flushSerial() {
  Serial.flush();
  while (Serial.available() > 0) {
    Serial.read();
  }
}

void clearScreen() {
#ifdef ARDUINO_AVR_NANO
  display.clear();
#else
  display.clearDisplay();
  display.setCursor(0, 0);
#endif
}

void displayScreen() {
#ifndef ARDUINO_AVR_NANO
  display.display();
#endif
}

void showStatus(String s) {
  display.setCursor(0, 32);
  display.print(s);
#ifndef ARDUINO_AVR_NANO
  display.display();
#endif
}

void disableAutoReading() {
  byte cmd[] = {0xa5, 0x00, 0xa5};
  flushSerial();
  Serial.write(cmd, 3);
}


// read rr, rg, rb, rc
int readTCS34725(unsigned int *r, unsigned int *g, unsigned int *b, unsigned int *c) {
  byte cmd[] = {0xa5, 0x51, 0xf6};
  byte buf[13];
  byte crc = 0;
  int i;

  for (i = 10; i > 0; i--) {
    memset(buf, 0, 13);
    flushSerial();
    Serial.write(cmd, 3);
    delay(50);
    Serial.readBytes(buf, 13);
    if (buf[0] == 'Z' && buf[1] == 'Z' && buf[2] == 0x15) {
      break;
    }
    delay(200);
  }
  if (i == 0) {
    showStatus(F("Read error"));
    return -1;
  }
  for (i = 0; i < 12; i++) {
    crc += buf[i];
  }
  if (crc != buf[i]) {
    showStatus("Invalid CRC");
    return -1;
  }
  *r = (unsigned int) buf[4] << 8 | buf[5];
  *g = (unsigned int) buf[6] << 8 | buf[7];
  *b = (unsigned int) buf[8] << 8 | buf[9];
  *c = (unsigned int) buf[10] << 8 | buf[11];
  return 0;
}

double calcAgtron(unsigned int r, unsigned int g, unsigned int b) {
  mtx_type hsv[3];
  mtx_type c0[10];
  mtx_type c1[10];
  mtx_type c2[5];
  mtx_type c3[5];
  mtx_type c4[3];
  mtx_type c5[3];
  mtx_type c6[1];
  mtx_type c7[1];
  double pred;

  rgb2hsv((double)r, (double)g, (double)b, (double *)hsv);
  Matrix.Print(hsv, 1, 3, "hsv");
  Matrix.Multiply((mtx_type*)hsv, (mtx_type*)W0, 1, 3, 10, (mtx_type*)c0);
  Matrix.Add((mtx_type*)c0, (mtx_type*)W1, 1, 10, (mtx_type*)c1);
  matrix_tanh(10, (mtx_type*)c1);
  Matrix.Print(c1, 1, 10, "c1");
  Matrix.Multiply((mtx_type*)c1, (mtx_type*)W2, 1, 10, 5, (mtx_type*)c2);
  Matrix.Add((mtx_type*)c2, (mtx_type*)W3, 1, 5, (mtx_type*)c3);
  matrix_tanh(5, (mtx_type*)c3);
  Matrix.Print(c3, 1, 5, "c3");
  Matrix.Multiply((mtx_type*)c3, (mtx_type*)W4, 1, 5, 3, (mtx_type*)c4);
  Matrix.Add((mtx_type*)c4, (mtx_type*)W5, 1, 3, (mtx_type*)c5);
  matrix_tanh(3, (mtx_type*)c5);
  Matrix.Print(c5, 1, 3, "c5");
  Matrix.Multiply((mtx_type*)c5, (mtx_type*)W6, 1, 3, 1, (mtx_type*)c6);
  Matrix.Add((mtx_type*)c6, (mtx_type*)W7, 1, 1, (mtx_type*)c7);
  Matrix.Print(c7, 1, 1, "c7");
  pred = c7[0] * 128 + offset;
  return pred;
}

void measure_it() {
  unsigned int r, g, b, c;
  double t;

  r = g = b = c = 0;
  clearScreen();
  display.println(F("Measuring"));
  displayScreen();

  digitalWrite(GY33_SW, HIGH);
  delay(3000);          // wait until LED is stable
  if (readTCS34725(&r, &g, &b, &c) == 0) {
    clearScreen();
    display.print(F("r = "));
    display.println(r);
    display.print(F("g = "));
    display.println(g);
    display.print(F("b = "));
    display.println(b);
    display.print(F("c = "));
    display.println(c);
    t = calcAgtron(r, g, b);
    display.print(F("t = "));
    display.print(t);
    displayScreen();
  }
  delay(1000);
  digitalWrite(GY33_SW, LOW);
}

void calibrate() {
  byte cmd[] = {0xa5, 0xbb, 0x60};

  clearScreen();
  display.println(F("Pre-heating"));
  displayScreen();
  digitalWrite(GY33_SW, HIGH);
  delay(5000);          // wait until LED is stable
  display.println(F("Calibrating"));
  flushSerial();
  Serial.write(cmd, 3);
  delay(2000);
  digitalWrite(GY33_SW, LOW);
  display.println("Done.");
  displayScreen();
  calibrated = true;
  delay(3 * 1000);      // not necessary, but user needs time to read msg.
}

void setup() {
  Serial.begin(9600);

#ifdef ARDUINO_AVR_NANO
  display.begin(&Adafruit128x64, 0x3C);
  display.setFont(Adafruit5x7);
#else
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.display();
#endif
  delay(2000);
#ifdef ARDUINO_AVR_NANO
  display.clear();
#else
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
#endif
  display.println(F("Maker Club"));
  display.println(F("Agtron Measurement"));
  display.println(F("v0.1"));
  displayScreen();

  pinMode(GY33_SW, OUTPUT);
  pinMode(MEASURE_BTN, INPUT_PULLUP);
  pinMode(CALIBRA_BTN, INPUT_PULLUP);
  digitalWrite(GY33_SW, LOW);
  delay(2000);
}


void loop() {
  if (calibrated == true && digitalRead(MEASURE_BTN) == LOW) {
    measure_it();
  } else if (digitalRead(CALIBRA_BTN) == LOW) {
    calibrate();
  }

  if (calibrated == false) {
    clearScreen();
    display.println(F("Please calibrate."));
    displayScreen();
    delay(500);
  }
  delay(100);
}
