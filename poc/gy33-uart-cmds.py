# -*- coding: utf8 -*-
# UART commands for GY33

from serial import Serial
from time import sleep

def connect():
    global sr
    sr = Serial(port='/dev/tty.usbserial-0001', timeout=10, baudrate=9600)

def disableAutoReading(sr):
    sr.flushInput()
    sr.flushOutput()
    sr.write(b'\xa5\x00\xa5')

def calibrate(sr):
    sr.flushInput()
    sr.flushOutput()
    sr.write(b'\xa5\xbb\x60')

def read_rgb(sr):
    sr.flushInput()
    sr.flushOutput()
    sr.write(b'\xa5\x54\xf9')           # Read cooked RGB
    sleep(.1)
    buf = sr.read(8)
    if buf[0:3] != b'ZZ\x45':
        print('Error reading:', buf)
        return
    r = buf[4]
    g = buf[5]
    b = buf[6]
    crc = buf[7]
    return 'cr = {}, cg = {}, cb = {}'.format(r, g, b)


def read_lux(sr):
    sr.flushInput()
    sr.flushOutput()
    sr.write(b'\xa5\x52\xf7')
    sleep(.1)
    buf = sr.read(11)
    if buf[0:3] != b'ZZ\x25':
        print('Error reading:', buf)
        return
    lux = buf[4] * 256 + buf[5]
    ct = buf[6] * 256 + buf[7]
    return 'lux = {}, ct = {} K'.format(lux, ct)


def read_raw(sr):
    sr.flushInput()
    sr.flushOutput()
    sr.write(b'\xa5\x51\xf6')
    sleep(.1)
    buf = sr.read(13)
    if buf[0:3] != b'ZZ\x15':
        print('Error reading:', buf)
        return
    r = buf[4] * 256 + buf[5]
    g = buf[6] * 256 + buf[7]
    b = buf[8] * 256 + buf[9]
    c = buf[10] * 256 + buf[11]
    crc = buf[12]
    return 'rr = {}, rg = {}, rb = {}, rc = {}'.format(r, g, b, c)

def read_all(sr):
    print(read_rgb(sr) + ', ' + read_lux(sr) + ', ' + read_raw(sr))
