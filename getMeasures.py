#!/usr/bin/env python

from __future__ import print_function
import sys
import time

from Phidgets.Devices.InterfaceKit import InterfaceKit
from Phidgets.PhidgetException import PhidgetException

def grabInterfaceKit():
    try:
        interface_kit = InterfaceKit()
        interface_kit.openPhidget()
        interface_kit.waitForAttach(0)
    except PhidgetException as e:
        print("Phidget Exception %i: %s" % (e.code, e.details))
        try:
            interface_kit.closePhidget()
        except PhidgetException as e:
            print("Phidget Exception %i: %s" % (e.code, e.details))
            print("Exiting....")
            exit(1)
        print("Exiting....")
        exit(1)
    return interface_kit


def print_help(program_name):
    print("syntax: ", program_name, " nb_of_devices [period]")
    print("    nb_of_device: number of connected devices (must be on the first ports)")
    print("    period: period between each measure in ms (default 1000 ms)")


def parseArgs(argv):
    for arg in argv:
        if arg == '-h':
            print_help(argv[0])
            exit(0)
    if len(argv) > 3 or len(argv) < 2:
        print_help(argv[0])
        exit(1)

    nbOfDevices = int(argv[1])
    if len(argv) == 3:
        periodOfDevice = int(argv[2])
    else:
        periodOfDevice = 1000

    return nbOfDevices, periodOfDevice

def value_changed_handler(event):
    print('{"timestamp" : %f, "sensor" : %i, "value" : %i}'
          % (time.time(), event.index, event.value))

def dump_as_csv(event):
    print('%f,%i'
          % (time.time(), event.value))

def main():
    (device_nb, device_period) = parseArgs(sys.argv)
    interface_kit = grabInterfaceKit()
    for i in range(interface_kit.getSensorCount()):
        if i < device_nb:
            interface_kit.setSensorChangeTrigger(i, 0)
            interface_kit.setDataRate(i, device_period)
    interface_kit.setOnSensorChangeHandler(value_changed_handler)
    sys.stdin.read(1)
    interface_kit.closePhidget()


if __name__ == '__main__':
    main()