#!/usr/bin/python
# Import Adafruit Motor HAT Library
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
# Import the device reading library
from evdev import InputDevice, categorize, ecodes, KeyEvent, list_devices

# Import additional libraries that support MotorHAT
import time
import atexit


# recommended for auto-disabling motors on shutdown!
def turnOffMotors():
	mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
	mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
	mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
	mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)
	
	
# create a default MotorHAT object, no changes to I2C address or frequency
mh = Adafruit_MotorHAT(addr=0x60)
lmotor = mh.getMotor(1)
rmotor = mh.getMotor(2)



atexit.register(turnOffMotors)


# Complete this function so:
# 1. values in the range 1 to 32768 make the motor spin forward faster and faster.
# 2. values in the range -1 to -32768 make the motor spin backward faster and faster.
# 3. any value equal to 0 makes the motor BRAKE.
# 4. any values less than -32768 and greater than 32768 use the max speed in the right direction.
def runMotor(motor, speed):
	""" motor - the motor object to control.
		speed - a number from -32768 (reverse) to 32768 (forward) """
	# COMPLETE THIS FUNCTION!
	if(speed <= -32768):
		motor.setSpeed(255)
		motor.run(Adafruit_MotorHAT.BACKWARD)
	
	elif(speed >= 32768):
		motor.setSpeed(255)
		motor.run(Adafruit_MotorHAT.FORWARD)
		
	elif(speed > -32768 and speed < 0):
		motor.setSpeed(int(-float(speed)/129.0))
		motor.run(Adafruit_MotorHAT.BACKWARD)
	
	elif(speed < 32768 and speed > 0):
		motor.setSpeed(int(float(speed)/129.0))
		motor.run(Adafruit_MotorHAT.FORWARD)
	
	else:
		motor.setSpeed(0)
		motor.run(Adafruit_MotorHAT.BRAKE)


def turnMotor(l_motor, r_motor, speed, direction):
	if(direction == 'left'):
		runMotor(r_motor, speed)
		runMotor(l_motor, -speed)
	elif(direction == 'right'):
		runMotor(r_motor, -speed)
		runMotor(l_motor, speed)
	else:
		runMotor(l_motor, 0)
		runMotor(r_motor, 0)


# Get the name of the Logitech Device
def getInputDeviceByName(name):
	devices = [InputDevice(fn) for fn in list_devices()]
	for device in devices:
		if device.name == name:
			return InputDevice(device.fn)
		return None
# Import our gamepad.
gamepad = getInputDeviceByName('Logitech Gamepad F710')

# Loop over the gamepad's inputs, reading it.
for event in gamepad.read_loop():
	if event.type == ecodes.EV_KEY:
		keyevent = categorize(event)
		if keyevent.keystate == KeyEvent.key_down:
			print(keyevent.keycode)
      # example key detection code
			if 'BTN_A' in keyevent.keycode:
        # Do something here when the A button is pressed
				pass
			elif 'BTN_START' in keyevent.keycode:
        # Do something here when the START button is pressed
				pass
	elif event.type == ecodes.EV_ABS:
		if event.code == 0:
			print('PAD_LR '+str(event.value))
		elif event.code == 1:
			print('PAD_UD '+str(event.value))
		elif event.code == 2:
			print('TRIG_L '+str(event.value))
			turnMotor(lmotor, rmotor, event.value*129.0, 'left')
		elif event.code == 3:
			print('JOY_LR '+str(event.value))
			if(event.value < 0):
				turnMotor(lmotor, rmotor, event.value, 'left')
			elif(event.value > 0):
				turnMotor(lmotor, rmotor, event.value, 'right')
		elif event.code == 4:
			print('JOY_UD '+str(event.value))
		elif event.code == 5:
			print('TRIG_R '+str(event.value))
			turnMotor(lmotor, rmotor, event.value*129.0, 'right')
		elif event.code == 16:
			print('HAT_LR '+str(event.value))
			if(event.value < 0):
				turnMotor(lmotor, rmotor, 255*129.0, 'left')
			elif(event.value > 0):
				turnMotor(lmotor, rmotor, 255*129.0, 'right')
			else:
				runMotor(lmotor, 0)
				runMotor(rmotor, 0)
		elif event.code == 17:
			print('HAT_UD '+str(event.value))
			if(event.value < 0):
				runMotor(lmotor, 32768)
				runMotor(rmotor, 32768)
			elif(event.value > 0):
				runMotor(lmotor, -32768)
				runMotor(rmotor, -32768)
			else:
				runMotor(lmotor, 0)
				runMotor(rmotor, 0)
		else:
			pass
			
while(True):
	pass





