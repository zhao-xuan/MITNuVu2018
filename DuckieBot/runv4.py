#!/usr/bin/python
# Import Adafruit Motor HAT Library
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
# Import additional libraries that support MotorHAT
import time
import atexit

# create a default MotorHAT object, no changes to I2C address or frequency
mh = Adafruit_MotorHAT(addr=0x60)
lmotor = mh.getMotor(1)
rmotor = mh.getMotor(2)

# recommended for auto-disabling motors on shutdown!
def turnOffMotors():
	mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
	mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
	mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
	mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

atexit.register(turnOffMotors)



# Complete this function so:
# 1. values in the range 1 to 32768 make the motor spin forward faster and faster.
# 2. values in the range -1 to -32768 make the motor spin backward faster and faster.
# 3. any value equal to 0 makes the motor BRAKE.
# 4. any values less than -32768 and greater than 32768 use the max speed in the right direction.
def runMotors(speed, extraoffset,sleep_time,stop):
    offset = 19
    if speed > -32768 and speed < 32768:   
        scaled_speed = 255*speed/32768
        if speed == 0:
            lmotor.setSpeed(0)
            rmotor.setSpeed(0)
            lmotor.run(Adafruit_MotorHAT.BRAKE)
            rmotor.run(Adafruit_MotorHAT.BRAKE)
        elif speed < 0:
    	    lmotor.setSpeed(scaled_speed-offset)
    	    rmotor.setSpeed(scaled_speed)
            lmotor.run(Adafruit_MotorHAT.BACKWARD)
            rmotor.run(Adafruit_MotorHAT.BACKWARD)
        elif speed > 0:
    	    lmotor.setSpeed(scaled_speed-offset)       
            rmotor.setSpeed(scaled_speed)       
            lmotor.run(Adafruit_MotorHAT.FORWARD)
            rmotor.run(Adafruit_MotorHAT.FORWARD)
    else:
        if speed <= -32768:
       	    lmotor.setSpeed(255-offset)
            rmotor.setSpeed(255)
            lmotor.run(Adafruit_MotorHAT.BACKWARD)
            rmotor.run(Adafruit_MotorHAT.BACKWARD)
        elif speed >= 32768:
    	    lmotor.setSpeed(255-offset)
    	    rmotor.setSpeed(255)
            lmotor.run(Adafruit_MotorHAT.FORWARD)
            rmotor.run(Adafruit_MotorHAT.FORWARD)
    time.sleep(sleep_time)
    if stop==True:
        lmotor.setSpeed(0)
        rmotor.setSpeed(0)
        lmotor.run(Adafruit_MotorHAT.BRAKE)
        rmotor.run(Adafruit_MotorHAT.BRAKE)

        
runMotors(32767, 0, 3, False)  
        
while True:
    pass



  

