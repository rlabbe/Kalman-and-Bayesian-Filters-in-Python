# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 15:20:47 2015

@author: rlabbe
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    data
except:
    cols = ('time','millis','ax','ay','az','rollrate','pitchrate','yawrate',
            'roll','pitch','yaw','speed','course','latitude','longitude',
            'altitude','pdop','hdop','vdop','epe')
    data = np.genfromtxt('2014-03-26-000-Data.csv', delimiter=',',
                         names=True, usecols=cols).view(np.recarray)



plt.subplot(311)
plt.plot(data.ax)
plt.subplot(312)
plt.plot(data.speed)
plt.subplot(313)
plt.plot(data.course)
plt.tight_layout()

plt.figure()
plt.subplot(311)
plt.plot(data.pitch)
plt.subplot(312)
plt.plot(data.roll)
plt.subplot(313)
plt.plot(data.yaw)
plt.figure()
plt.plot(data.longitude, data.latitude)