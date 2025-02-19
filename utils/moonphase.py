"""
Code inspired by:

moonphase.py - Calculate Lunar Phase
Author: Sean B. Palmer, inamidst.com
Cf. http://en.wikipedia.org/wiki/Lunar_phase#Lunar_phase_calculation
"""
import math, decimal, datetime


def phase(datetime_str=None):
    dec = decimal.Decimal
    if datetime_str is None: 
        now = datetime.datetime.now()
    
    now = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')

    diff = now - datetime.datetime(2001, 1, 1)
    days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
    lunations = dec("0.20439731") + (days * dec("0.03386319269"))
    pos = lunations % dec(1)

    index = (pos * dec(8)) + dec("0.5")
    index = math.floor(index)
    return {
        0: "New Moon", 
        1: "Waxing Crescent", 
        2: "First Quarter", 
        3: "Waxing Gibbous", 
        4: "Full Moon", 
        5: "Waning Gibbous", 
        6: "Last Quarter", 
        7: "Waning Crescent"
    }[int(index) & 7]