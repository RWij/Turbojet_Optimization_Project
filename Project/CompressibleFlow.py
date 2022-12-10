# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 05:43:01 2022

@author: ryant
"""

import numpy as np;

def ARatio(M, gamma):
    return np.power((1 + (gamma - 1) * 0.5 * M * M)/\
                    (0.5 * (gamma + 1)), (gamma + 1) / (2 * (gamma - 1))) / M;
        
def TR(M, gamma=1.4):
    return 1 + ((gamma - 1) * 0.5 * M * M);

def PR(M, gamma=1.4):
    return np.power(1 + ((gamma - 1) * 0.5 * M * M),\
                    gamma / (gamma - 1));
        
def RhoR(M, gamma=1.4):
    return np.power(1 + (0.5 * (gamma - 1) * M * M), 1 / (gamma - 1));

def mdot(M, area, Po, To, R, gamma=1.4):
    return M * (Po * np.sqrt(gamma / (R * To))) * area /\
        np.power(1 + (gamma - 1) * 0.5 * M * M, (gamma + 1) / (2 * (gamma - 1)));
        
def Area(mdot, To, Po, R, gamma, M):
    return mdot * (np.sqrt(To) / Po) * (np.sqrt(R / gamma) / M)\
        * np.power(1 + (0.5 * (gamma - 1) * M**2), (gamma + 1) / (2 * (gamma - 1)));