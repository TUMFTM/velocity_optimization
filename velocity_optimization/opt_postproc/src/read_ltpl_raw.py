import sys
import os
import numpy as np
import linecache
import csv
import json
from matplotlib import pyplot as plt
try:
    import tikzplotlib
except ImportError:
    print('Warning: No module tikzplotlib found. Not necessary on car but for development.')

# custom modules
vel_opt_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(vel_opt_path)


def read_ltpl_raw(csv_name: str) -> tuple:

    """
    Python version: 3.5
    Created by: Thomas Herrmann (thomas.herrmann@tum.de)
    Created on: 01.02.2020

    Documentation: Read log data from TUM trajectory planner module

    Inputs:
    csv_name: Path and file name to log file

    Outputs:
    s_glob_arr: array filled with global s coordinate values [m]
    v_arr: array containing the driven velocity [m/s]
    ax_arr: array containing the driven longitudinal acceleration [m/s^2]
    ay_arr: array containing the driven lateral acceleration [m/s^2]
    """

    # [m]
    s_glob_arr = []
    # [m/s]
    v_arr = []
    # [m/s^2]
    ax_arr = []
    # [rad/m]
    kappa_arr = []

    with open(csv_name) as csvfile:
        row_count = sum(1 for row in
                        csv.reader(csvfile,
                                   delimiter=';',
                                   lineterminator='\n'))

    for i in range(3, row_count):
        row_lc = linecache.getline(csv_name, i)
        row_lc = row_lc[:-1].rsplit(';')

        s_glob = json.loads(row_lc[1])
        s_glob_arr.append(s_glob)
        v = json.loads(row_lc[8])['straight'][0][0]
        v_arr.append(v)
        ax = json.loads(row_lc[9])['straight'][0][0]
        ax_arr.append(ax)
        kappa = json.loads(row_lc[11])['straight'][0][0]
        kappa_arr.append(kappa)

    ################################################################################################################
    # Calculations with LTPL log data
    ################################################################################################################
    ay_arr = kappa_arr * np.square(v_arr)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(s_glob_arr, v_arr)
    plt.xlabel(r's in m')
    plt.ylabel(r'v in m/s')
    plt.subplot(2, 1, 2)
    plt.plot(s_glob_arr, ay_arr)
    plt.xlabel(r's in m')
    plt.ylabel(r'a_y in m/s2')

    tikzplotlib.save('ltpl.tex')
    plt.show()

    return s_glob_arr, v_arr, ax_arr, ay_arr


if __name__ == '__main__':

    csv_name = vel_opt_path + '/../logs/ltpl/2020_04_01/18_47_51_data.csv'

    read_ltpl_raw(csv_name=csv_name)
