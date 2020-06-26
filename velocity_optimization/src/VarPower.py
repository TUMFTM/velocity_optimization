import os
import numpy as np
from scipy import interpolate
import json
import csv


class VarPowerLimits:

    __slots__ = ('__s_var_pwr',
                 's_max_var_pwr',
                 '__P_var_pwr',
                 'f_pwr_intp')

    def __init__(self):

        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.02.2020

        Documentation: Class to store and interpolate variable power limits to be used during driving to feed into the
        variable P_max constraint.
        """

        toppath_sqp = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        # s coordinate [m]
        self.__s_var_pwr = []
        # max. s coordinate [m]
        self.s_max_var_pwr = 0
        # max. power values [kW]
        self.__P_var_pwr = []

        with open(toppath_sqp + '/inputs/var_power_db.csv', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.__s_var_pwr = np.append(self.__s_var_pwr, json.loads(row[0]))
                self.__P_var_pwr = np.append(self.__P_var_pwr, json.loads(row[1]))

            # --- Postprocess variable power array (no negative values)
            self.__P_var_pwr[self.__P_var_pwr < 0] = 0

            self.s_max_var_pwr = np.max(self.__s_var_pwr)

            self.f_pwr_intp = interpolate.interp1d(self.__s_var_pwr, self.__P_var_pwr)


if __name__ == '__main__':
    vpl = VarPowerLimits()
    print(vpl.f_pwr_intp([2, 3, 400]))
