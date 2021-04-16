import numpy as np
from scipy import interpolate
import json
import csv
import os
import sys
import time

module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(module_path)
from velocity_optimization.interface.Receiver import ZMQReceiver as Rec
from velocity_optimization.interface.Sender import ZMQSender as Snd


ESIM_FINISHED = 1
ESIM_CALCULATING = 0
ESIM_UPDATED = 1
ESIM_OUTDATED = 0


class VarPowerLimits:

    __slots__ = ('input_path',
                 '__s_var_pwr',
                 's_max_var_pwr',
                 '__P_var_pwr',
                 'f_pwr_intp',
                 'rec_recalc',
                 'snd_recalc')

    def __init__(self,
                 input_path: str):

        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.02.2020

        Documentation: Class to store and interpolate variable power limits to be used during driving to feed into the
        variable P_max constraint.

        Inputs:
        input_path: absolute path to folder containing variable vehicle and track information
        """

        self.input_path = input_path

        self.snd_recalc = None
        self.rec_recalc = None

        # s coordinate [m]
        self.__s_var_pwr = []
        # max. s coordinate [m]
        self.s_max_var_pwr = 0
        # max. power values [kW]
        self.__P_var_pwr = []

        with open(self.input_path + 'var_power_db.csv', newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.__s_var_pwr = np.append(self.__s_var_pwr, json.loads(row[0]))
                self.__P_var_pwr = np.append(self.__P_var_pwr, json.loads(row[1]))

            # --- Postprocess variable power array (no negative values)
            self.__P_var_pwr[self.__P_var_pwr < 0] = 0

            self.s_max_var_pwr = np.max(self.__s_var_pwr)

            self.f_pwr_intp = interpolate.interp1d(self.__s_var_pwr, self.__P_var_pwr,
                                                   bounds_error=False)

    def init_interface_recalc(self):

        # --- Initialize sender to trigger ESIM recalculation
        self.snd_recalc = Snd(theme='receiver_exp_esim')

        # --- Initialize receiver for information from ESIM
        self.rec_recalc = Rec(theme='sender_imp_vplanner')

    def receive_esim_update(self):
        """Function to receive ESIM update. Call this function manually instead of a while loop in order not to block
        the velocity planner function."""

        # receive updates from ESIM
        update_esim = self.rec_recalc.run()

        if update_esim is not None:
            s_glo_old = self.__s_var_pwr
            P_old = self.__P_var_pwr
            # match first s-coordinate of new plan into indices of old plan
            idx_old = (np.abs(s_glo_old - update_esim[0, 0])).argmin()
            idx_old -= 1
            if idx_old < 0:
                idx_old = 0
            # retrieve s-coordinate of old plan, which is a few meters behind starting s-coordinate of new plan
            s_glo_old = s_glo_old[idx_old]
            P_old = P_old[idx_old]

            self.__s_var_pwr = update_esim[:, 0]
            self.__P_var_pwr = update_esim[:, 1]

            # if old last s coordinate is smaller than new first s-coordinate, reuse this value to enforce smooth
            # transitions between updates
            if s_glo_old < self.__s_var_pwr[0]:
                self.__s_var_pwr = np.insert(self.__s_var_pwr, 0, s_glo_old)
                self.__P_var_pwr = np.insert(self.__P_var_pwr, 0, P_old)

            # --- Postprocess variable power array (no negative values)
            self.__P_var_pwr[self.__P_var_pwr < 0] = 0

            # re-interpolate
            self.f_pwr_intp = interpolate.interp1d(self.__s_var_pwr, self.__P_var_pwr,
                                                   bounds_error=False, assume_sorted=True)

            return ESIM_UPDATED

        else:

            return ESIM_CALCULATING

    def trigger_recalc(self,
                       s_meas: float,
                       meas_diff: np.ndarray,
                       phase: str = 'r',
                       track: str = 'mnt',
                       laps: int = 12,
                       x0: np.ndarray = np.array([1, 0, 0, 0.5, 35, 35, 35, 35, 35])):

        """Triggers energy strategy recalculation with measurement values.

        :param s_meas: global s-coordinate where vehicle currently is
        :param meas_diff: measurement values expressed as difference to energy strtaegy values at current position

        Only necessary for the initialization phase are the following parameters:
        :param phase: which ES phase to call: 'r', recalculation (default); 'i', init.; 'v', v-ref calculation
        :param track: which track: 'mnt', Monteblanco (default), ...
        :param laps: how many laps: any integer, 12 (default)
        :param x0: starting values

        :Authors:
            Thomas Herrmann <thomas.herrmann@tum.de>

        :Created on:
            01.02.2020
        """

        # test interface: start ESIM and v-planner afterwards: a message on the ESIM-side should appear.
        # --- v_ref: 'v' + track-ID
        phase = phase
        track = track
        # --- init.: 'i' + track-ID + number of laps + initial state x0
        laps = laps
        x0 = x0
        # --- re-optim.: 'r' + global coordinate of measurement diff. [m] + measurement diff. [various]
        s_meas = s_meas
        # s [m], v[m/s], t[s], soc_batt [], temp_batt, temp_mach, temp_inv, temp_cool_mi, temp_cool_b [°C]
        meas_diff = meas_diff

        zs_data = dict()
        zs_data['phase'] = phase
        zs_data['track'] = track
        zs_data['num_laps'] = laps
        zs_data['x0'] = x0
        zs_data['s_meas'] = s_meas
        zs_data['meas_diff'] = meas_diff

        self.snd_recalc.send(zs_data)


if __name__ == '__main__':

    vpl = VarPowerLimits(module_path + '/velocity_optimization/inputs/')
    s_meas_ = 70  # meters

    # initialize interfaces to energy strategy
    vpl.init_interface_recalc()
    # trigger recalculation
    '''
    vpl.trigger_recalc(s_meas=0.0,
                       meas_diff=np.array(0),
                       phase='i',
                       track='mnt',
                       laps=12,
                       x0=np.array([1, 0, 0, 0.5, 35, 35, 35, 35, 35]))
    '''
    vpl.trigger_recalc(s_meas=s_meas_,
                       meas_diff=np.array([0, 0, 0, 0, 0.5, 0, 0, 0, 0]))

    # receive an update by the energy strategy if calculation has finished
    r = ESIM_CALCULATING
    while r is not ESIM_UPDATED:
        r = vpl.receive_esim_update()
        time.sleep(1)
        print("Power on global coordinates s_meas, 1, 2 km: ", vpl.f_pwr_intp([s_meas_, 1000, 2000]))
