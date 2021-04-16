import json
import os
import configparser
import zmq
import time
import ad_interface_functions
import numpy as np


class ZMQSender:

    def __init__(self,
                 theme: str):
        # --------------------------------------------------------------------------------------------------------------
        # IMPORT INTERFACE CONFIG PARAMETERS ---------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        parser = configparser.ConfigParser()
        pars = {}

        if not parser.read(os.path.join(repo_path, "params/interface_config.ini")):
            raise ValueError('Specified config file does not exist or is empty!')

        pars["receiver_exp_zmq"] = json.loads(parser.get('INTERFACE_SPEC_SENDER', theme))

        # --------------------------------------------------------------------------------------------------------------
        # OPEN INTERFACES ----------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # initialization -----------------------------------------------------------------------------------------------
        zmq_context = zmq.Context()
        int_receiver_zmq = {"opts_exp": pars["receiver_exp_zmq"]}

        # RECEIVER via ZMQ ---------------------------------------------------------------------------------------------
        int_receiver_zmq["sock_exp"] = zmq_context.socket(zmq.PUB)
        int_receiver_zmq["sock_exp"].bind("tcp://*:%s" % int_receiver_zmq["opts_exp"]["port"])

        # wait a short time until all sockets are really bound (ZMQ specific problem) ----------------------------------
        time.sleep(0.5)

        self.int_receiver_zmq = int_receiver_zmq

        print("All sockets opened (sender)!")

    def send(self, data):
        # --------------------------------------------------------------------------------------------------------------
        # SEND MESSAGES ------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # LTPL via ZMQ -------------------------------------------------------------------------------------------------
        ad_interface_functions.zmq_export.zmq_export(sock=self.int_receiver_zmq["sock_exp"],
                                                     topic=self.int_receiver_zmq["opts_exp"]["topic"],
                                                     data=data,
                                                     datatype='pyobj')

    def __del__(self):
        # --------------------------------------------------------------------------------------------------------------
        # CLOSE SOCKETS ------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        self.int_receiver_zmq["sock_exp"].close()

        time.sleep(0.5)

        print("All sockets closed (sender)!")


if __name__ == "__main__":

    zs = ZMQSender(theme='receiver_exp_esim')

    # test interface: start ESIM and v-planner afterwards: a message on the ESIM-side should appear.
    # --- v_ref: 'v' + track-ID
    phase = 'r'
    track = 'mnt'
    # --- init.: 'i' + track-ID + number of laps + initial state x0
    laps = 12
    x0 = np.array([1, 0, 0, 0.5, 35, 35, 35, 35, 35])
    # --- re-optim.: 'r' + global coordinate of measurement diff. [m] + measurement diff. [various]
    s_meas = 70
    # s [m], v[m/s], t[s], soc_batt [], temp_batt, temp_mach, temp_inv, temp_cool_mi, temp_cool_b [°C]
    meas_diff = np.array([0, 0, 0, 0, 0.5, 0, 0, 0, 0])

    zs_data = dict()
    zs_data['phase'] = phase
    zs_data['track'] = track
    zs_data['num_laps'] = laps
    zs_data['x0'] = x0
    zs_data['s_meas'] = s_meas
    zs_data['meas_diff'] = meas_diff

    zs.send(zs_data)
