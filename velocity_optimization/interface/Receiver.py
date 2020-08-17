import json
import os
import configparser
import zmq
import time
import ad_interface_functions


class ZMQReceiver:

    def __init__(self,
                 theme: str):

        # --------------------------------------------------------------------------------------------------------------
        # IMPORT INTERFACE CONFIG PARAMETERS ---------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        parser = configparser.ConfigParser()
        pars = {}

        if not parser.read(os.path.join(repo_path, "velocity_optimization/params/interface_config.ini")):
            raise ValueError('Specified config file does not exist or is empty!')

        pars["sender_imp_zmq"] = json.loads(parser.get('INTERFACE_SPEC_RECEIVER', theme))

        # --------------------------------------------------------------------------------------------------------------
        # OPEN INTERFACES ----------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # initialization -----------------------------------------------------------------------------------------------
        zmq_context = zmq.Context()
        int_sender_zmq = {"opts_imp": pars["sender_imp_zmq"]}

        # SENDER via ZMQ -----------------------------------------------------------------------------------------------
        int_sender_zmq["sock_imp"] = zmq_context.socket(zmq.SUB)
        int_sender_zmq["sock_imp"].connect("tcp://%s:%s" % (int_sender_zmq["opts_imp"]["ip"],
                                                            int_sender_zmq["opts_imp"]["port"]))
        int_sender_zmq["sock_imp"].setsockopt_string(zmq.SUBSCRIBE, int_sender_zmq["opts_imp"]["topic"])

        # wait a short time until all sockets are really bound (ZMQ specific problem) ----------------------------------
        time.sleep(0.5)

        self.int_sender_zmq = int_sender_zmq

        print("All sockets opened (receiver)!")

        # --------------------------------------------------------------------------------------------------------------
        # FETCH MESSAGES -----------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

    def run(self):
        # SENDER via ZMQ -----------------------------------------------------------------------------------------------
        recv_data = ad_interface_functions.zmq_import.zmq_import(sock=self.int_sender_zmq["sock_imp"],
                                                                 blocking=False,
                                                                 datatype='pyobj')

        if recv_data is not None:
            return recv_data

    def __del__(self):
        # --------------------------------------------------------------------------------------------------------------
        # CLOSE SOCKETS ------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        self.int_sender_zmq["sock_imp"].close()

        time.sleep(0.5)

        print("All sockets closed (receiver)!")


if __name__ == "__main__":
    zmq_receiver = ZMQReceiver(theme="empty_theme")
    zmq_receiver.run()
