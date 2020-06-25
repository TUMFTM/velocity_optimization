import numpy as np
import time


class IniSQP:

    __slots__ = ('__f_v0',
                 '__slr_v0',
                 '__f_action_id',
                 '__slr_action_id',
                 '__f_time',
                 '__slr_time',
                 '__t_threshold',
                 '__b_print_sm')

    def __init__(self):

        """
        Python version: 3.5
        Created by: Thomas Herrmann (thomas.herrmann@tum.de)
        Created on: 01.01.2020

        Documentation: Class to check which initialization for the velocity SQP-solver is appropriate online.
        """

        # --- Last velocity profiles that have been planned [m/s]
        self.__f_v0 = None
        self.__slr_v0 = None
        # --- Last action ID
        self.__f_action_id = None
        self.__slr_action_id = None
        # --- Timestamp of last planned velocity profile [UNIX timestamp]
        self.__f_time = None
        self.__slr_time = None

        # Time threshold [s]
        self.__t_threshold = 1
        self.__b_print_sm = False

    def set_vx(self,
               plan: str,
               action_id: str,
               vx: np.ndarray):
        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.02.2020

        Documentation: Stores the currently optimized velocity profile.

        Inputs:
        plan: follow 'f' or straight/left/right 'slr'
        action_id: 'follow', 'straight', 'left' or 'right'
        vx: currently optimized velocity profile
        """

        # --- Check planner mode and store information
        if plan == 'slr':
            self.__slr_v0 = vx
            self.__slr_action_id = action_id
            self.__slr_time = time.time()

        elif plan == 'f':
            self.__f_v0 = vx
            self.__f_action_id = action_id
            self.__f_time = time.time()

    def get_v0(self,
               plan: str,
               action_id: str,
               m: int,
               b_print_sm: bool = False) -> np.array:

        """
        Python version: 3.5
        Created by: Thomas Herrmann
        Created on: 01.02.2020

        Documentation: Get best initialization for velocity profile to be optimized based on past velocity profiles
        and time information.

        Inputs:
        plan: follow 'f' or straight/left/right 'slr'
        action_id: 'follow', 'straight', 'left' or 'right'
        m: number of velocity points
        b_print_sm: print initialization statemachine decision logic

        Outputs:
        v0: velocity initial guess [m/s]
        """

        t = time.time()
        v0 = None

        ################################################################################################################
        # --- Check planner mode and retrieve initialization
        ################################################################################################################
        if plan == 'slr':
            ############################################################################################################
            # --- GO STRAIGHT
            ############################################################################################################
            # --- Case: go straight and slr was planned a short time ago (prio 1)
            if action_id == 'straight' and \
                    (self.__slr_action_id == 'straight'
                     or self.__slr_action_id == 'left'
                     or self.__slr_action_id == 'right') and \
                    (t - self.__slr_time) <= self.__t_threshold:
                v0 = self.__slr_v0
                if b_print_sm:
                    print('to straight from straight')
                return v0
            # --- Case: go straight and follow plan is present (prio 2)
            elif action_id == 'straight' and \
                    self.__f_action_id == 'follow':
                v0 = self.__f_v0
                if b_print_sm:
                    print('to straight from follow')
                return v0

            ############################################################################################################
            # --- GO LEFT
            ############################################################################################################
            # --- Case: left and left was planned shortly before (prio 1)
            if action_id == 'left' and \
                    self.__slr_action_id == 'left':
                v0 = self.__slr_v0
                if b_print_sm:
                    print('to left from left')
                return v0
            # --- Case: switch from planned overtake to right to left overtake (exception)
            elif action_id == 'left' and \
                    self.__slr_action_id == 'right' and \
                    self.__f_v0 is not None:
                v0 = self.__f_v0
                if b_print_sm:
                    print('to left from follow')
                return v0
            # --- Case: Plan overtake when in follow mode (prio 2)
            elif action_id == 'left' and \
                    self.__f_v0 is not None:
                v0 = self.__f_v0
                if b_print_sm:
                    print('to left from follow')
                return v0
            # --- Case: Plan overtake from straight profile (prio 3)
            elif action_id == 'left' and \
                    self.__slr_v0 is not None:
                v0 = self.__slr_v0
                if b_print_sm:
                    print('to left from straight')
                return v0

            ############################################################################################################
            # --- GO RIGHT
            ############################################################################################################
            # --- Case: right and right was planned shortly before (prio 1)
            if action_id == 'right' and \
                    self.__slr_action_id == 'right':
                v0 = self.__slr_v0
                if b_print_sm:
                    print('to right form right')
                return v0
            # --- Case: switch from planned overtake to left to right overtake (exception)
            elif action_id == 'right' and \
                    self.__slr_action_id == 'left' and \
                    self.__f_v0 is not None:
                v0 = self.__f_v0
                if b_print_sm:
                    print('to right from follow')
                return v0
            # --- Case: Plan overtake when in follow mode (prio 2)
            elif action_id == 'right' and \
                    self.__f_v0 is not None:
                v0 = self.__f_v0
                if b_print_sm:
                    print('to right from follow')
                return v0
            # --- Case: Plan overtake from straight profile (prio 3)
            elif action_id == 'right' and \
                    self.__slr_v0 is not None:
                v0 = self.__slr_v0
                if b_print_sm:
                    print('to right from follow')
                return v0

            # --- Check if no case matched
            if v0 is None:
                v0 = 5 * np.ones((m, ))
                print('Exception velocity-guess taken!')
                return v0

        elif plan == 'f':
            ############################################################################################################
            # --- FOLLOW
            ############################################################################################################
            # --- Case: follow and f was planned a short time ago
            if action_id == 'follow' and \
                    self.__f_action_id == 'follow' and \
                    t - self.__f_time <= self.__t_threshold:
                v0 = self.__f_v0
                if b_print_sm:
                    print('to follow from follow')
                return v0
            # --- Case: Opponent reached TODO: modify x0 to meet v_max-requirements
            elif action_id == 'follow' and \
                    self.__slr_action_id == 'straight' and self.__slr_v0 is not None:
                v0 = self.__slr_v0
                if b_print_sm:
                    print('to follow from straight')
                return v0
            elif action_id == 'follow' and \
                    self.__slr_action_id == 'left' or \
                    self.__slr_action_id == 'right' and \
                    self.__slr_v0 is not None:
                v0 = self.__slr_v0
                if b_print_sm:
                    print('to follow from left/right')
                return v0

            else:
                v0 = 20 * np.ones((m, ))
                print('Exception velocity-guess for follow taken!')
                return v0


if __name__ == '__main__':
    pass
