#!/usr/bin/env python2

import numpy as np
import math
from scan_simulator_2d import PyScanSimulator2D

# Try to change to just `from scan_simulator_2d import PyScanSimulator2D`
# if any error re: scan_simulator_2d occurs

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler


class SensorModel:
    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param(
            "~lidar_scale_to_map_scale", 1.0
        )

        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = None
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization,
        )

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback, queue_size=1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        # downselect 1000 scans to 100
        # find probablity of scan based on map and ground truth
        # average probably for 100 scans (for each particle) to find probability of particle
        # put those in lookup table

        def p_hit(z_k, d, sigma, z_max=self.table_width):
            if (0 <= z_k) and (z_k <= z_max - 1):
                return (1.0 / math.sqrt(2.0 * math.pi * (sigma**2))) * math.exp(
                    -((z_k - d) ** 2) / (2.0 * (sigma**2))
                )
            return 0

        def p_short(z_k, d):
            if (0 <= z_k) and (z_k <= d) and (d != 0):
                return (2.0 / d) * (1.0 - (float(z_k) / d))
            return 0

        def p_max(z_k, z_max):
            if z_k == z_max - 1:
                # z_max-1 = final pixel
                return 1
            return 0

        def p_rand(z_k, z_max):
            if (0 <= z_k) and (z_k <= z_max - 1):
                return 1.0 / (z_max - 1)
            return 0

        # Construct normalized table for just p_hit
        p_hit_table = np.empty((self.table_width, self.table_width))  # initalize table
        for z_k in range(self.table_width):
            for d in range(self.table_width):
                p_hit_table[z_k, d] = p_hit(z_k, d, self.sigma_hit)
        p_hit_table = p_hit_table / p_hit_table.sum(
            axis=0, keepdims=1
        )  # normalize p_hit

        # ----

        def p_zk(z_k, d, z_max=self.table_width):
            """
            Use precomputed p_hit_table to return p_zk value for given (z_k, d)
            """
            return (
                self.alpha_hit * p_hit_table[z_k, d]  # find p_hit from normalized table
                + self.alpha_short * p_short(z_k, d)
                + self.alpha_max * p_max(z_k, z_max)
                + self.alpha_rand * p_rand(z_k, z_max)
            )

        # Use p_hit_table to construct final table
        table = np.empty((self.table_width, self.table_width))  # initalize table
        for z_k in range(self.table_width):  # iterative over every space in grid
            for d in range(self.table_width):
                # put p_zk value in each spot in table
                table[z_k, d] = p_zk(z_k, d)

        table = table / table.sum(axis=0, keepdims=1)  # normalize final table
        self.sensor_model_table = table

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle

        # Find raycast 'ground truth'
        scans = self.scan_sim.scan(particles)

        # Downsample, if necessary
        if len(observation) > self.num_beams_per_particle:
            assert (
                len(observation) >= self.num_beams_per_particle
            ), "Can't downsample LIDAR data, more ray-traced beams than actual LIDAR beams!"
            obs_downsampled = np.zeros(self.num_beams_per_particle)
            for i in range(self.num_beams_per_particle):
                j = int(
                    np.linspace(0, len(observation) - 1, self.num_beams_per_particle)[i]
                )
                obs_downsampled[i] = observation[j]
            observation = obs_downsampled

        # Convert distance -> pixels
        conversion_d_px = 1.0 / (self.map_resolution * self.lidar_scale_to_map_scale)
        observation = np.multiply(observation, conversion_d_px)
        scans = np.multiply(scans, conversion_d_px)

        # def evaluate_particle(particle_scans, lidar_data):
        #     """
        #     Helper function for finding probability of a single particle.

        #     Assumes
        #         - particle_scans and lidar_data are 1D arrays of equal length
        #         - both arrays are in px, not m
        #         - both arrays are clipped between 0 and self.table_width-1
        #     """
        #     n = len(particle_scans)
        #     probabilities = np.zeros(n)

        #     for i in range(n):
        #         z_k = lidar_data[i]
        #         d = particle_scans[i]
        #         probabilities[i] = self.sensor_model_table[z_k, d]

        #     return np.prod(probabilities)**(1.0/2.2)

        # # Assign probability to each particle
        scans = np.rint(np.clip(scans, 0, self.table_width - 1)).astype(int)
        observation = np.rint(np.clip(observation, 0, self.table_width - 1)).astype(int)
        # probabilities = np.zeros(len(particles))

        # for i in range(len(particles)):
        #     probabilities[i] = evaluate_particle(scans[i], observation)
        # return probabilities

        # --- VECTORIZED
        n, m = len(scans), len(observation)
        observation = np.repeat(np.array(observation[:, np.newaxis]), n, axis=1).T
        probabilities = self.sensor_model_table[observation, scans]
        return np.power(np.prod(probabilities, axis=1), 1.0 / 2.2)
        # ---
        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.0
        self.map = np.clip(self.map, 0, 1)
        self.map_resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion(
            (origin_o.x, origin_o.y, origin_o.z, origin_o.w)
        )
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5,
        )  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
