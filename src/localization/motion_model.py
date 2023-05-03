#!/usr/bin/env python2

import random
import numpy as np
import rospy


class MotionModel:
    def __init__(self):
        self.odometry = None

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        self.odometry = odometry
        if rospy.get_param("~deterministic"):
            return np.apply_along_axis(
                self.apply_deterministic_odometry, 1, np.array(particles)
            )
        else:
            return np.apply_along_axis(self.apply_odometry, 1, np.array(particles))

    def noisy_odometry(self):
        """return an odometry reading with noise applied if not deterministic, otherwise return the base odometry reading

        Returns:
            array: odometry reading
        """
        dx = self.odometry[0] + random.gauss(mu=0.0, sigma=0.04)
        dy = self.odometry[1] + random.gauss(mu=0.0, sigma=0.04)
        dtheta = self.odometry[2] + random.gauss(mu=0.0, sigma=0.08)

        return [dx, dy, dtheta]

    def apply_odometry(self, particle, odometry):
        """for a single particle apply an odometry reading to it

        Returns:
            array: particle
        """
        theta = particle[2]
        sin_val = np.sin(theta)
        cos_val = np.cos(theta)

        matrix = np.matrix(
            [[cos_val, -sin_val, 0.0], [sin_val, cos_val, 0.0], [0.0, 0.0, 1.0]]
        )

        return np.dot(matrix, self.noisy_odometry()) + np.array(particle)

    def apply_deterministic_odometry(self, particle, odometry):
        """for a single particle apply an odometry reading to it

        Returns:
            array: particle
        """
        theta = particle.reshape((3,))[2]
        sin_val = np.sin(theta)
        cos_val = np.cos(theta)

        matrix = np.matrix(
            [[cos_val, -sin_val, 0.0], [sin_val, cos_val, 0.0], [0.0, 0.0, 1.0]]
        )

        return np.dot(matrix, self.odometry) + np.array(particle)
