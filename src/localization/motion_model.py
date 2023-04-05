#!/usr/bin/env python2

import random
import numpy as np
import rospy


class MotionModel:

    def __init__(self):
        self.is_deterministic = rospy.get_param("~deterministic")

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
        particles = np.array(particles)
        noisy_odometry = self.noisy_odometry(odometry)
        result = self.apply_odometry(noisy_odometry, particles)

        # result = []
        # for particle in particles:
        #     result.append(self.apply_odometry(
        #         self.noisy_odometry(odometry), particle))
        """ result = np.apply_along_axis(
            self.apply_odometry, 1, particles, odometry) """
        return np.array(result)

    def noisy_odometry(self, odometry):
        if self.is_deterministic:
            return odometry
        else:
            dx = odometry[0] + random.gauss(mu=0.0, sigma=0.01)
            dy = odometry[1] + random.gauss(mu=0.0, sigma=0.01)
            dtheta = odometry[2] + random.gauss(mu=0.0, sigma=0.008)

            return [dx, dy, dtheta]

    # def apply_odometry(self, odometry, particle):
    #     #for a single particle
    #     theta = particle[2]
    #     sin_val = np.sin(theta)
    #     cos_val = np.cos(theta)

    #     matrix = np.matrix([[cos_val, -sin_val, 0.0],
    #                         [sin_val, cos_val, 0.0],
    #                         [0.0, 0.0, 1.0]])

    #     result = np.dot(matrix, odometry) + np.array(particle)

    #     result = [result[0, 0], result[0, 1], result[0, 2]]

    #     return result

    def apply_odometry(self, odometry, particles):
        #for all particles as a numpy array 
        theta = particles[:, 2]
        sin_val = np.sin(theta)
        cos_val = np.cos(theta)

        matrix = np.stack((cos_val, -sin_val, np.zeros_like(theta)), axis=1)
        matrix = np.concatenate((matrix, np.stack((sin_val, cos_val, np.zeros_like(theta)), axis=1)), axis=0)
        matrix = np.concatenate((matrix, np.stack((np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta)), axis=1)), axis=0)

        result = np.dot(matrix, odometry.T).T + particles

        return result


