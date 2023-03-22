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
        for particle in particles:
            particle = self.apply_odometry(
                self.noisy_odometry(odometry), particle)

        return particles

    def noisy_odometry(self, odometry):
        if self.is_deterministic:
            return odometry
        else:
            dx = odometry[0] + random.gauss(mu=0.0, sigma=0.5)
            dy = odometry[1] + random.gauss(mu=0.0, sigma=0.5)
            dtheta = odometry[2] + random.gauss(mu=0.0, sigma=0.5)

            return [dx, dy, dtheta]

    def apply_odometry(self, odometry, particle):
        theta = particle[2]
        sin_val = np.sin(theta)
        cos_val = np.cos(theta)

        matrix = np.matrix([[cos_val, -sin_val, 0.0],
                            [sin_val, cos_val, 0.0],
                            [0.0, 0.0, 1.0]])

        result = np.dot(matrix, odometry) + np.array(particle)

        result = [result[0, 0], result[0, 1], result[0, 2]]

        return result
