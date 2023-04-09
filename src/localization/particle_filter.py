#!/usr/bin/env python2

import rospy
import tf
import tf2_ros
from sensor_model import SensorModel
from motion_model import MotionModel
import message_filters

# import random
import numpy as np
# from scipy.stats import circmean
# from astropy.stats import circmean

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped


class ParticleFilter:

    def __init__(self):
        self.prev_time = rospy.Time.now()

        # how many particles we're using
        self.num_particles = rospy.get_param("~num_particles")

        # our particles list
        self.particles = np.array([[0.0, 0.0, 0.0]
                                  for _ in range(self.num_particles)])

        # self.weights = [ 1.0 / self.num_particles for _ in range(self.num_particles)]

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Get parameters
        self.particle_filter_frame = \
            rospy.get_param("~particle_filter_frame")

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")

        self.laser_sub = message_filters.Subscriber(scan_topic, LaserScan)
        self.odom_sub = message_filters.Subscriber(odom_topic, Odometry)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.laser_sub, self.odom_sub], 10, 0.05)
        ts.registerCallback(self.lidar_odometry_callback)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                         self.pose_callback,  # TODO: Fill this in
                                         queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub = rospy.Publisher(
            "/pf/pose/odom", Odometry, queue_size=1)

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.map_tf = tf2_ros.TransformBroadcaster()

    def lidar_odometry_callback(self, laser_msg, odom_msg):
        """
        Propagate motion model with odometry information.
        """
        time = rospy.Time.now()
        dt = (time - self.prev_time).to_sec()
        self.prev_time = time

        """ self.particles = self.motion_model.evaluate(
            self.particles, [(msg.twist.twist.linear.x * dt) + random.gauss(mu=0.0, sigma=0.1),
                             (msg.twist.twist.linear.y * dt) +
                             random.gauss(mu=0.0, sigma=0.1),
                             (msg.twist.twist.angular.z * dt) + random.gauss(mu=0.0, sigma=0.08)]) """

        self.particles = self.motion_model.evaluate(
            self.particles, [(odom_msg.twist.twist.linear.x * dt),
                             (odom_msg.twist.twist.linear.y * dt),
                             (odom_msg.twist.twist.angular.z * dt)])
        """
        Resample and duplicate 1/f of the particles according to sensor model probabilities.
        """

        f = 2
        n = np.rint(self.num_particles/f).astype(int)
        p = self.sensor_model.evaluate(
            self.particles, laser_msg.ranges)  # msg.ranges[20:980]
        # test this to see if it gets rid of weird data on rviz
        if p is None:
            return  # Map is not set!

        p /= np.sum(p)

        self.average_pose(p)

        indices = np.array(range(0, len(self.particles)))
        resampled_indices = np.random.choice(
            indices, size=n, replace=False, p=p)

        resampled = self.particles[resampled_indices]
        self.particles = np.repeat(resampled, f, axis=0)

    def average_pose(self, probabilities):
        """
        Take self.particles and return an Odometry message with the average pose.
        """
        x = np.average(self.particles[:, 0], probabilities)
        y = np.average(self.particles[:, 1], probabilities)
        theta = circular_mean(self.particles[:, 2], probabilities)

        result = Odometry()

        result.header.stamp = rospy.Time.now()
        result.header.frame_id = self.particle_filter_frame

        result.pose.pose.position.x = x
        result.pose.pose.position.y = y
        result.pose.pose.position.z = 0.0
        quat = tf.transformations.quaternion_from_euler(0.0, 0.0, theta)
        result.pose.pose.orientation.x = quat[0]
        result.pose.pose.orientation.y = quat[1]
        result.pose.pose.orientation.z = quat[2]
        result.pose.pose.orientation.w = quat[3]

        self.odom_pub.publish(result)
        self.map_transform(result)
        return result

    def lidar_callback(self, msg):
        """
        Resample and duplicate 1/f of the particles according to sensor model probabilities.
        """
        f = 2
        n = np.rint(self.num_particles/f).astype(int)
        p = self.sensor_model.evaluate(
            self.particles, msg.ranges)  # msg.ranges[20:980]
        # test this to see if it gets rid of weird data on rviz
        if p is None:
            return  # Map is not set!

        p /= np.sum(p)

        self.weights = p

        self.average_pose(self.weights)

        indices = np.array(range(0, len(self.particles)))
        resampled_indices = np.random.choice(
            indices, size=n, replace=False, p=p)

        resampled = self.particles[resampled_indices]
        self.particles = np.repeat(resampled, f, axis=0)

    def odometry_callback(self, msg):
        """
        Propagate motion model with odometry information.
        """
        time = rospy.Time.now()
        dt = (time - self.prev_time).to_sec()

        """ self.particles = self.motion_model.evaluate(
            self.particles, [(msg.twist.twist.linear.x * dt) + random.gauss(mu=0.0, sigma=0.1),
                             (msg.twist.twist.linear.y * dt) +
                             random.gauss(mu=0.0, sigma=0.1),
                             (msg.twist.twist.angular.z * dt) + random.gauss(mu=0.0, sigma=0.08)]) """

        self.particles = self.motion_model.evaluate(
            self.particles, [(msg.twist.twist.linear.x * dt),
                             (msg.twist.twist.linear.y * dt),
                             (msg.twist.twist.angular.z * dt)])

        self.average_pose(self.weights)
        self.prev_time = time

    def pose_callback(self, msg):
        """
        Set initial pose using rviz 2D pose estimate.
        """
        pose = msg.pose.pose
        theta = tf.transformations.euler_from_quaternion(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])[2]
        self.particles = np.array([[pose.position.x, pose.position.y, theta]
                                   for _ in range(self.num_particles)])

        # self.weights = [1.0 / self.num_particles for _ in range(self.num_particles)]

    def map_transform(self, msg):
        """
        Broadcast a dynamic transform for the map onto the estimated pose.
        """
        pose = msg.pose.pose
        tr = TransformStamped()
        tr.header.stamp = rospy.Time.now()
        tr.header.frame_id = self.particle_filter_frame
        tr.child_frame_id = "/map"

        tr.transform.translation.x = pose.position.x
        tr.transform.translation.y = pose.position.y
        tr.transform.translation.z = pose.position.z

        tr.transform.rotation.x = pose.orientation.x
        tr.transform.rotation.y = pose.orientation.y
        tr.transform.rotation.z = pose.orientation.z
        tr.transform.rotation.w = pose.orientation.w

        self.map_tf.sendTransform(tr)


def circular_mean(angles, weights=None):

    # https://en.wikipedia.org/wiki/Circular_mean

    if weights is None:
        weights = np.ones(len(angles))

    vectors = [[w*np.cos(a), w*np.sin(a)] for a, w in zip(angles, weights)]

    vector = np.sum(vectors, axis=0) / np.sum(weights)

    x, y = vector

    angle_mean = np.arctan2(y, x)
    angle_variance = 1. - np.linalg.norm(vector)  # x*2+y*2 = hypot(x,y)

    return angle_mean


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
