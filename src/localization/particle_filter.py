#!/usr/bin/env python2

import rospy
import tf
from sensor_model import SensorModel
from motion_model import MotionModel

import random
import numpy as np
from scipy.stats import circmean

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

        # average pose of our particles in Odometry.pose
        self.pose_estimate = Odometry()

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

        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.lidar_callback,  # TODO: Fill this in
                                          queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry,
                                         self.odometry_callback,  # TODO: Fill this in
                                         queue_size=1)

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
        self.map_tf = tf.TransformBroadcaster()


    # TODO

    def average_pose(self):
        # take self.particles and find the average pose, return an Odometry message with the pose
        x = np.mean(self.particles[:, 0])
        y = np.mean(self.particles[:, 1])
        theta = circmean(self.particles[:, 2])

        result = Odometry()

        result.header.frame_id = self.particle_filter_frame

        result.pose.pose.position.x = x
        result.pose.pose.position.y = y
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
        f = 10
        n = np.rint(self.num_particles/f).astype(int)
        p = self.sensor_model.evaluate(self.particles, msg.ranges[20:980]) # msg.ranges[20:980] 
                                                                   # test this to see if it gets rid of weird data on rviz
        if p is None:
            return  # Map is not set!

        p /= np.sum(p)
        indices = np.array(range(0, len(self.particles)))
        resampled_indices = np.random.choice(
            indices, size=n, replace=False, p=p)

        resampled = self.particles[resampled_indices]
        self.particles = np.repeat(resampled, f, axis=0)

        self.pose_estimate = self.average_pose()

    def odometry_callback(self, msg):
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

        self.pose_estimate = self.average_pose()
        self.prev_time = time

    def pose_callback(self, msg):
        rospy.loginfo("got a pose!")
        pose = msg.pose.pose
        theta = tf.transformations.euler_from_quaternion(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])[2]
        self.particles = np.array([[pose.position.x, pose.position.y, theta]
                                   for _ in range(self.num_particles)])

        self.pose_estimate.pose = msg.pose

    def map_transform(self, msg):
        """
        Broadcast a dynamic transform for the map onto the estimated pose.
        """
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.particle_filter_frame
        transform.child_frame_id = "/map"

        pose = msg.pose.pose
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation.x = pose.orientation.x
        transform.transform.rotation.y = pose.orientation.y
        transform.transform.rotation.z = pose.orientation.z
        transform.transform.rotation.w = pose.orientation.w

        #---
        self.map_tf.sendTransform((pose.position.x, pose.position.y, 0),
            (pose.orientation.x, pose.orientation.y, pose.orientation.z, -pose.orientation.w), # 180 deg tf
            rospy.Time.now(), "/map", "/base_link")


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
