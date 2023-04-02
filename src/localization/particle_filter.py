#!/usr/bin/env python2

import rospy
from sensor_model import SensorModel
from motion_model import MotionModel
from scipy.stats import circmean

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped


class ParticleFilter:

    def __init__(self):
        # how many particles we're using
        self.num_particles = rospy.get_param("~num_particles")

        # our particles list
        self.particles = [[0.0, 0.0, 0.0] for _ in range(self.num_particles)]

        # average pose of our particles in Odometry.pose
        self.pose_estimate = Odometry()

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

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    # TODO

    def average_pose(self):
        # take self.particles and find the average pose, return an Odometry message with the pose
        pass

    # TODO
    def lidar_callback(self, msg):
        pass

    def odometry_callback(self, msg):
        MotionModel.evaluate(
            self.particles, [msg.twist[0], msg.twist[1], msg.twist[5]])

        self.pose_estimate = self.average_pose()

    def pose_callback(self, msg):
        self.particles = [[msg.pose[0], msg.pose[1], msg.pose[5]]
                          for _ in range(self.num_particles)]

        self.pose_estimate.pose = msg


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
