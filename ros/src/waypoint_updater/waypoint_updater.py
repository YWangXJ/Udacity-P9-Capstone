#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np
import math
from std_msgs.msg import Int32

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5
MAX_VEL = 20  # Speed limit mph
ONE_MPH = 0.444704
STOP_DIST = 5.0


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_lane = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1
        self.max_vel = MAX_VEL * ONE_MPH  # mph to m/s

        self.driving = False  # bool to see if car need accelerate or decelerate

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                # get closest way point
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        # rospy.logwarn("stopline_wp_idx: {0}".format(self.stopline_wp_idx))

        # if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
        #     lane.waypoints = base_waypoints
        # else:
        #     lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)


        # detect red light ahead
        if self.red_ahead(closest_idx, farthest_idx):
            lane = self.decelerate_waypoints(closest_idx)
            self.driving = False

        #  if stopped and no red light ahead
        elif not self.driving:
            self.driving = True

        # if driving
        if self.driving:
            lane.waypoints = self.accelerate_waypoints(closest_idx, farthest_idx)

        return lane

    def red_ahead(self, closest_idx, farthest_idx):
        if 0 < self.stopline_wp_idx <= farthest_idx and self.stopline_wp_idx >= closest_idx:
            return True

        return False

    def accelerate_waypoints(self, closest_idx, farthest_idx):
        lane = self.base_lane.waypoints[closest_idx:farthest_idx]

        for i in range(len(lane)):
            vel = self.get_waypoint_velocity(lane, i)
            vel = min(vel + (i+1) * 0.05, self.max_vel)
            self.set_waypoint_velocity(lane, i, vel)

        return lane

    def decelerate_waypoints(self, closest_idx):
        # two waypoints back from line so front of the car is behind the stop line
        stop_idx = self.stopline_wp_idx - closest_idx -2
        lane = self.base_lane.waypoints[closest_idx:closest_idx + stop_idx + 1]

        # set vel for each waypoints in lane
        for i in range(len(lane)):
            if i >= stop_idx:
                self.set_waypoint_velocity(lane, i, 0)
            else:
                dist = self.distance(lane, i, stop_idx)
                dist = max(0, dist - STOP_DIST)
                vel = min(math.sqrt(2 * MAX_DECEL * dist), self.max_vel)

                if vel < 1.5:
                    vel = 0.0
                    self.set_waypoint_velocity(lane, i, vel)

        return lane

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # rospy.loginfo('Constructing waypoint tree')
        self.base_lane = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
            # rospy.loginfo('Waypoint_tree data = %s', self.waypoint_tree)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint, idx):
        return waypoint[idx].twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
