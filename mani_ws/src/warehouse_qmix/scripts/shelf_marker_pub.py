#!/usr/bin/env python3
"""发布货架和投递区的 MarkerArray，让 RViz 可视化仓库布局。"""

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

SHELVES = [
    (-1.5,  6.5), (1.5,  6.5),
    (-8.5,  1.5), (-5.5, 1.5),
    (-8.5, -0.5), (-5.5,-0.5),
    ( 5.5,  1.5), ( 8.5, 1.5),
    ( 5.5, -0.5), ( 8.5,-0.5),
]

DELIVERIES = [
    (-2.5, -5.5, (0.15, 0.80, 0.15), 'A'),
    (-0.5, -5.5, (0.90, 0.85, 0.10), 'B'),
    ( 1.5, -5.5, (0.15, 0.15, 0.85), 'C'),
]


def make_shelf_marker(mid, x, y):
    m = Marker()
    m.header.frame_id = 'map'
    m.ns = 'shelves'
    m.id = mid
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0.3
    m.pose.orientation.w = 1.0
    m.scale.x = 0.4
    m.scale.y = 0.4
    m.scale.z = 0.6
    m.color.r = 0.55
    m.color.g = 0.27
    m.color.b = 0.07
    m.color.a = 0.85
    m.lifetime = rospy.Duration(0)
    return m


def make_shelf_label(mid, x, y, idx):
    m = Marker()
    m.header.frame_id = 'map'
    m.ns = 'shelf_labels'
    m.id = mid
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0.8
    m.pose.orientation.w = 1.0
    m.scale.z = 0.3
    m.color.r = 1.0
    m.color.g = 1.0
    m.color.b = 1.0
    m.color.a = 1.0
    m.text = str(idx)
    m.lifetime = rospy.Duration(0)
    return m


def make_delivery_marker(mid, x, y, rgb, label):
    m = Marker()
    m.header.frame_id = 'map'
    m.ns = 'deliveries'
    m.id = mid
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0.001
    m.pose.orientation.w = 1.0
    m.scale.x = 0.8
    m.scale.y = 0.8
    m.scale.z = 0.002
    m.color.r = rgb[0]
    m.color.g = rgb[1]
    m.color.b = rgb[2]
    m.color.a = 0.9
    m.lifetime = rospy.Duration(0)

    t = Marker()
    t.header.frame_id = 'map'
    t.ns = 'delivery_labels'
    t.id = mid + 100
    t.type = Marker.TEXT_VIEW_FACING
    t.action = Marker.ADD
    t.pose.position.x = x
    t.pose.position.y = y
    t.pose.position.z = 0.4
    t.pose.orientation.w = 1.0
    t.scale.z = 0.35
    t.color.r = 1.0
    t.color.g = 1.0
    t.color.b = 1.0
    t.color.a = 1.0
    t.text = label
    t.lifetime = rospy.Duration(0)
    return m, t


def main():
    rospy.init_node('shelf_marker_pub', anonymous=False)
    pub = rospy.Publisher('/warehouse/shelf_markers', MarkerArray, queue_size=1, latch=True)

    ma = MarkerArray()
    for i, (x, y) in enumerate(SHELVES):
        ma.markers.append(make_shelf_marker(i, x, y))
        ma.markers.append(make_shelf_label(i + 50, x, y, i))

    for i, (x, y, rgb, label) in enumerate(DELIVERIES):
        box, txt = make_delivery_marker(i, x, y, rgb, label)
        ma.markers.append(box)
        ma.markers.append(txt)

    rospy.sleep(0.5)
    pub.publish(ma)
    rospy.loginfo('[shelf_marker_pub] 已发布 %d 个标记', len(ma.markers))
    rospy.spin()


if __name__ == '__main__':
    main()
