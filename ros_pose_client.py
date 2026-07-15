#!/usr/bin/env python3
"""ROS1 bridge node: subscribes to synchronized RGB-D + CameraInfo, forwards each
frame to ros_pose_server.py over a websocket, and republishes the returned 6D pose
as a geometry_msgs/PoseStamped + a TF transform.

Wire protocol matches ros_pose_server.py exactly:
  1. Connect, then send one JSON text handshake with intrinsics.
  2. For every frame: send [uint32 LE color_len][JPEG][uint32 LE depth_len][PNG16].
  3. Receive one JSON text reply per frame with the 4x4 pose (or an error).
The connection is a strict request/response loop (send frame, wait for reply,
send next frame) -- this node uses the synchronous `websocket-client` library
(`pip install websocket-client`) rather than asyncio, since rospy callbacks are
themselves synchronous.

Topic names / intrinsics fallback / server URL are all rospy params so they can be
set from a launch file without touching this script:
    ~color_topic        (default /camera/color/image_raw)
    ~depth_topic        (default /camera/depth/image_raw, must be pixel-aligned to color)
    ~camera_info_topic  (default /camera/color/camera_info)
    ~server_url         (default ws://localhost:8001/ws/pose)
    ~prompt_x, ~prompt_y (default -1 -> image center; SAM2 first-frame click prompt)
    ~depth_scale        (default 1000.0, i.e. depth image is uint16 millimeters)
    ~object_frame       (default "object", TF child frame published for the pose)
    ~camera_frame       (default "", TF parent frame; falls back to the depth image's header.frame_id)
"""

import json
import struct
import threading

import cv2
import numpy as np
import rospy
import tf2_ros
import websocket
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CameraInfo, Image
from tf.transformations import quaternion_from_matrix

FRAME_HEADER = struct.Struct('<I')


def pack_frame(color_bgr, depth_uint16):
  ok, color_enc = cv2.imencode('.jpg', color_bgr)
  if not ok:
    raise RuntimeError('failed to JPEG-encode color frame')
  ok, depth_enc = cv2.imencode('.png', depth_uint16)
  if not ok:
    raise RuntimeError('failed to PNG-encode depth frame')
  color_bytes = color_enc.tobytes()
  depth_bytes = depth_enc.tobytes()
  return FRAME_HEADER.pack(len(color_bytes)) + color_bytes + FRAME_HEADER.pack(len(depth_bytes)) + depth_bytes


class PoseBridge:
  def __init__(self):
    self.bridge = CvBridge()
    self.tf_broadcaster = tf2_ros.TransformBroadcaster()
    self.pose_pub = rospy.Publisher('~pose', PoseStamped, queue_size=1)

    self.server_url = rospy.get_param('~server_url', 'ws://localhost:8001/ws/pose')
    self.prompt_x = rospy.get_param('~prompt_x', -1.0)
    self.prompt_y = rospy.get_param('~prompt_y', -1.0)
    self.depth_scale = rospy.get_param('~depth_scale', 1000.0)
    self.object_frame = rospy.get_param('~object_frame', 'object')
    self.camera_frame_override = rospy.get_param('~camera_frame', '')

    self.K = None  # filled in from the first CameraInfo message
    self.ws = None
    self.ws_lock = threading.Lock()
    self.connected = False

    color_topic = rospy.get_param('~color_topic', '/femto/color/image_raw')
    depth_topic = rospy.get_param('~depth_topic', '/femto/depth/image_raw')
    camera_info_topic = rospy.get_param('~camera_info_topic', '/femto/color/camera_info')

    rospy.loginfo(f'waiting for one CameraInfo message on {camera_info_topic}')
    info = rospy.wait_for_message(camera_info_topic, CameraInfo)
    self.K = {'fx': info.K[0], 'fy': info.K[4], 'cx': info.K[2], 'cy': info.K[5]}

    self.color_sub = Subscriber(color_topic, Image)
    self.depth_sub = Subscriber(depth_topic, Image)
    self.sync = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=5, slop=0.05)
    self.sync.registerCallback(self.on_frame)

    rospy.loginfo(f'FoundationPose bridge ready, color={color_topic} depth={depth_topic} server={self.server_url}')

  def connect(self):
    url = f'{self.server_url}?x={self.prompt_x}&y={self.prompt_y}'
    self.ws = websocket.create_connection(url, timeout=10)
    self.ws.send(json.dumps({
      'fx': self.K['fx'], 'fy': self.K['fy'], 'cx': self.K['cx'], 'cy': self.K['cy'],
      'depth_scale': self.depth_scale,
    }))
    self.connected = True
    rospy.loginfo('connected to ros_pose_server.py and sent intrinsics handshake')

  def on_frame(self, color_msg, depth_msg):
    with self.ws_lock:
      if not self.connected:
        try:
          self.connect()
        except Exception as e:
          rospy.logerr_throttle(5, f'could not connect to pose server: {e}')
          return

      color_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
      depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
      if depth.dtype != np.uint16:
        # e.g. depth published as float32 meters -- convert to the uint16-mm wire format
        depth = (depth.astype(np.float32) * self.depth_scale).astype(np.uint16)

      try:
        self.ws.send_binary(pack_frame(color_bgr, depth))
        reply = json.loads(self.ws.recv())
      except Exception as e:
        rospy.logerr(f'websocket send/recv failed, will reconnect on next frame: {e}')
        self.connected = False
        return

      if not reply.get('success'):
        rospy.logwarn(f'pose estimation failed: {reply.get("error")}\n{reply.get("traceback", "")}')
        return

      self.publish_pose(np.array(reply['pose']), depth_msg.header)

  def publish_pose(self, pose_mat, header):
    camera_frame = self.camera_frame_override or header.frame_id
    quat = quaternion_from_matrix(pose_mat)

    msg = PoseStamped()
    msg.header.stamp = header.stamp
    msg.header.frame_id = camera_frame
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose_mat[:3, 3]
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
    self.pose_pub.publish(msg)

    tf_msg = TransformStamped()
    tf_msg.header = msg.header
    tf_msg.child_frame_id = self.object_frame
    tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z = pose_mat[:3, 3]
    tf_msg.transform.rotation = msg.pose.orientation
    self.tf_broadcaster.sendTransform(tf_msg)


if __name__ == '__main__':
  rospy.init_node('foundationpose_bridge')
  PoseBridge()
  rospy.spin()
