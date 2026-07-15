#!/usr/bin/env python3
"""ROS1 bridge node for the split SAM2 / FoundationPose setup: the frame-0 mask
comes from a ROS topic (published by whatever SAM2 process you run yourself,
independently of this node) instead of this node calling SAM2 directly. This
node only talks to ros_pose_server_split.py -- it has no notion of SAM2 at all.

Sequence:
  1. Get the frame-0 mask, either:
       - ~mask_file: read once from disk at startup (e.g. a YCB-style
         demo_data/<obj>/mask/000000.png -- same file run_demo.py's
         YcbineoatReader.get_mask(0) would read), reused for every
         (re-)registration; or, if ~mask_file is not set,
       - ~mask_topic: block for one message (mono8, nonzero = foreground) each
         time a mask is needed. You are responsible for making sure it
         corresponds to the object pose in the next synchronized color/depth
         frame this node picks up.
  2. Connect to ros_pose_server_split.py, send the intrinsics handshake, then
     send the next synchronized color+depth+that mask as the registration frame.
  3. Every frame after that: send color+depth only -- FoundationPose tracks on
     its own from here.
  4. If a frame reply comes back with success=false while still registering,
     or the connection drops, this node goes back to step 1 (re-reads
     ~mask_file, or waits on ~mask_topic again) before retrying registration.

Wire protocol matches ros_pose_server_split.py exactly: JSON intrinsics
handshake, then per frame
    [uint32 LE color_len][JPEG][uint32 LE depth_len][PNG16]
    (+[uint32 LE mask_len][PNG mask] only while (re-)registering)
    -> one JSON pose reply per frame

rospy params (settable from a launch file, no code changes needed):
    ~color_topic        (default /camera/color/image_raw)
    ~depth_topic        (default /camera/depth/image_raw, must be pixel-aligned to color)
    ~camera_info_topic  (default /camera/color/camera_info)
    ~mask_file          (optional; path to a mask image read once at startup, e.g.
                         demo_data/021_bleach_cleanser/mask/000000.png -- takes
                         priority over ~mask_topic if set)
    ~mask_topic         (default /sam2/mask, mono8, published by your own SAM2 process;
                         only used if ~mask_file is not set)
    ~fp_url             (default ws://localhost:8001/ws/pose)
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


def pack_chunk(data: bytes) -> bytes:
  return FRAME_HEADER.pack(len(data)) + data


def encode_color(color_bgr) -> bytes:
  ok, enc = cv2.imencode('.jpg', color_bgr)
  if not ok:
    raise RuntimeError('failed to JPEG-encode color frame')
  return enc.tobytes()


def encode_depth(depth_uint16) -> bytes:
  ok, enc = cv2.imencode('.png', depth_uint16)
  if not ok:
    raise RuntimeError('failed to PNG-encode depth frame')
  return enc.tobytes()


def encode_mask(mask_bool) -> bytes:
  ok, enc = cv2.imencode('.png', (mask_bool.astype(np.uint8)) * 255)
  if not ok:
    raise RuntimeError('failed to PNG-encode mask')
  return enc.tobytes()


class PoseBridge:
  def __init__(self):
    self.bridge = CvBridge()
    self.tf_broadcaster = tf2_ros.TransformBroadcaster()
    self.pose_pub = rospy.Publisher('~pose', PoseStamped, queue_size=1)

    self.fp_url = rospy.get_param('~fp_url', 'ws://localhost:8001/ws/pose')
    self.mask_file = rospy.get_param('~mask_file', '')
    self.mask_topic = rospy.get_param('~mask_topic', '/sam2/mask')
    self.depth_scale = rospy.get_param('~depth_scale', 1000.0)

    self.static_mask = None
    if self.mask_file:
      m = cv2.imread(self.mask_file, -1)
      if m is None:
        raise RuntimeError(f'could not read ~mask_file {self.mask_file}')
      if m.ndim == 3:
        m = m[..., 0]
      self.static_mask = m > 0
      rospy.loginfo(f'loaded static registration mask from {self.mask_file}')
    self.object_frame = rospy.get_param('~object_frame', 'object')
    self.camera_frame_override = rospy.get_param('~camera_frame', '')

    self.K = None  # filled in from the first CameraInfo message
    self.fp_ws = None
    self.lock = threading.Lock()
    self.have_mask = False   # have we registered with FoundationPose yet?
    self.fp_connected = False

    color_topic = rospy.get_param('~color_topic', '/camera/color/image_raw')
    depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_raw')
    camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/color/camera_info')

    rospy.loginfo(f'waiting for one CameraInfo message on {camera_info_topic}')
    info = rospy.wait_for_message(camera_info_topic, CameraInfo)
    self.K = {'fx': info.K[0], 'fy': info.K[4], 'cx': info.K[2], 'cy': info.K[5]}

    self.color_sub = Subscriber(color_topic, Image)
    self.depth_sub = Subscriber(depth_topic, Image)
    self.sync = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=5, slop=0.05)
    self.sync.registerCallback(self.on_frame)

    rospy.loginfo(f'bridge ready, color={color_topic} depth={depth_topic} mask={self.mask_topic} fp={self.fp_url}')

  def get_mask(self):
    """~mask_file (if set) wins and is just reused every time a mask is
    needed. Otherwise block for one message on ~mask_topic -- whatever runs
    your SAM2 process is responsible for publishing there. Returns a bool
    array, or None if neither is available yet."""
    if self.static_mask is not None:
      return self.static_mask
    try:
      msg = rospy.wait_for_message(self.mask_topic, Image, timeout=1.0)
    except rospy.ROSException:
      return None
    mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
    return mask > 0

  def connect_fp(self):
    self.fp_ws = websocket.create_connection(self.fp_url, timeout=10)
    self.fp_ws.send(json.dumps({
      'fx': self.K['fx'], 'fy': self.K['fy'], 'cx': self.K['cx'], 'cy': self.K['cy'],
      'depth_scale': self.depth_scale,
    }))
    self.fp_connected = True
    rospy.loginfo('connected to ros_pose_server_split.py and sent intrinsics handshake')

  def on_frame(self, color_msg, depth_msg):
    with self.lock:
      color_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
      depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
      if depth.dtype != np.uint16:
        # e.g. depth published as float32 meters -- convert to the uint16-mm wire format
        depth = (depth.astype(np.float32) * self.depth_scale).astype(np.uint16)

      chunks = [pack_chunk(encode_color(color_bgr)), pack_chunk(encode_depth(depth))]

      if not self.have_mask:
        mask = self.get_mask()
        if mask is None:
          rospy.logwarn_throttle(5, f'no mask on {self.mask_topic} yet, waiting for your SAM2 process')
          return
        chunks.append(pack_chunk(encode_mask(mask)))

      try:
        if not self.fp_connected:
          self.connect_fp()
        self.fp_ws.send_binary(b''.join(chunks))
        reply = json.loads(self.fp_ws.recv())
      except Exception as e:
        rospy.logerr(f'websocket send/recv failed, will re-register on next frame: {e}')
        self.fp_connected = False
        self.have_mask = False
        return

      if not reply.get('success'):
        rospy.logwarn(f'pose estimation failed: {reply.get("error")}\n{reply.get("traceback", "")}')
        if not self.have_mask:
          # registration failed: ros_pose_server_split.py already closed the
          # connection, so reconnect and wait for a fresh mask next frame.
          self.fp_connected = False
        return

      self.have_mask = True
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
  rospy.init_node('foundationpose_bridge_split')
  PoseBridge()
  rospy.spin()
