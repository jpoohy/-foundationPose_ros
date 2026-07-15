"""FastAPI websocket bridge between a ROS RGB-D image stream and FoundationPose.

Mirrors the pattern in segment-anything-2-real-time/sam2_rt_eroded.py, but instead
of returning a mask it runs full 6D pose estimation/tracking and returns the pose
to ROS. SAM2 is used internally, once, to get the mask that initializes
FoundationPose's register() on the first frame; every frame after that is handled
by FoundationPose's own tracker (est.track_one), so SAM2 is not run per-frame.

Wire protocol (one active ROS client at a time, same assumption as
sam2_rt_eroded.py -- the SAM2 camera predictor and the FoundationPose tracker both
hold single-stream state):

  1. Connect to ws://<host>:<port>/ws/pose?x=<px>&y=<py>
     x, y are pixel coords of a click prompt on the target object in frame 0
     (SAM2 point prompt). Both default to the image center if omitted.
  2. First message on the socket: JSON text with the camera intrinsics, e.g.
       {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0, "depth_scale": 1000.0}
     depth_scale converts the raw uint16 depth PNG values to meters
     (1000.0 for millimeter-encoded depth, the common RealSense/ROS convention).
  3. Every frame after that: one binary message packed as
       [uint32 LE: len(color_bytes)] [color_bytes: JPEG, BGR-as-encoded-by-cv2]
       [uint32 LE: len(depth_bytes)] [depth_bytes: 16-bit single-channel PNG]
  4. Server replies with one JSON text message per frame:
       {"frame_idx": i, "success": true, "pose": [[4x4 row-major]], "t_ms": 12.3}
     or on failure:
       {"frame_idx": i, "success": false, "error": "..."}

Run (inside the FoundationPose conda/docker env, with the sam2 package installed
-- `pip install -e .` in segment-anything-2-real-time):
    python ros_pose_server.py --mesh_file demo_data/mustard0/mesh/textured_simple.obj
"""

import argparse
import json
import struct
import sys
import time
import traceback

from estimater import *  # ScorePredictor, PoseRefinePredictor, FoundationPose, dr, trimesh, np, cv2, logging, set_logging_format, set_seed, ...
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

FRAME_HEADER = struct.Struct('<I')  # one little-endian uint32 length prefix


def parse_args():
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/021_bleach_cleanser/google_16k/textured.obj')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--host', type=str, default='0.0.0.0')
  parser.add_argument('--port', type=int, default=8001)
  parser.add_argument('--sam2_checkpoint', type=str, default='/home/zsj/segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt')
  parser.add_argument('--sam2_config', type=str, default='configs/sam2.1/sam2.1_hiera_t.yaml')
  parser.add_argument('--sam2_repo_dir', type=str, default='/home/zsj/segment-anything-2-real-time',
                       help='Added to sys.path so `import sam2` resolves even if the package is not pip-installed in this env.')
  parser.add_argument('--erosion_kernel_size', type=int, default=5)
  parser.add_argument('--erosion_iterations', type=int, default=6)
  return parser.parse_args()


args = parse_args()
sys.path.insert(0, args.sam2_repo_dir)
from sam2.build_sam import build_sam2_camera_predictor  # noqa: E402  (needs sys.path patched first)

set_logging_format()
set_seed(0)

logging.info(f'loading mesh {args.mesh_file}')
mesh = trimesh.load(args.mesh_file)
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                      scorer=scorer, refiner=refiner, debug=0, glctx=glctx)
logging.info('FoundationPose estimator ready')

logging.info('loading SAM2 real-time predictor')
sam2_predictor = build_sam2_camera_predictor(args.sam2_config, args.sam2_checkpoint)
_erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erosion_kernel_size, args.erosion_kernel_size))
logging.info('SAM2 ready')

app = FastAPI()


def _recv_exact(buf: bytes, offset: int, n: int) -> tuple:
  return buf[offset:offset + n], offset + n


def decode_frame(payload: bytes):
  """Unpack one [len][jpeg color][len][png16 depth] binary message."""
  offset = 0
  (color_len,) = FRAME_HEADER.unpack_from(payload, offset)
  offset += FRAME_HEADER.size
  color_bytes, offset = _recv_exact(payload, offset, color_len)
  (depth_len,) = FRAME_HEADER.unpack_from(payload, offset)
  offset += FRAME_HEADER.size
  depth_bytes, offset = _recv_exact(payload, offset, depth_len)

  color_bgr = cv2.imdecode(np.frombuffer(color_bytes, np.uint8), cv2.IMREAD_COLOR)
  rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

  depth_raw = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
  return rgb, depth_raw


def get_mask_from_sam2(rgb, px, py):
  sam2_predictor.load_first_frame(rgb)
  points = np.array([[px, py]], dtype=np.float32)
  labels = np.array([1], dtype=np.int32)
  _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, points=points, labels=labels)
  mask_bool = (out_mask_logits[0][0] > 0.0).cpu().numpy()
  mask_uint8 = mask_bool.astype(np.uint8) * 255
  mask_uint8 = cv2.erode(mask_uint8, _erosion_kernel, iterations=args.erosion_iterations)
  return mask_uint8.astype(bool)


@app.websocket('/ws/pose')
async def pose_stream(websocket: WebSocket, x: float = -1.0, y: float = -1.0):
  await websocket.accept()

  # 1. Handshake: camera intrinsics as the first message.
  intrinsics_raw = await websocket.receive_text()
  intrinsics = json.loads(intrinsics_raw)
  K = np.array([
    [intrinsics['fx'], 0, intrinsics['cx']],
    [0, intrinsics['fy'], intrinsics['cy']],
    [0, 0, 1],
  ], dtype=np.float64)
  depth_scale = float(intrinsics.get('depth_scale', 1000.0))
  logging.info(f'K handshake received, depth_scale={depth_scale}')

  if_init = False
  frame_idx = 0

  try:
    while True:
      # 2. Receive one packed RGB-D frame from ROS.
      payload = await websocket.receive_bytes()
      t0 = time.time()
      rgb, depth_raw = decode_frame(payload)
      depth = depth_raw.astype(np.float32) / depth_scale
      depth[depth < 0.001] = 0

      try:
        with torch.inference_mode():
          if not if_init:
            h, w = rgb.shape[:2]
            px = x if x > 0 else w / 2
            py = y if y > 0 else h / 2
            # FoundationPose's register()/track_one() call
            # Utils.compute_crop_window_tf_batch(), which does
            # torch.set_default_tensor_type('torch.cuda.FloatTensor') and never
            # restores it -- a process-wide side effect that leaks into SAM2's
            # own tensor creation (torch.tensor(...) with no explicit device)
            # once any register()/track_one() call has happened in this
            # process. Force it back to the normal CPU-by-default behavior
            # before every SAM2 call so SAM2's from_numpy (always CPU) and
            # torch.tensor (device-implicit) calls stay on the same device.
            torch.set_default_tensor_type('torch.FloatTensor')
            # bfloat16 autocast is only for SAM2's forward pass; FoundationPose's
            # register()/track_one() do their own internal linalg (e.g.
            # torch.linalg.inv) which does not support bfloat16 inputs, so they
            # must run outside this autocast context.
            with torch.autocast('cuda', dtype=torch.bfloat16):
              mask = get_mask_from_sam2(rgb, px, py)
            pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            if_init = True
          else:
            pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_refine_iter)

        reply = {
          'frame_idx': frame_idx,
          'success': True,
          'pose': pose.reshape(4, 4).tolist(),
          't_ms': (time.time() - t0) * 1000.0,
        }
      except Exception as e:
        tb = traceback.format_exc()
        logging.error(f'pose estimation failed on frame {frame_idx}\n{tb}')
        if not if_init:
          # register() never completed: SAM2's stream state (load_first_frame /
          # add_new_prompt already ran) is now stale, so reset it before the
          # next frame retries initialization, or the retry fails differently
          # (e.g. device-mismatched cached tensors) instead of cleanly retrying.
          sam2_predictor.reset_state()
        # Send the full traceback back to ROS too, so it shows up in the
        # client's log without needing to check this process's own console.
        reply = {'frame_idx': frame_idx, 'success': False, 'error': str(e), 'traceback': tb}

      # 3. Send the pose back to ROS.
      await websocket.send_text(json.dumps(reply))

      frame_idx += 1
      fps = 1.0 / max(time.time() - t0, 1e-6)
      logging.info(f'frame {frame_idx} processed in {(time.time() - t0) * 1000:.1f} ms | {fps:.1f} FPS')

  except WebSocketDisconnect:
    logging.info('ROS client disconnected')


if __name__ == '__main__':
  uvicorn.run(app, host=args.host, port=args.port)
