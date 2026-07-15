"""FastAPI websocket bridge between a ROS RGB-D image stream and FoundationPose,
with SAM2 running as a *separate* process/service (e.g. sam2_rt_eroded.py) instead
of in this one.

This is the split-service counterpart to ros_pose_server.py. It exists because
sharing one Python process between SAM2 and FoundationPose is actually risky:
FoundationPose's Utils.compute_crop_window_tf_batch() calls
torch.set_default_tensor_type('torch.cuda.FloatTensor') and never restores it --
a process-wide side effect that corrupts SAM2's own (device-implicit)
torch.tensor(...) calls once any register()/track_one() has run. Running SAM2 in
its own process sidesteps that entirely, so this server has no SAM2 import, no
bfloat16 autocast, and no default-tensor-type workaround -- it only ever calls
FoundationPose, exactly like run_demo.py.

Wire protocol (one active ROS client at a time -- FoundationPose's tracker still
holds single-stream state):

  1. Connect to ws://<host>:<port>/ws/pose
  2. First message: JSON text with the camera intrinsics, e.g.
       {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0, "depth_scale": 1000.0}
  3. First binary frame (registration) MUST include a mask, packed as:
       [uint32 LE: len(color_bytes)] [color_bytes: JPEG]
       [uint32 LE: len(depth_bytes)] [depth_bytes: 16-bit single-channel PNG]
       [uint32 LE: len(mask_bytes)]  [mask_bytes: 8-bit single-channel PNG, 0/255]
     The client is expected to have gotten this mask from a separate SAM2
     service (or any other segmenter) already.
  4. Every binary frame after that (tracking) has no mask segment:
       [uint32 LE: len(color_bytes)] [color_bytes: JPEG]
       [uint32 LE: len(depth_bytes)] [depth_bytes: 16-bit single-channel PNG]
  5. Server replies with one JSON text message per frame:
       {"frame_idx": i, "success": true, "pose": [[4x4 row-major]], "t_ms": 12.3}
     or on failure:
       {"frame_idx": i, "success": false, "error": "...", "traceback": "..."}
     If registration (frame 0) fails, the connection is closed -- the client
     must reconnect and re-send a mask to try again.

Run (inside the FoundationPose conda/docker env):
    python ros_pose_server_split.py --mesh_file demo_data/mustard0/mesh/textured_simple.obj
"""

import argparse
import json
import struct
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
  return parser.parse_args()


args = parse_args()

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

app = FastAPI()


def _read_chunk(buf: bytes, offset: int) -> tuple:
  (n,) = FRAME_HEADER.unpack_from(buf, offset)
  offset += FRAME_HEADER.size
  chunk, offset = buf[offset:offset + n], offset + n
  return chunk, offset


def decode_frame(payload: bytes, expect_mask: bool):
  """Unpack [len][jpeg color][len][png16 depth](+[len][png mask] if expect_mask)."""
  offset = 0
  color_bytes, offset = _read_chunk(payload, offset)
  depth_bytes, offset = _read_chunk(payload, offset)

  color_bgr = cv2.imdecode(np.frombuffer(color_bytes, np.uint8), cv2.IMREAD_COLOR)
  rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
  depth_raw = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

  mask = None
  if expect_mask:
    mask_bytes, offset = _read_chunk(payload, offset)
    mask_raw = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    mask = mask_raw.astype(bool)

  return rgb, depth_raw, mask


@app.websocket('/ws/pose')
async def pose_stream(websocket: WebSocket):
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
      # 2. Receive one packed frame from ROS (mask included only on frame 0).
      payload = await websocket.receive_bytes()
      t0 = time.time()
      rgb, depth_raw, mask = decode_frame(payload, expect_mask=not if_init)
      depth = depth_raw.astype(np.float32) / depth_scale
      depth[depth < 0.001] = 0

      try:
        with torch.inference_mode():
          if not if_init:
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
        await websocket.send_text(json.dumps(reply))
      except Exception as e:
        tb = traceback.format_exc()
        logging.error(f'pose estimation failed on frame {frame_idx}\n{tb}')
        reply = {'frame_idx': frame_idx, 'success': False, 'error': str(e), 'traceback': tb}
        await websocket.send_text(json.dumps(reply))
        if not if_init:
          # Registration itself failed -- there is no mask to retry with on the
          # next binary message (the client only sends one on frame 0), so
          # force a clean reconnect instead of silently mis-decoding frame 1+
          # as if it were the still-missing registration frame.
          await websocket.close()
          return

      frame_idx += 1
      fps = 1.0 / max(time.time() - t0, 1e-6)
      logging.info(f'frame {frame_idx} processed in {(time.time() - t0) * 1000:.1f} ms | {fps:.1f} FPS')

  except WebSocketDisconnect:
    logging.info('ROS client disconnected')


if __name__ == '__main__':
  uvicorn.run(app, host=args.host, port=args.port)
