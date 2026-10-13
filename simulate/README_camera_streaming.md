# MuJoCo Camera Streaming

The C++ simulator publishes its rendered head camera as a
[TeleImager](https://github.com/nmarticorena/teleimager)-compatible JPEG stream
over ZeroMQ. Camera images are not sent through DDS; DDS remains dedicated to
robot control and low-bandwidth state.

## Configuration

Edit `simulate/config.yaml`:

```yaml
publish_camera: 1
camera_name: "head_camera"
camera_width: 640
camera_height: 480
camera_rate_hz: 15
camera_zmq_port: 55555
camera_request_port: 60000
camera_jpeg_quality: 85
```

The selected MuJoCo model must contain a fixed camera with the configured name.
The G1 model provides `head_camera`.

Install dependencies and run the simulator:

```bash
pixi install
pixi run sim
```

The camera can also be enabled without editing the file:

```bash
pixi run ./simulate/build/unitree_mujoco --publish_camera 1
```

## TeleImager client

The server exposes:

- JPEG PUB stream on `tcp://<simulator-host>:55555`;
- camera configuration REP service on `tcp://<simulator-host>:60000`.

The REP service allows TeleImager's high-level `ImageClient` to discover the
manually configured resolution, rate, and stream port. With TeleImager
installed, run:

```bash
teleimager-client --host 127.0.0.1
```

Or use its API:

```python
from teleimager.image_client import ImageClient

client = ImageClient(host="127.0.0.1", request_bgr=True)
frame = client.get_head_frame()
if frame.bgr is not None:
    print(frame.bgr.shape)
```

A plain ZMQ subscriber can connect directly if configuration discovery is not
needed. Each message is one complete JPEG byte array with no topic prefix or
custom header:

```python
import cv2
import numpy as np
import zmq

context = zmq.Context.instance()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.RCVHWM, 1)
socket.setsockopt_string(zmq.SUBSCRIBE, "")
socket.connect("tcp://127.0.0.1:55555")

while True:
    image = cv2.imdecode(np.frombuffer(socket.recv(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("MuJoCo head camera", image)
    if cv2.waitKey(1) == ord("q"):
        break
```

## Implementation

`CameraZmqServer` in `simulate/src/main.cc` owns the OpenGL context, JPEG
encoder, PUB socket, and REP socket on one worker thread. It:

- renders only RGB, with no depth readback or camera-pose packing;
- reads directly into a reusable RGB buffer;
- uses TurboJPEG's bottom-up input flag instead of copying and flipping rows;
- encodes into a reusable allocation;
- sets `ZMQ_SNDHWM=1` and sends non-blocking, so slow clients receive current
  frames rather than building latency;
- returns disabled wrist-camera entries for compatibility with TeleImager's
  command-line client.

JPEG quality controls the CPU, bandwidth, and image-quality tradeoff. Start at
`85`; lower it if encoding or network use is too high.

## Limitations and alternatives

This minimal protocol carries RGB only. It deliberately omits depth, camera
pose, intrinsics, sequence numbers, and simulation timestamps to remain
compatible with TeleImager's existing JPEG subscriber.

For deterministic ArUco tests, use MuJoCo body/geom poses as ground truth. For
sim-to-real perception testing, run the actual ArUco detector as an external
TeleImager/ZMQ client. If synchronized depth or dataset metadata becomes a
requirement, add a separate versioned stream rather than changing the RGB JPEG
message.

Other useful transports are shared memory for same-host raw RGB/depth, WebRTC
through TeleImager for browser or VR viewing, and GStreamer for hardware video
encoding. They add complexity and are not part of this basic implementation.
