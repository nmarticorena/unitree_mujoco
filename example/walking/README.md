# walking_g1_mjlab observations

`walking_g1_mjlab.cpp` builds two targets from the same source
(`example/walking/CMakeLists.txt`):

| Target                     | `MJLAB_DOLLY_TASK` | Policy file(s)                        | `NUM_OBS` |
|-----------------------------|:---:|----------------------------------------|:---:|
| `walking_g1_mjlab`          | off | `mjlab.onnx`                           | 99  |
| `walking_g1_mjlab_dolly`    | on  | `mjlab.onnx` + `mjlab_dolly.onnx`       | 106 |

This note documents the observation vector the **dolly** build feeds to the
policy, since it extends the plain locomotion observation with 7 extra
values and changes one of the existing terms.

## Observation layout (`buildCurrentObservation`, dolly build, 106 values)

| Index range | Count | Term | Notes |
|---|---|---|---|
| 0–2   | 3  | Base linear velocity | Sport-mode world velocity rotated into the **pelvis/IMU frame** (`imu_quaternion_.conjugate() * world_velocity`). In the non-dolly build this term is left in world frame. |
| 3–5   | 3  | Angular velocity (gyro) | Raw IMU gyroscope, rad/s. |
| 6–8   | 3  | Projected gravity | Gravity vector rotated into the IMU frame. |
| 9–37  | 29 | Joint position error | `q - default_q` for each of the 29 policy joints, in DDS motor order. |
| 38–66 | 29 | Joint velocity | `dq` for each of the 29 policy joints. |
| 67–95 | 29 | Previous action | Raw (pre action-scale) action output on the last policy step. |
| 96–98 | 3  | Velocity command | `(vx, vy, wz)` from the joystick/`setCommand`. |
| 99–105 | 7 | **Dolly observation** | See below. Clipped to `[-3, 3]`, except index 99 (the availability flag) which is clipped to `[0, 1]`. |

(`3+3+3+3*29+3+7 = 106`, matching the `static_assert` in the source.)

The non-dolly build simply stops after the velocity command (`3+3+3+3*29+3 = 99`).

## The 7-value dolly payload

Published by `DollyObservationPublisher` in `simulate/src/main.cc` on
`rt/dolly_observation` (a `std_msgs/String`, space-separated floats), at 50 Hz,
and consumed by `DollyObservationMessageHandler` in the controller.

| Payload index | Meaning |
|---|---|
| 0 | **Availability flag.** Currently always `1.0` when the publisher is enabled (all required sites exist). The controller gates on this (`>= 0.5`) *and* a 250 ms staleness timeout (`DOLLY_OBSERVATION_TIMEOUT`) to decide whether cart-relative data is "available" — going stale (sim/DDS hiccup, publisher disabled) drops it to the locomotion-only policy. |
| 1–3 | Left palm → left cart-handle vector, expressed in the **pelvis** frame (`left_lego_centroid` site → `left_hand_target` site). |
| 4–6 | Right palm → right cart-handle vector, in the pelvis frame (`right_lego_centroid` site → `right_hand_target` site). |

Requires these MJCF sites to resolve on the loaded model, or the publisher
disables itself and logs a warning: `pelvis` (body), `left_lego_centroid`,
`right_lego_centroid`, `left_hand_target`, `right_hand_target` (sites — the
handle targets live on the cart body, see `unitree_robots/g1/cart/cart_real.xml`).
Enable/disable it via `publish_dolly_observation` in `simulate/config.yaml`.

## Policy blending

`runPolicy()` always evaluates the locomotion policy; it only evaluates the
dolly policy while cart markers are available. The mix is a linear blend
(`dolly_blend_`) that ramps over `POLICY_TRANSITION_SECONDS` (1 s) toward
whichever policy the joystick last requested:

- Hold **A** → ramp toward the dolly policy (only takes effect once markers
  are available).
- Release **A** → ramp back toward locomotion.
- **B** → zero the command and exit the controller.
- If markers become unavailable mid-blend, `dolly_blend_` is forced to `0`
  (falls back to pure locomotion) and a message is printed:
  `Cart markers unavailable; using walking policy`.

While the dolly policy is active, the joystick's max commanded forward speed
is raised to `1.2 m/s` (vs `1.0 m/s` for locomotion-only) and lateral motion
is disabled (`vy = 0`) — only forward/back and yaw are commanded.
