#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <boost/circular_buffer.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <functional>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/robots/g1/g1.h>
#include <unitree/dds_wrapper/robots/go2/go2.h>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/idl/ros2/String_.hpp>


#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

#if defined(FALCON_LOCOMOTION) || defined(HOMIE_LOCOMOTION)
#include "premotion_targets.hpp"
#endif

using namespace unitree::common;
using namespace unitree::robot;

volatile std::sig_atomic_t keep_running = 1;

void handleShutdownSignal(int)
{
    keep_running = 0;
}

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_SPORTMODE "rt/sportmodestate"
#define TOPIC_ARM_SDK "rt/arm_sdk"
#define TOPIC_WIRELESS_CONTROLLER "rt/wirelesscontroller"
#define TOPIC_DOLLY_OBSERVATION "rt/dolly_observation"

const int G1_NUM_MOTOR = 29;
constexpr int G1_LOW_CMD_NUM_MOTOR = 35;

#ifdef MJLAB_DOLLY_TASK
constexpr int NUM_OBS = 109;
constexpr const char* MJLAB_CONFIG_PATH = "weights/1/config.yaml";
#ifdef FALCON_LOCOMOTION
constexpr int NUM_LOCOMOTION_OBS = 575;
constexpr int NUM_FALCON_STEP_OBS = 115;
constexpr int NUM_FALCON_HISTORY = 5;
constexpr const char* DEFAULT_MODEL_PATH = "falcon_g1_29dof.onnx";
constexpr const char* FALCON_CONFIG_PATH = "falcon_g1_29dof.yaml";
constexpr const char* DOLLY_MODEL_PATH = "weights/1/policy.onnx";
#elif defined(HOMIE_LOCOMOTION)
constexpr int NUM_LOCOMOTION_OBS = 480;
constexpr int NUM_HOMIE_STEP_OBS = 80;
constexpr int NUM_HOMIE_HISTORY = 6;
constexpr int NUM_HOMIE_ACTIONS = 12;
constexpr const char* DEFAULT_MODEL_PATH = "homie.onnx";
constexpr const char* DOLLY_MODEL_PATH = "weights/1/policy.onnx";
#else
constexpr int NUM_LOCOMOTION_OBS = 99;
constexpr const char* DEFAULT_MODEL_PATH = "weights/1/policy.onnx";
constexpr const char* LOCOMOTION_MODEL_PATH = "mjlab.onnx";
#endif
#else
constexpr int NUM_OBS = 99;
constexpr const char* DEFAULT_MODEL_PATH = "mjlab.onnx";
#endif
constexpr int DEFAULT_NUM_OBS_HISTORY = 1;

constexpr int NUM_ACTIONS = G1_NUM_MOTOR;
constexpr int NUM_POLICY_JOINTS = G1_NUM_MOTOR;
#ifdef FALCON_LOCOMOTION
constexpr int DEFAULT_POSE_RAMP_STEPS = 500;
#else
constexpr int DEFAULT_POSE_RAMP_STEPS = 100;
#endif
constexpr int POLICY_WARMUP_STEPS = 20;
constexpr float JOYSTICK_DEADZONE = 0.08f;
constexpr float JOYSTICK_MAX_FORWARD_SPEED =
#ifdef MJLAB_DOLLY_TASK
    1.2f;
#else
    1.0f;
#endif
constexpr float JOYSTICK_MAX_LATERAL_SPEED = 0.5f;
constexpr float JOYSTICK_MAX_YAW_SPEED = 1.0f;
constexpr float COMMAND_MOTION_THRESHOLD = 1.0e-4f;
constexpr uint16_t JOYSTICK_DEADMAN_RB = 1U << 0;
#ifdef MJLAB_DOLLY_TASK
constexpr uint16_t JOYSTICK_TRANSITION_A = 1U << 8;
constexpr uint16_t JOYSTICK_EXIT_B = 1U << 9;
#ifdef FALCON_LOCOMOTION
constexpr uint16_t JOYSTICK_FALCON_WALK_TOGGLE_R2 = 1U << 4;
#endif
constexpr float POLICY_TRANSITION_SECONDS = 1.0f;
#if defined(FALCON_LOCOMOTION) || defined(HOMIE_LOCOMOTION)
constexpr float PREMOTION_PHASE_END = 1.0f;
#endif
#ifdef FALCON_LOCOMOTION
constexpr float FALCON_ACTION_SCALE = 0.25f;
constexpr float FALCON_ARM_REFERENCE_MAX_SPEED = 3.0f;
constexpr float FALCON_ARM_TARGET_MAX_SPEED = 3.0f;
#elif defined(HOMIE_LOCOMOTION)
constexpr float HOMIE_ARM_REFERENCE_MAX_SPEED = 3.0f;
constexpr float HOMIE_WAIST_KP = 300.0f;
constexpr float HOMIE_WAIST_KD = 5.0f;
#endif
constexpr auto DOLLY_OBSERVATION_TIMEOUT = std::chrono::milliseconds(250);
#endif
static_assert(NUM_ACTIONS == NUM_POLICY_JOINTS);
#ifdef MJLAB_DOLLY_TASK
static_assert(NUM_OBS == 3 + 3 + 3 + 3 * NUM_POLICY_JOINTS + 3 + 1 + 9);
#else
static_assert(NUM_OBS == 3 + 3 + 3 + 3 * NUM_POLICY_JOINTS + 3);
#endif

constexpr float ACTION_SCALE_5020 = 0.4385773139f;
constexpr float ACTION_SCALE_7520_14 = 0.5475464630f;
constexpr float ACTION_SCALE_7520_22 = 0.3506614664f;
constexpr float ACTION_SCALE_4010 = 0.0745008703f;

constexpr float KP_5020 = 14.2506231f;
constexpr float KP_7520_14 = 40.1792386f;
constexpr float KP_7520_22 = 99.0984278f;
constexpr float KP_4010 = 16.7783275f;
constexpr float KP_5020_PARALLEL = 28.5012462f;

constexpr float KD_5020 = 0.90722284f;
constexpr float KD_7520_14 = 2.55788978f;
constexpr float KD_7520_22 = 6.30880185f;
constexpr float KD_4010 = 1.06814150f;
constexpr float KD_5020_PARALLEL = 1.81444569f;

// mjlab G1 actuator stiffness in DDS joint order.
std::array<float, G1_NUM_MOTOR> Kp{
    KP_7520_14, KP_7520_22, KP_7520_14, KP_7520_22, KP_5020_PARALLEL, KP_5020_PARALLEL,
    KP_7520_14, KP_7520_22, KP_7520_14, KP_7520_22, KP_5020_PARALLEL, KP_5020_PARALLEL,
    KP_7520_14, KP_5020_PARALLEL, KP_5020_PARALLEL,
    KP_5020, KP_5020, KP_5020, KP_5020, KP_5020, KP_4010, KP_4010,
    KP_5020, KP_5020, KP_5020, KP_5020, KP_5020, KP_4010, KP_4010
};

// mjlab G1 actuator damping in DDS joint order.
std::array<float, G1_NUM_MOTOR> Kd{
    KD_7520_14, KD_7520_22, KD_7520_14, KD_7520_22, KD_5020_PARALLEL, KD_5020_PARALLEL,
    KD_7520_14, KD_7520_22, KD_7520_14, KD_7520_22, KD_5020_PARALLEL, KD_5020_PARALLEL,
    KD_7520_14, KD_5020_PARALLEL, KD_5020_PARALLEL,
    KD_5020, KD_5020, KD_5020, KD_5020, KD_5020, KD_4010, KD_4010,
    KD_5020, KD_5020, KD_5020, KD_5020, KD_5020, KD_4010, KD_4010
};

enum G1JointIndex {
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnklePitch = 4,
  LeftAnkleB = 4,
  LeftAnkleRoll = 5,
  LeftAnkleA = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  WaistYaw = 12,
  WaistRoll = 13,        // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistA = 13,           // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistPitch = 14,       // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistB = 14,           // NOTE INVALID for g1 23dof/29dof with waist locked
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20,   // NOTE INVALID for g1 23dof
  LeftWristYaw = 21,     // NOTE INVALID for g1 23dof
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27,  // NOTE INVALID for g1 23dof
  RightWristYaw = 28     // NOTE INVALID for g1 23dof
};

const std::array<int, NUM_POLICY_JOINTS> policy_motor_indices = {
    LeftHipPitch,
    LeftHipRoll,
    LeftHipYaw,
    LeftKnee,
    LeftAnklePitch,
    LeftAnkleRoll,

    RightHipPitch,
    RightHipRoll,
    RightHipYaw,
    RightKnee,
    RightAnklePitch,
    RightAnkleRoll,

    WaistYaw,
    WaistRoll,
    WaistPitch,

    LeftShoulderPitch,
    LeftShoulderRoll,
    LeftShoulderYaw,
    LeftElbow,
    LeftWristRoll,
    LeftWristPitch,
    LeftWristYaw,

    RightShoulderPitch,
    RightShoulderRoll,
    RightShoulderYaw,
    RightElbow,
    RightWristRoll,
    RightWristPitch,
    RightWristYaw
};

constexpr int ARM_SDK_ENABLE_INDEX = 29;
constexpr float ARM_SDK_ENABLE_THRESHOLD = 0.5f;
constexpr int NUM_ARM_SDK_MOTORS = 14;

const std::array<int, NUM_ARM_SDK_MOTORS> arm_sdk_motor_indices = {
    LeftShoulderPitch,
    LeftShoulderRoll,
    LeftShoulderYaw,
    LeftElbow,
    LeftWristRoll,
    LeftWristPitch,
    LeftWristYaw,
    RightShoulderPitch,
    RightShoulderRoll,
    RightShoulderYaw,
    RightElbow,
    RightWristRoll,
    RightWristPitch,
    RightWristYaw
};

const std::array<float, NUM_ARM_SDK_MOTORS> default_arm_joint_positions = {
    0.2f, 0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f,
    0.2f, -0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f
};

static_assert(ARM_SDK_ENABLE_INDEX < G1_LOW_CMD_NUM_MOTOR);

const std::array<float, NUM_POLICY_JOINTS> action_scale = {
    ACTION_SCALE_7520_14,
    ACTION_SCALE_7520_22,
    ACTION_SCALE_7520_14,
    ACTION_SCALE_7520_22,
    ACTION_SCALE_5020,
    ACTION_SCALE_5020,

    ACTION_SCALE_7520_14,
    ACTION_SCALE_7520_22,
    ACTION_SCALE_7520_14,
    ACTION_SCALE_7520_22,
    ACTION_SCALE_5020,
    ACTION_SCALE_5020,

    ACTION_SCALE_7520_14,
    ACTION_SCALE_5020,
    ACTION_SCALE_5020,

    ACTION_SCALE_5020,
    ACTION_SCALE_5020,
    ACTION_SCALE_5020,
    ACTION_SCALE_5020,
    ACTION_SCALE_5020,
    ACTION_SCALE_4010,
    ACTION_SCALE_4010,

    ACTION_SCALE_5020,
    ACTION_SCALE_5020,
    ACTION_SCALE_5020,
    ACTION_SCALE_5020,
    ACTION_SCALE_5020,
    ACTION_SCALE_4010,
    ACTION_SCALE_4010
};

const std::array<float, G1_NUM_MOTOR> default_joint_positions{
    -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f,
    -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.2f, 0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f,
    0.2f, -0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f
};

const std::array<float, G1_NUM_MOTOR> joint_position_min{
    -2.5307f, -0.5236f, -2.7576f, -0.087267f, -0.87267f, -0.2618f,
    -2.5307f, -2.9671f, -2.7576f, -0.087267f, -0.87267f, -0.2618f,
    -2.618f, -0.52f, -0.52f,
    -3.0892f, -1.5882f, -2.618f, -1.0472f, -1.97222f, -1.61443f, -1.61443f,
    -3.0892f, -2.2515f, -2.618f, -1.0472f, -1.97222f, -1.61443f, -1.61443f
};

const std::array<float, G1_NUM_MOTOR> joint_position_max{
    2.8798f, 2.9671f, 2.7576f, 2.8798f, 0.5236f, 0.2618f,
    2.8798f, 0.5236f, 2.7576f, 2.8798f, 0.5236f, 0.2618f,
    2.618f, 0.52f, 0.52f,
    2.6704f, 2.2515f, 2.618f, 2.0944f, 1.97222f, 1.61443f, 1.61443f,
    2.6704f, 1.5882f, 2.618f, 2.0944f, 1.97222f, 1.61443f, 1.61443f
};

struct MotorState {
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
};

struct PolicyBuffers {
    std::array<float, NUM_OBS> current_obs{};
    boost::circular_buffer<std::array<float, NUM_OBS>> obs_history{
        DEFAULT_NUM_OBS_HISTORY
    };

    std::vector<float> stacked_obs = std::vector<float>(NUM_OBS);
    std::array<float, NUM_ACTIONS> raw_action{};
    std::array<float, NUM_ACTIONS> previous_action{};
    std::array<float, G1_NUM_MOTOR> target_q{};
};

#ifdef FALCON_LOCOMOTION
struct FalconPolicyBuffers {
    std::array<float, NUM_FALCON_STEP_OBS> current_obs{};
    boost::circular_buffer<std::array<float, NUM_FALCON_STEP_OBS>> obs_history{
        NUM_FALCON_HISTORY
    };
    std::array<float, NUM_LOCOMOTION_OBS> stacked_obs{};
    std::array<float, NUM_ACTIONS> raw_action{};
    std::array<float, NUM_ACTIONS> previous_action{};
    std::array<float, G1_NUM_MOTOR> target_q{};
};

const std::array<float, G1_NUM_MOTOR> falcon_default_joint_positions{
    -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
    -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

const std::array<float, G1_NUM_MOTOR> falcon_kp{
    100.0f, 100.0f, 100.0f, 200.0f, 20.0f, 20.0f,
    100.0f, 100.0f, 100.0f, 200.0f, 20.0f, 20.0f,
    300.0f, 300.0f, 300.0f,
    90.0f, 60.0f, 20.0f, 60.0f, 4.0f, 4.0f, 4.0f,
    90.0f, 60.0f, 20.0f, 60.0f, 4.0f, 4.0f, 4.0f
};

const std::array<float, G1_NUM_MOTOR> falcon_kd{
    2.5f, 2.5f, 2.5f, 5.0f, 0.2f, 0.1f,
    2.5f, 2.5f, 2.5f, 5.0f, 0.2f, 0.1f,
    5.0f, 5.0f, 5.0f,
    2.0f, 1.0f, 0.4f, 1.0f, 0.2f, 0.2f, 0.2f,
    2.0f, 1.0f, 0.4f, 1.0f, 0.2f, 0.2f, 0.2f
};
#elif defined(HOMIE_LOCOMOTION)
struct HomiePolicyBuffers {
    std::array<float, NUM_HOMIE_STEP_OBS> current_obs{};
    boost::circular_buffer<std::array<float, NUM_HOMIE_STEP_OBS>> obs_history{
        NUM_HOMIE_HISTORY
    };
    std::array<float, NUM_LOCOMOTION_OBS> stacked_obs{};
    std::array<float, NUM_HOMIE_ACTIONS> raw_action{};
    std::array<float, NUM_HOMIE_ACTIONS> previous_action{};
    std::array<float, G1_NUM_MOTOR> target_q{};
};

const std::array<float, G1_NUM_MOTOR> homie_default_joint_positions{
    -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
    -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.2f, 0.2f, 0.0f, 1.28f, 0.0f, 0.0f, 0.0f,
    0.2f, -0.2f, 0.0f, 1.28f, 0.0f, 0.0f, 0.0f
};
#endif


float clip(float x, float lo, float hi) {
    return std::max(lo, std::min(x, hi));
}

#if defined(FALCON_LOCOMOTION) || defined(HOMIE_LOCOMOTION)
float samplePremotion(float phase, std::size_t joint_index, float initial)
{
    const float time = clip(phase, 0.0f, PREMOTION_PHASE_END) *
                       PREMOTION_DURATION_SECONDS;
    if (time <= PREMOTION_SAMPLE_DT_SECONDS) {
        const float alpha = time / PREMOTION_SAMPLE_DT_SECONDS;
        return initial + alpha *
            (PREMOTION_ARM_TARGETS.front()[joint_index] - initial);
    }
    if (time >= PREMOTION_DURATION_SECONDS) {
        return PREMOTION_ARM_TARGETS.back()[joint_index];
    }

    const float sample_position =
        time / PREMOTION_SAMPLE_DT_SECONDS - 1.0f;
    const std::size_t lower = static_cast<std::size_t>(sample_position);
    const std::size_t upper = std::min(
        lower + 1, PREMOTION_ARM_TARGETS.size() - 1);
    const float alpha = sample_position - static_cast<float>(lower);
    return PREMOTION_ARM_TARGETS[lower][joint_index] + alpha *
        (PREMOTION_ARM_TARGETS[upper][joint_index] -
         PREMOTION_ARM_TARGETS[lower][joint_index]);
}
#endif

class LocomotionPolicyController
{
public:
    explicit LocomotionPolicyController(
        std::string model_path = DEFAULT_MODEL_PATH,
        bool enable_arm_sdk = false,
        bool enable_joystick = false,
        double dt = 0.02 // default 50Hz
    );

    void init();
    void shutdown();
    void setCommand(double vx, double vy, double wz);
    void update();
    void publishCommand();

private:
    void InitLowCmd();
    void LoadONNX();
#ifdef MJLAB_DOLLY_TASK
    void LoadMjlabConfig();
#endif
#ifdef FALCON_LOCOMOTION
    void LoadFalconConfig();
#endif
    void LowStateMessageHandler(const void *messages);
    void SportModeMessageHandler(const void *messages);
    void ArmSdkMessageHandler(const void *messages);
    void JoystickMessageHandler(const void *messages);
#ifdef MJLAB_DOLLY_TASK
    void DollyObservationMessageHandler(const void *messages);
    bool dollyObservationAvailable();
#endif

    void copyLowStateToMotorState();
    void buildCurrentObservation();
#ifdef FALCON_LOCOMOTION
    void buildFalconObservation();
    bool buildFalconStackedObservation();
#elif defined(HOMIE_LOCOMOTION)
    void buildHomieObservation();
    bool buildHomieStackedObservation();
#endif
    bool buildStackedObservation();
    void runPolicy();
    void postProcessAction();

private:
    std::string model_path_;
    bool enable_arm_sdk_;
    bool enable_joystick_;
    double dt_;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
#ifdef MJLAB_DOLLY_TASK
    Ort::Session locomotion_session_{nullptr};
    std::atomic<bool> transition_to_dolly_{false};
    float dolly_blend_{0.0f};
    bool dolly_markers_missing_logged_{false};
#ifdef FALCON_LOCOMOTION
    std::atomic<bool> falcon_walk_mode_{false};
    bool falcon_walk_toggle_pressed_{false};
#endif
#endif

    // Robot state
    MotorState motor_state_;
    unitree_hg::msg::dds_::LowCmd_ low_cmd{};     // default init
    unitree_hg::msg::dds_::LowCmd_ arm_sdk_cmd{}; // default init
    unitree_hg::msg::dds_::LowState_ low_state{}; // default init
    

    std::array<float, 3> command_{
        0.0f, 0.0f, 0.0f 
    }; 


    // Controller
    PolicyBuffers policy_buffers_;
    int num_obs_history_{DEFAULT_NUM_OBS_HISTORY};
#ifdef MJLAB_DOLLY_TASK
    bool dolly_was_available_{false};
#endif
#if defined(MJLAB_DOLLY_TASK) && !defined(FALCON_LOCOMOTION) && !defined(HOMIE_LOCOMOTION)
    std::array<float, NUM_ACTIONS> dolly_previous_action_{};
    std::array<float, NUM_ACTIONS> locomotion_previous_action_{};
#endif
#ifdef FALCON_LOCOMOTION
    FalconPolicyBuffers falcon_buffers_;
    std::array<float, NUM_ARM_SDK_MOTORS> falcon_arm_reference_{};
    std::array<float, G1_NUM_MOTOR> falcon_default_q_{
        falcon_default_joint_positions};
    std::array<float, G1_NUM_MOTOR> falcon_kp_{falcon_kp};
    std::array<float, G1_NUM_MOTOR> falcon_kd_{falcon_kd};
    std::array<float, G1_NUM_MOTOR> command_kp_{falcon_kp};
    std::array<float, G1_NUM_MOTOR> command_kd_{falcon_kd};
#elif defined(HOMIE_LOCOMOTION)
    HomiePolicyBuffers homie_buffers_;
    std::array<float, NUM_ARM_SDK_MOTORS> homie_arm_reference_{};
    std::atomic<bool> homie_use_dolly_commands_{false};
    std::array<float, G1_NUM_MOTOR> homie_kp_{Kp};
    std::array<float, G1_NUM_MOTOR> homie_kd_{Kd};
    std::array<float, G1_NUM_MOTOR> command_kp_{Kp};
    std::array<float, G1_NUM_MOTOR> command_kd_{Kd};
#endif

    // IMU state
    std::array<float, 3> gyro_{0.0f, 0.0f, 0.0f};
    Eigen::Quaterniond imu_quaternion_{1.0, 0.0, 0.0, 0.0};
    std::array<float, 3> projected_gravity_{0.0f, 0.0f, 0.0f};

    // Velocity tracking
    std::array<float, 3> velocity_{0.0f, 0.0f, 0.0f};
#ifdef MJLAB_DOLLY_TASK
    std::array<float, 10> dolly_observation_{};
    std::chrono::steady_clock::time_point last_dolly_observation_time_{};
#endif


    std::array<float, G1_NUM_MOTOR> default_q_{
#ifdef FALCON_LOCOMOTION
        falcon_default_joint_positions
#elif defined(HOMIE_LOCOMOTION)
        homie_default_joint_positions
#else
        default_joint_positions
#endif
    };
    std::array<float, G1_NUM_MOTOR> startup_q_{};
    int default_pose_ramp_step_{0};
    int policy_warmup_step_{0};
    bool startup_q_initialized_{false};
    
    
    /*publisher*/
    ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher;
    /*subscriber*/
    ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> sportmode_subscriber;
    ChannelSubscriberPtr<unitree_hg::msg::dds_::LowCmd_> arm_sdk_subscriber;
    ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> joystick_subscriber;
#ifdef MJLAB_DOLLY_TASK
    ChannelSubscriberPtr<std_msgs::msg::dds_::String_> dolly_observation_subscriber;
#endif

    /*LowCmd write thread*/
    ThreadPtr lowCmdWriteThreadPtr;

    std::mutex low_state_mutex_;
    std::mutex arm_sdk_mutex_;
    std::atomic<uint64_t> low_state_sequence_{0};
    std::atomic<uint64_t> arm_sdk_sequence_{0};
    uint64_t last_processed_low_state_sequence_{0};
};

void LocomotionPolicyController::update()
{
    const uint64_t low_state_sequence =
        low_state_sequence_.load(std::memory_order_acquire);
    if (low_state_sequence == 0 ||
        low_state_sequence == last_processed_low_state_sequence_) {
        return;
    }
    last_processed_low_state_sequence_ = low_state_sequence;

    copyLowStateToMotorState();

    if (default_pose_ramp_step_ < DEFAULT_POSE_RAMP_STEPS) {
        if (!startup_q_initialized_) {
            startup_q_ = motor_state_.q;
            startup_q_initialized_ = true;
        }

        const float phase =
            static_cast<float>(default_pose_ramp_step_ + 1) /
            static_cast<float>(DEFAULT_POSE_RAMP_STEPS);

        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
            policy_buffers_.target_q[i] =
                (1.0f - phase) * startup_q_[i] + phase * default_q_[i];
        }

        policy_buffers_.obs_history.clear();
        policy_buffers_.previous_action.fill(0.0f);
#if defined(MJLAB_DOLLY_TASK) && !defined(FALCON_LOCOMOTION) && !defined(HOMIE_LOCOMOTION)
        dolly_previous_action_.fill(0.0f);
        locomotion_previous_action_.fill(0.0f);
#endif
#ifdef FALCON_LOCOMOTION
        falcon_buffers_.obs_history.clear();
        falcon_buffers_.previous_action.fill(0.0f);
#elif defined(HOMIE_LOCOMOTION)
        homie_buffers_.obs_history.clear();
        homie_buffers_.previous_action.fill(0.0f);
#endif
        ++default_pose_ramp_step_;
        publishCommand();
        return;
    }

#ifdef MJLAB_DOLLY_TASK
    const bool dolly_available = dollyObservationAvailable();
    if (dolly_available != dolly_was_available_) {
        policy_buffers_.obs_history.clear();
        policy_buffers_.previous_action.fill(0.0f);
#if !defined(FALCON_LOCOMOTION) && !defined(HOMIE_LOCOMOTION)
        dolly_previous_action_.fill(0.0f);
#endif
        dolly_was_available_ = dolly_available;
    }
#endif

    buildCurrentObservation();

    policy_buffers_.obs_history.push_back(policy_buffers_.current_obs);

    if (!buildStackedObservation()) {
        return;
    }
#ifdef FALCON_LOCOMOTION
    buildFalconObservation();
    falcon_buffers_.obs_history.push_back(falcon_buffers_.current_obs);
    if (!buildFalconStackedObservation()) {
        return;
    }
#elif defined(HOMIE_LOCOMOTION)
    buildHomieObservation();
    homie_buffers_.obs_history.push_back(homie_buffers_.current_obs);
    if (!buildHomieStackedObservation()) {
        return;
    }
#endif

    if (policy_warmup_step_ < POLICY_WARMUP_STEPS) {
        runPolicy();
        postProcessAction();
        policy_buffers_.target_q = default_q_;
        publishCommand();
        ++policy_warmup_step_;
        return;
    }

    runPolicy();
    postProcessAction();
    publishCommand();
}

void LocomotionPolicyController::publishCommand()
{
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        low_cmd.motor_cmd()[i].mode() = 0x01;
        low_cmd.motor_cmd()[i].q() = policy_buffers_.target_q[i];
        low_cmd.motor_cmd()[i].dq() = 0.0f;
#if defined(FALCON_LOCOMOTION) || defined(HOMIE_LOCOMOTION)
        low_cmd.motor_cmd()[i].kp() = command_kp_[i];
        low_cmd.motor_cmd()[i].kd() = command_kd_[i];
#else
        low_cmd.motor_cmd()[i].kp() = Kp[i];
        low_cmd.motor_cmd()[i].kd() = Kd[i];
#endif
        low_cmd.motor_cmd()[i].tau() = 0.0f;
    }

    if (enable_arm_sdk_) {
        unitree_hg::msg::dds_::LowCmd_ arm_sdk_cmd_copy{};
        bool use_arm_sdk_cmd = false;

        if (arm_sdk_sequence_.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> lock(arm_sdk_mutex_);
            arm_sdk_cmd_copy = arm_sdk_cmd;
            use_arm_sdk_cmd =
                arm_sdk_cmd_copy.motor_cmd()[ARM_SDK_ENABLE_INDEX].q() >
                ARM_SDK_ENABLE_THRESHOLD;
        }

        for (std::size_t i = 0; i < arm_sdk_motor_indices.size(); ++i) {
            const int motor_idx = arm_sdk_motor_indices[i];
            auto& dst = low_cmd.motor_cmd()[motor_idx];

            if (use_arm_sdk_cmd) {
                const auto& src = arm_sdk_cmd_copy.motor_cmd()[motor_idx];

                dst.mode() = src.mode();
                dst.q() = clip(
                    src.q(),
                    joint_position_min[motor_idx],
                    joint_position_max[motor_idx]
                );
                dst.dq() = src.dq();
                dst.tau() = src.tau();
                if (src.kp() > 0.0f) {
                    dst.kp() = src.kp();
                }
                if (src.kd() > 0.0f) {
                    dst.kd() = src.kd();
                }
            } else {
                dst.q() = clip(
                    default_arm_joint_positions[i],
                    joint_position_min[motor_idx],
                    joint_position_max[motor_idx]
                );
                dst.dq() = 0.0f;
                dst.tau() = 0.0f;
            }
        }
    }

    lowcmd_publisher->Write(low_cmd);
}

void LocomotionPolicyController::runPolicy()
{
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    auto infer = [&](Ort::Session& policy, const float* observations,
                     int observation_count, const char* input_name,
                     const char* output_name,
                     auto& actions) {
        const char* input_names[] = {input_name};
        const char* output_names[] = {output_name};
        std::array<int64_t, 2> input_shape{1, observation_count};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(observations),
            observation_count,
            input_shape.data(),
            input_shape.size()
        );
        auto output_tensors = policy.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
            output_names, 1);
        const float* output_data = output_tensors.front().GetTensorData<float>();
        std::copy_n(output_data, actions.size(), actions.begin());
    };

#ifdef MJLAB_DOLLY_TASK
    const bool dolly_requested =
        transition_to_dolly_.load(std::memory_order_relaxed) &&
        policy_warmup_step_ >= POLICY_WARMUP_STEPS;
    const bool markers_available = dollyObservationAvailable();
    if (!markers_available) {
        if (dolly_blend_ > 0.0f && !dolly_markers_missing_logged_) {
            std::cout << "Cart markers unavailable; using walking policy"
                      << std::endl;
            dolly_markers_missing_logged_ = true;
        }
#if defined(FALCON_LOCOMOTION) || defined(HOMIE_LOCOMOTION)
        const float seconds = dolly_blend_ > PREMOTION_PHASE_END
                                  ? POLICY_TRANSITION_SECONDS
                                  : PREMOTION_DURATION_SECONDS;
        dolly_blend_ = std::max(
            0.0f, dolly_blend_ - static_cast<float>(dt_) / seconds);
#else
        dolly_blend_ = 0.0f;
#endif
    } else {
        dolly_markers_missing_logged_ = false;
#if defined(FALCON_LOCOMOTION) || defined(HOMIE_LOCOMOTION)
        const bool in_premotion = dolly_requested
            ? dolly_blend_ < PREMOTION_PHASE_END
            : dolly_blend_ <= PREMOTION_PHASE_END;
        const float blend_step = static_cast<float>(dt_) /
            (in_premotion ? PREMOTION_DURATION_SECONDS
                          : POLICY_TRANSITION_SECONDS);
#else
        const float blend_step =
            static_cast<float>(dt_) / POLICY_TRANSITION_SECONDS;
#endif
        dolly_blend_ = clip(
            dolly_blend_ + (dolly_requested ? blend_step : -blend_step),
            0.0f,
#if defined(FALCON_LOCOMOTION) || defined(HOMIE_LOCOMOTION)
            PREMOTION_PHASE_END + 1.0f);
#else
            1.0f);
#endif
    }

#ifdef HOMIE_LOCOMOTION
    homie_use_dolly_commands_.store(
        dolly_blend_ > PREMOTION_PHASE_END,
        std::memory_order_relaxed);
#endif

#ifdef FALCON_LOCOMOTION
    std::array<float, NUM_ACTIONS> dolly_action{};
    if (markers_available && dolly_blend_ > 0.0f) {
        infer(session_, policy_buffers_.stacked_obs.data(),
              static_cast<int>(policy_buffers_.stacked_obs.size()),
              "obs", "actions", dolly_action);
        const bool valid = std::all_of(
            dolly_action.begin(), dolly_action.end(),
            [](float action) { return std::isfinite(action); });
        if (valid) {
            policy_buffers_.raw_action = dolly_action;
        } else {
            std::cerr << "Ignoring non-finite dolly policy output" << std::endl;
            const float seconds = dolly_blend_ > PREMOTION_PHASE_END
                                       ? POLICY_TRANSITION_SECONDS
                                       : PREMOTION_DURATION_SECONDS;
            dolly_blend_ = std::max(
                0.0f, dolly_blend_ - static_cast<float>(dt_) / seconds);
        }
    }
    infer(locomotion_session_, falcon_buffers_.stacked_obs.data(),
          NUM_LOCOMOTION_OBS, "actor_obs", "action",
          falcon_buffers_.raw_action);
    for (float& action : falcon_buffers_.raw_action) {
        action = std::isfinite(action) ? clip(action, -100.0f, 100.0f) : 0.0f;
    }
#elif defined(HOMIE_LOCOMOTION)
    std::array<float, NUM_ACTIONS> dolly_action{};
    if (markers_available && dolly_blend_ > 0.0f) {
        infer(session_, policy_buffers_.stacked_obs.data(),
              static_cast<int>(policy_buffers_.stacked_obs.size()),
              "obs", "actions", dolly_action);
        const bool valid = std::all_of(
            dolly_action.begin(), dolly_action.end(),
            [](float action) { return std::isfinite(action); });
        if (valid) {
            policy_buffers_.raw_action = dolly_action;
        } else {
            std::cerr << "Ignoring non-finite dolly policy output" << std::endl;
            const float seconds = dolly_blend_ > PREMOTION_PHASE_END
                                       ? POLICY_TRANSITION_SECONDS
                                       : PREMOTION_DURATION_SECONDS;
            dolly_blend_ = std::max(
                0.0f, dolly_blend_ - static_cast<float>(dt_) / seconds);
        }
    }
    infer(locomotion_session_, homie_buffers_.stacked_obs.data(),
          NUM_LOCOMOTION_OBS, "obs", "actions", homie_buffers_.raw_action);
    for (float& action : homie_buffers_.raw_action) {
        action = std::isfinite(action) ? clip(action, -100.0f, 100.0f) : 0.0f;
    }
#else
    std::array<float, NUM_ACTIONS> locomotion_action{};
    std::array<float, NUM_ACTIONS> dolly_action{};
    if (dolly_blend_ < 1.0f) {
        std::array<float, NUM_LOCOMOTION_OBS> locomotion_obs{};
        std::copy_n(policy_buffers_.current_obs.begin(), 67,
                    locomotion_obs.begin());
        std::copy(locomotion_previous_action_.begin(),
                  locomotion_previous_action_.end(),
                  locomotion_obs.begin() + 67);
        std::copy_n(policy_buffers_.current_obs.begin() + 96, 3,
                    locomotion_obs.begin() + 96);
        infer(locomotion_session_, locomotion_obs.data(),
              NUM_LOCOMOTION_OBS, "obs", "actions", locomotion_action);
        locomotion_previous_action_ = locomotion_action;
    }
    if (dolly_blend_ > 0.0f) {
        infer(session_, policy_buffers_.stacked_obs.data(),
              static_cast<int>(policy_buffers_.stacked_obs.size()),
              "obs", "actions", dolly_action);
        dolly_previous_action_ = dolly_action;
    }
    const bool valid = std::all_of(
        locomotion_action.begin(), locomotion_action.end(),
        [](float action) { return std::isfinite(action); }) &&
        std::all_of(dolly_action.begin(), dolly_action.end(),
        [](float action) { return std::isfinite(action); });
    if (!valid) {
        std::cerr << "Ignoring non-finite MJLab policy output" << std::endl;
        policy_buffers_.raw_action.fill(0.0f);
        dolly_previous_action_.fill(0.0f);
        locomotion_previous_action_.fill(0.0f);
        return;
    }
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        policy_buffers_.raw_action[i] =
            (1.0f - dolly_blend_) * locomotion_action[i] +
            dolly_blend_ * dolly_action[i];
    }
#endif
#else
    infer(session_, policy_buffers_.stacked_obs.data(),
          static_cast<int>(policy_buffers_.stacked_obs.size()),
          "obs", "actions", policy_buffers_.raw_action);
#endif
}

void LocomotionPolicyController::postProcessAction()
{
#ifdef FALCON_LOCOMOTION
    std::array<float, G1_NUM_MOTOR> dolly_target{default_joint_positions};
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        dolly_target[i] += action_scale[i] * policy_buffers_.raw_action[i];
    }

    const float trajectory_phase =
        std::min(dolly_blend_, PREMOTION_PHASE_END);
    const float policy_blend =
        std::max(dolly_blend_ - PREMOTION_PHASE_END, 0.0f);
    for (std::size_t i = 0; i < arm_sdk_motor_indices.size(); ++i) {
        const int motor_idx = arm_sdk_motor_indices[i];
        const float desired_reference = samplePremotion(
            trajectory_phase, i, falcon_default_q_[motor_idx]);
        const float max_reference_step =
            FALCON_ARM_REFERENCE_MAX_SPEED * static_cast<float>(dt_);
        falcon_arm_reference_[i] += clip(
            desired_reference - falcon_arm_reference_[i],
            -max_reference_step, max_reference_step);
    }

    falcon_buffers_.target_q = falcon_default_q_;
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        falcon_buffers_.target_q[i] +=
            FALCON_ACTION_SCALE * falcon_buffers_.raw_action[i];
    }
    for (std::size_t i = 0; i < arm_sdk_motor_indices.size(); ++i) {
        const int motor_idx = arm_sdk_motor_indices[i];
        const float desired_target = falcon_arm_reference_[i] +
            FALCON_ACTION_SCALE * falcon_buffers_.raw_action[motor_idx];
        const float max_target_step =
            FALCON_ARM_TARGET_MAX_SPEED * static_cast<float>(dt_);
        falcon_buffers_.target_q[motor_idx] =
            policy_buffers_.target_q[motor_idx] + clip(
                desired_target - policy_buffers_.target_q[motor_idx],
                -max_target_step, max_target_step);
    }

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        policy_buffers_.target_q[i] = clip(
            (1.0f - policy_blend) * falcon_buffers_.target_q[i] +
                policy_blend * dolly_target[i],
            joint_position_min[i], joint_position_max[i]);
        command_kp_[i] =
            (1.0f - policy_blend) * falcon_kp_[i] + policy_blend * Kp[i];
        command_kd_[i] =
            (1.0f - policy_blend) * falcon_kd_[i] + policy_blend * Kd[i];
    }
    policy_buffers_.previous_action = policy_buffers_.raw_action;
    falcon_buffers_.previous_action = falcon_buffers_.raw_action;
    if (dolly_blend_ == 0.0f) {
        policy_buffers_.raw_action.fill(0.0f);
        policy_buffers_.previous_action.fill(0.0f);
    }
#elif defined(HOMIE_LOCOMOTION)
    std::array<float, G1_NUM_MOTOR> dolly_target{default_joint_positions};
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        dolly_target[i] += action_scale[i] * policy_buffers_.raw_action[i];
    }

    const float trajectory_phase =
        std::min(dolly_blend_, PREMOTION_PHASE_END);
    const float policy_blend =
        std::max(dolly_blend_ - PREMOTION_PHASE_END, 0.0f);
    for (std::size_t i = 0; i < arm_sdk_motor_indices.size(); ++i) {
        const float desired_reference = samplePremotion(
            trajectory_phase, i,
            homie_default_joint_positions[arm_sdk_motor_indices[i]]);
        const float max_reference_step =
            HOMIE_ARM_REFERENCE_MAX_SPEED * static_cast<float>(dt_);
        homie_arm_reference_[i] += clip(
            desired_reference - homie_arm_reference_[i],
            -max_reference_step, max_reference_step);
    }

    homie_buffers_.target_q = homie_default_joint_positions;
    for (int i = 0; i < NUM_HOMIE_ACTIONS; ++i) {
        homie_buffers_.target_q[i] +=
            action_scale[i] * homie_buffers_.raw_action[i];
    }
    for (std::size_t i = 0; i < arm_sdk_motor_indices.size(); ++i) {
        homie_buffers_.target_q[arm_sdk_motor_indices[i]] =
            homie_arm_reference_[i];
    }

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        policy_buffers_.target_q[i] = clip(
            (1.0f - policy_blend) * homie_buffers_.target_q[i] +
                policy_blend * dolly_target[i],
            joint_position_min[i], joint_position_max[i]);
        command_kp_[i] =
            (1.0f - policy_blend) * homie_kp_[i] + policy_blend * Kp[i];
        command_kd_[i] =
            (1.0f - policy_blend) * homie_kd_[i] + policy_blend * Kd[i];
    }
    policy_buffers_.previous_action = policy_buffers_.raw_action;
    homie_buffers_.previous_action = homie_buffers_.raw_action;
    if (dolly_blend_ == 0.0f) {
        policy_buffers_.raw_action.fill(0.0f);
        policy_buffers_.previous_action.fill(0.0f);
    }
#else
    policy_buffers_.target_q = default_q_;

    for (int i = 0; i < NUM_ACTIONS; ++i) {
        const int motor_idx = policy_motor_indices[i];
        policy_buffers_.target_q[motor_idx] =
            default_q_[motor_idx] + action_scale[i] * policy_buffers_.raw_action[i];
    }

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        policy_buffers_.target_q[i] = clip(
            policy_buffers_.target_q[i],
            joint_position_min[i],
            joint_position_max[i]
        );
    }

    policy_buffers_.previous_action = policy_buffers_.raw_action;
#endif
}

LocomotionPolicyController::LocomotionPolicyController(
    std::string model_path,
    bool enable_arm_sdk,
    bool enable_joystick,
    double dt
)
    : model_path_(std::move(model_path)),
      enable_arm_sdk_(enable_arm_sdk),
      enable_joystick_(enable_joystick),
      dt_(dt),
      env_(ORT_LOGGING_LEVEL_WARNING, "walking_g1"),
      session_options_(),
      session_(nullptr)
{
}

void LocomotionPolicyController::init()
{
    InitLowCmd();
#ifdef MJLAB_DOLLY_TASK
    LoadMjlabConfig();
#endif
#ifdef FALCON_LOCOMOTION
    LoadFalconConfig();
#elif defined(HOMIE_LOCOMOTION)
    for (int motor_idx : {WaistYaw, WaistRoll, WaistPitch}) {
        homie_kp_[motor_idx] = HOMIE_WAIST_KP;
        homie_kd_[motor_idx] = HOMIE_WAIST_KD;
    }
    command_kp_ = homie_kp_;
    command_kd_ = homie_kd_;
    for (std::size_t i = 0; i < arm_sdk_motor_indices.size(); ++i) {
        homie_arm_reference_[i] =
            homie_default_joint_positions[arm_sdk_motor_indices[i]];
    }
#endif
    LoadONNX();
    /*create publisher*/
    lowcmd_publisher.reset(new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();

    /*create subscriber*/
    lowstate_subscriber.reset(new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&LocomotionPolicyController::LowStateMessageHandler, this, std::placeholders::_1), 1);
    sportmode_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_SPORTMODE));
    sportmode_subscriber->InitChannel(std::bind(&LocomotionPolicyController::SportModeMessageHandler, this, std::placeholders::_1), 1);
    if (enable_arm_sdk_) {
        arm_sdk_subscriber.reset(new ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_>(TOPIC_ARM_SDK));
        arm_sdk_subscriber->InitChannel(std::bind(&LocomotionPolicyController::ArmSdkMessageHandler, this, std::placeholders::_1), 1);
    }
    if (enable_joystick_) {
        joystick_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_WIRELESS_CONTROLLER));
        joystick_subscriber->InitChannel(std::bind(&LocomotionPolicyController::JoystickMessageHandler, this, std::placeholders::_1), 1);
#ifdef MJLAB_DOLLY_TASK
#ifdef FALCON_LOCOMOTION
        std::cout << "Joystick enabled: press R2 to toggle FALCON walk/stance; hold A to transition to the dolly policy; hold RB to command motion; press B to exit" << std::endl;
#else
        std::cout << "Joystick enabled: hold A to transition to the dolly policy; hold RB to command motion; press B to exit" << std::endl;
#endif
#else
        std::cout << "Joystick enabled: hold RB to drive, left stick moves, right stick turns" << std::endl;
#endif
    }
#ifdef MJLAB_DOLLY_TASK
    dolly_observation_subscriber.reset(
        new ChannelSubscriber<std_msgs::msg::dds_::String_>(TOPIC_DOLLY_OBSERVATION));
    dolly_observation_subscriber->InitChannel(
        std::bind(&LocomotionPolicyController::DollyObservationMessageHandler,
                  this, std::placeholders::_1), 1);
#endif
    
    /*loop publishing thread*/
    lowCmdWriteThreadPtr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, int(dt_ * 1000000), &LocomotionPolicyController::update, this);
}

void LocomotionPolicyController::shutdown()
{
    lowCmdWriteThreadPtr.reset();
    arm_sdk_subscriber.reset();
    joystick_subscriber.reset();
#ifdef MJLAB_DOLLY_TASK
    dolly_observation_subscriber.reset();
#endif

    const unitree_hg::msg::dds_::LowCmd_ release_command{};
    for (int i = 0; i < 5; ++i) {
        lowcmd_publisher->Write(release_command);
        usleep(20000);
    }
}

void LocomotionPolicyController::setCommand(double vx, double vy, double wz)
{
    std::lock_guard<std::mutex> lock(low_state_mutex_);
    command_[0] = static_cast<float>(vx);
    command_[1] = static_cast<float>(vy);
    command_[2] = static_cast<float>(wz);
}

void LocomotionPolicyController::JoystickMessageHandler(const void *message)
{
    const auto *joystick =
        static_cast<const unitree_go::msg::dds_::WirelessController_*>(message);

#ifdef MJLAB_DOLLY_TASK
    if ((joystick->keys() & JOYSTICK_EXIT_B) != 0) {
        setCommand(0.0, 0.0, 0.0);
        keep_running = 0;
        return;
    }
    const bool request_dolly_policy =
        (joystick->keys() & JOYSTICK_TRANSITION_A) != 0;
    const bool was_using_dolly = transition_to_dolly_.exchange(
        request_dolly_policy, std::memory_order_relaxed);
    if (request_dolly_policy != was_using_dolly) {
        std::cout << (request_dolly_policy
                          ? "Transitioning to dolly behavior"
                          : "Transitioning to walking behavior")
                  << std::endl;
    }
#ifdef HOMIE_LOCOMOTION
    const bool use_dolly_policy =
        homie_use_dolly_commands_.load(std::memory_order_relaxed) &&
        dollyObservationAvailable();
#else
    const bool use_dolly_policy =
        request_dolly_policy && dollyObservationAvailable();
#endif
#ifdef FALCON_LOCOMOTION
    const bool walk_toggle_pressed =
        (joystick->keys() & JOYSTICK_FALCON_WALK_TOGGLE_R2) != 0;
    if (walk_toggle_pressed && !falcon_walk_toggle_pressed_) {
        const bool walk_mode = !falcon_walk_mode_.load(std::memory_order_relaxed);
        falcon_walk_mode_.store(walk_mode, std::memory_order_relaxed);
        std::cout << (walk_mode ? "FALCON walk mode" : "FALCON stance mode")
                  << std::endl;
    }
    falcon_walk_toggle_pressed_ = walk_toggle_pressed;
#endif
#endif

    const bool deadman_pressed =
        (joystick->keys() & JOYSTICK_DEADMAN_RB) != 0;
    auto apply_deadzone = [](float value) {
        return std::abs(value) < JOYSTICK_DEADZONE ? 0.0f : value;
    };

    if (deadman_pressed
#ifdef FALCON_LOCOMOTION
        && (falcon_walk_mode_.load(std::memory_order_relaxed) ||
            use_dolly_policy)
#endif
    ) {
#ifdef MJLAB_DOLLY_TASK
        if (use_dolly_policy) {
            setCommand(
                apply_deadzone(joystick->ly()) * JOYSTICK_MAX_FORWARD_SPEED,
                0.0,
                -apply_deadzone(joystick->rx()) * JOYSTICK_MAX_YAW_SPEED
            );
        } else {
            setCommand(
#ifdef HOMIE_LOCOMOTION
                apply_deadzone(joystick->ly()) *
                    (joystick->ly() >= 0.0f ? 1.2f : 0.8f),
                -apply_deadzone(joystick->lx()) * JOYSTICK_MAX_LATERAL_SPEED,
                -apply_deadzone(joystick->rx()) * 0.8f
#else
                apply_deadzone(joystick->ly()),
#ifdef FALCON_LOCOMOTION
                -apply_deadzone(joystick->lx()) * JOYSTICK_MAX_LATERAL_SPEED,
#else
                apply_deadzone(joystick->lx()) * JOYSTICK_MAX_LATERAL_SPEED,
#endif
                -apply_deadzone(joystick->rx()) * JOYSTICK_MAX_YAW_SPEED
#endif
            );
        }
#else
        setCommand(
            apply_deadzone(joystick->ly()) * JOYSTICK_MAX_FORWARD_SPEED,
            apply_deadzone(joystick->lx()) * JOYSTICK_MAX_LATERAL_SPEED,
            -apply_deadzone(joystick->rx()) * JOYSTICK_MAX_YAW_SPEED
        );
#endif
    } else {
        setCommand(0.0, 0.0, 0.0);
    }
}

#ifdef MJLAB_DOLLY_TASK
void LocomotionPolicyController::DollyObservationMessageHandler(const void *message)
{
    const auto *observation =
        static_cast<const std_msgs::msg::dds_::String_*>(message);
    std::istringstream input(observation->data());
    std::vector<float> values;
    float value = 0.0f;
    while (input >> value) {
        if (!std::isfinite(value)) {
            std::cerr << "Ignoring invalid dolly observation" << std::endl;
            return;
        }
        values.push_back(value);
    }
    if (!input.eof()) {
        std::cerr << "Ignoring invalid dolly observation" << std::endl;
        return;
    }

    std::lock_guard<std::mutex> lock(low_state_mutex_);
    if (values.size() != dolly_observation_.size()) {
        std::cerr << "Ignoring dolly observation: the weights/1 policy requires "
                  << "10 values (present, cart x/y/yaw, left xyz, right xyz), got "
                  << values.size() << std::endl;
        return;
    }
    std::copy(values.begin(), values.end(), dolly_observation_.begin());
    last_dolly_observation_time_ = std::chrono::steady_clock::now();
}

bool LocomotionPolicyController::dollyObservationAvailable()
{
    std::lock_guard<std::mutex> lock(low_state_mutex_);
    return dolly_observation_[0] >= 0.5f &&
           std::chrono::steady_clock::now() - last_dolly_observation_time_ <=
               DOLLY_OBSERVATION_TIMEOUT;
}
#endif

void LocomotionPolicyController::LoadONNX(){
#if defined(FALCON_LOCOMOTION) || defined(HOMIE_LOCOMOTION)
    auto validate_shape = [](Ort::Session& policy, int64_t input_size,
                             int64_t output_size, const char* policy_name) {
        if (policy.GetInputCount() != 1 || policy.GetOutputCount() != 1) {
            throw std::runtime_error(
                std::string(policy_name) + " must have one input and one output");
        }
        const auto input_shape = policy.GetInputTypeInfo(0)
                                     .GetTensorTypeAndShapeInfo().GetShape();
        const auto output_shape = policy.GetOutputTypeInfo(0)
                                      .GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape != std::vector<int64_t>{1, input_size} ||
            output_shape != std::vector<int64_t>{1, output_size}) {
            throw std::runtime_error(
                std::string(policy_name) + " has incompatible tensor shapes");
        }
    };

#ifdef FALCON_LOCOMOTION
    locomotion_session_ =
        Ort::Session(env_, model_path_.c_str(), session_options_);
    validate_shape(locomotion_session_, NUM_LOCOMOTION_OBS, NUM_ACTIONS,
                   "FALCON policy");
    std::cout << "FALCON model loaded successfully: " << model_path_
              << std::endl;
    session_ = Ort::Session(env_, DOLLY_MODEL_PATH, session_options_);
    validate_shape(session_, static_cast<int64_t>(policy_buffers_.stacked_obs.size()),
                   NUM_ACTIONS, "Dolly policy");
    std::cout << "Dolly model loaded successfully: " << DOLLY_MODEL_PATH
              << std::endl;
#else
    locomotion_session_ =
        Ort::Session(env_, model_path_.c_str(), session_options_);
    validate_shape(locomotion_session_, NUM_LOCOMOTION_OBS,
                   NUM_HOMIE_ACTIONS, "HoMIeRL policy");
    std::cout << "HoMIeRL model loaded successfully: " << model_path_
              << std::endl;
    session_ = Ort::Session(env_, DOLLY_MODEL_PATH, session_options_);
    validate_shape(session_, static_cast<int64_t>(policy_buffers_.stacked_obs.size()),
                   NUM_ACTIONS, "Dolly policy");
    std::cout << "Dolly model loaded successfully: " << DOLLY_MODEL_PATH
              << std::endl;
#endif
#else
    session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
    const auto input_shape = session_.GetInputTypeInfo(0)
                                 .GetTensorTypeAndShapeInfo().GetShape();
    const auto output_shape = session_.GetOutputTypeInfo(0)
                                  .GetTensorTypeAndShapeInfo().GetShape();
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1 ||
        input_shape != std::vector<int64_t>{
            1, static_cast<int64_t>(policy_buffers_.stacked_obs.size())} ||
        output_shape != std::vector<int64_t>{1, NUM_ACTIONS}) {
        throw std::runtime_error("MJLab policy has incompatible tensor shapes");
    }
    std::cout << "Model loaded successfully: " << model_path_ << std::endl;
#ifdef MJLAB_DOLLY_TASK
    locomotion_session_ =
        Ort::Session(env_, LOCOMOTION_MODEL_PATH, session_options_);
    std::cout << "Locomotion model loaded successfully: "
              << LOCOMOTION_MODEL_PATH << std::endl;
#endif
#endif

}

#ifdef MJLAB_DOLLY_TASK
void LocomotionPolicyController::LoadMjlabConfig()
{
    const YAML::Node config = YAML::LoadFile(MJLAB_CONFIG_PATH);
    const YAML::Node history =
        config["env_cfg"]["value"]["observations"]["actor"]["history_length"];
    if (!history || !history.IsScalar()) {
        throw std::runtime_error(
            std::string("Missing actor history_length in ") + MJLAB_CONFIG_PATH);
    }

    num_obs_history_ = history.as<int>();
    if (num_obs_history_ < 1) {
        throw std::runtime_error("MJLab actor history_length must be positive");
    }
    policy_buffers_.obs_history.set_capacity(num_obs_history_);
    policy_buffers_.stacked_obs.assign(NUM_OBS * num_obs_history_, 0.0f);
    std::cout << "MJLab config loaded successfully: " << MJLAB_CONFIG_PATH
              << " (observation horizon: " << num_obs_history_ << ")"
              << std::endl;
}
#endif

#ifdef FALCON_LOCOMOTION
void LocomotionPolicyController::LoadFalconConfig()
{
    auto load_array = [&](const char* key,
                          std::array<float, G1_NUM_MOTOR>& destination) {
        std::ifstream config(FALCON_CONFIG_PATH);
        if (!config) {
            throw std::runtime_error(
                std::string("Unable to open FALCON config: ") +
                FALCON_CONFIG_PATH);
        }

        std::string values;
        std::string line;
        bool reading = false;
        while (std::getline(config, line)) {
            if (!reading) {
                const std::size_t key_position = line.find(key);
                if (key_position == std::string::npos ||
                    line.find(':', key_position + std::string(key).size()) ==
                        std::string::npos) {
                    continue;
                }
                reading = true;
            }
            const std::size_t comment = line.find('#');
            if (comment != std::string::npos) {
                line.erase(comment);
            }
            values += line;
            values += ' ';
            if (line.find(']') != std::string::npos) {
                break;
            }
        }

        for (char& character : values) {
            if (character == '[' || character == ']' || character == ',') {
                character = ' ';
            }
        }
        const std::size_t separator = values.find(':');
        std::istringstream input(
            separator == std::string::npos ? std::string{} :
                                             values.substr(separator + 1));
        std::size_t count = 0;
        while (count < destination.size() && input >> destination[count]) {
            ++count;
        }
        float extra = 0.0f;
        if (!reading || count != destination.size() || input >> extra) {
            throw std::runtime_error(
                std::string("Invalid FALCON config field: ") + key);
        }
    };

    load_array("DEFAULT_DOF_ANGLES", falcon_default_q_);
    load_array("MOTOR_KP", falcon_kp_);
    load_array("MOTOR_KD", falcon_kd_);
    default_q_ = falcon_default_q_;
    command_kp_ = falcon_kp_;
    command_kd_ = falcon_kd_;
    std::cout << "FALCON control config loaded successfully: "
              << FALCON_CONFIG_PATH << std::endl;
}
#endif

void LocomotionPolicyController::InitLowCmd()
{
}

void LocomotionPolicyController::SportModeMessageHandler(const void *message)
{
    auto state =
        (const unitree_go::msg::dds_::SportModeState_*)message;
    {
        std::lock_guard<std::mutex> lock(low_state_mutex_);
        velocity_[0] = state->velocity()[0];
        velocity_[1] = state->velocity()[1];
        velocity_[2] = state->velocity()[2];
    }
   }

void LocomotionPolicyController::LowStateMessageHandler(const void *message)
{
    auto state =
        (const unitree_hg::msg::dds_::LowState_*)message;
    {
        std::lock_guard<std::mutex> lock(low_state_mutex_);
        low_state = *state;
    }
    low_state_sequence_.fetch_add(1, std::memory_order_release);
}

void LocomotionPolicyController::ArmSdkMessageHandler(const void *message)
{
    auto command =
        (const unitree_hg::msg::dds_::LowCmd_*)message;
    {
        std::lock_guard<std::mutex> lock(arm_sdk_mutex_);
        arm_sdk_cmd = *command;
    }
    arm_sdk_sequence_.fetch_add(1, std::memory_order_release);
}

void LocomotionPolicyController::copyLowStateToMotorState() {
    unitree_hg::msg::dds_::LowState_ state_copy{};

    {
        std::lock_guard<std::mutex> lock(low_state_mutex_);
        state_copy = low_state;
    }

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        motor_state_.q[i] = state_copy.motor_state()[i].q();
        motor_state_.dq[i] = state_copy.motor_state()[i].dq();
    }

    gyro_[0] = state_copy.imu_state().gyroscope()[0];
    gyro_[1] = state_copy.imu_state().gyroscope()[1];
    gyro_[2] = state_copy.imu_state().gyroscope()[2];

    imu_quaternion_.w() = state_copy.imu_state().quaternion()[0];
    imu_quaternion_.x() = state_copy.imu_state().quaternion()[1];
    imu_quaternion_.y() = state_copy.imu_state().quaternion()[2];
    imu_quaternion_.z() = state_copy.imu_state().quaternion()[3];

    Eigen::Vector3d gravity(0.0, 0.0, -1.0);
    Eigen::Matrix3d rotation_matrix = imu_quaternion_.toRotationMatrix();
    Eigen::Vector3d projected_gravity = rotation_matrix.transpose() * gravity;

    projected_gravity_[0] = projected_gravity[0];
    projected_gravity_[1] = projected_gravity[1];
    projected_gravity_[2] = projected_gravity[2];
}

void LocomotionPolicyController::buildCurrentObservation()
{
    int k = 0;
    std::array<float, 3> velocity{};
    std::array<float, 3> command{};

    {
        std::lock_guard<std::mutex> lock(low_state_mutex_);
        velocity = velocity_;
        command = command_;
    }
#ifdef FALCON_LOCOMOTION
    if (dolly_blend_ > 0.0f && dolly_blend_ <= PREMOTION_PHASE_END) {
        command.fill(0.0f);
    }
#elif defined(HOMIE_LOCOMOTION)
    if (dolly_blend_ > 0.0f && dolly_blend_ <= PREMOTION_PHASE_END) {
        command.fill(0.0f);
    }
#endif

#ifdef MJLAB_DOLLY_TASK
    const Eigen::Vector3d world_velocity(velocity[0], velocity[1], velocity[2]);
    const Eigen::Vector3d local_velocity = imu_quaternion_.conjugate() * world_velocity;
    for (int i = 0; i < 3; ++i) {
        velocity[i] = static_cast<float>(local_velocity[i]);
    }
#endif

    for (int i = 0; i < 3; ++i) {
        policy_buffers_.current_obs[k++] = velocity[i];
    }
    
    for (int i = 0; i < 3; ++i) {
        policy_buffers_.current_obs[k++] = gyro_[i];
    }

    for (int i = 0; i < 3; ++i) {
        policy_buffers_.current_obs[k++] = projected_gravity_[i];
    }

    for (int motor_idx : policy_motor_indices) {
        policy_buffers_.current_obs[k++] =
#if defined(FALCON_LOCOMOTION) || defined(HOMIE_LOCOMOTION)
            motor_state_.q[motor_idx] - default_joint_positions[motor_idx];
#else
            motor_state_.q[motor_idx] - default_q_[motor_idx];
#endif
    }

    for (int motor_idx : policy_motor_indices) {
        policy_buffers_.current_obs[k++] = motor_state_.dq[motor_idx];
    }

    for (int i = 0; i < NUM_ACTIONS; ++i) {
        policy_buffers_.current_obs[k++] =
#if defined(MJLAB_DOLLY_TASK) && !defined(FALCON_LOCOMOTION) && !defined(HOMIE_LOCOMOTION)
            dolly_previous_action_[i];
#else
            policy_buffers_.previous_action[i];
#endif
    }


    for (int i = 0; i < 3; ++i) {
        policy_buffers_.current_obs[k++] = command[i];
    }

#ifdef MJLAB_DOLLY_TASK
    decltype(dolly_observation_) dolly_observation{};
    {
        std::lock_guard<std::mutex> lock(low_state_mutex_);
        dolly_observation = dolly_observation_;
    }
    if (!dollyObservationAvailable()) {
        dolly_observation[0] = 0.0f;
    }
    policy_buffers_.current_obs[k++] = clip(dolly_observation[0], 0.0f, 1.0f);
    for (std::size_t i = 1; i < dolly_observation.size(); ++i) {
        policy_buffers_.current_obs[k++] = clip(
            dolly_observation[i],
            -5.0f, 5.0f
        );
    }
#endif
    
    if (k != NUM_OBS) {
        throw std::runtime_error("Observation size mismatch");
    }

}

#ifdef FALCON_LOCOMOTION
void LocomotionPolicyController::buildFalconObservation()
{
    int k = 0;
    std::array<float, 3> command{};
    {
        std::lock_guard<std::mutex> lock(low_state_mutex_);
        command = command_;
    }
    if (dolly_blend_ > 0.0f && dolly_blend_ <= PREMOTION_PHASE_END) {
        command.fill(0.0f);
    }

    // FALCON's deployment code sorts observation term names alphabetically.
    for (float action : falcon_buffers_.previous_action) {
        falcon_buffers_.current_obs[k++] = action;
    }
    for (float value : gyro_) {
        falcon_buffers_.current_obs[k++] = 0.25f * value;
    }
    falcon_buffers_.current_obs[k++] = command[2];
    falcon_buffers_.current_obs[k++] = 2.0f * 0.75f;
    falcon_buffers_.current_obs[k++] = command[0];
    falcon_buffers_.current_obs[k++] = command[1];
    const float command_norm =
        std::abs(command[0]) + std::abs(command[1]) + std::abs(command[2]);
    const bool walking =
        falcon_walk_mode_.load(std::memory_order_relaxed) &&
        command_norm > COMMAND_MOTION_THRESHOLD;
    falcon_buffers_.current_obs[k++] = walking ? 1.0f : 0.0f;
    falcon_buffers_.current_obs[k++] = 0.0f;
    falcon_buffers_.current_obs[k++] = 0.0f;
    falcon_buffers_.current_obs[k++] = 0.0f;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        falcon_buffers_.current_obs[k++] =
            motor_state_.q[i] - falcon_default_q_[i];
    }
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        falcon_buffers_.current_obs[k++] = 0.05f * motor_state_.dq[i];
    }
    for (float value : projected_gravity_) {
        falcon_buffers_.current_obs[k++] = value;
    }
    for (float value : falcon_arm_reference_) {
        falcon_buffers_.current_obs[k++] = value;
    }

    if (k != NUM_FALCON_STEP_OBS) {
        throw std::runtime_error("FALCON observation size mismatch");
    }
}

bool LocomotionPolicyController::buildFalconStackedObservation()
{
    while (falcon_buffers_.obs_history.size() < NUM_FALCON_HISTORY) {
        falcon_buffers_.obs_history.push_front({});
    }
    int k = 0;
    for (const auto& obs : falcon_buffers_.obs_history) {
        for (float value : obs) {
            falcon_buffers_.stacked_obs[k++] = value;
        }
    }
    return k == NUM_LOCOMOTION_OBS;
}
#endif

#ifdef HOMIE_LOCOMOTION
void LocomotionPolicyController::buildHomieObservation()
{
    int k = 0;
    std::array<float, 3> command{};
    {
        std::lock_guard<std::mutex> lock(low_state_mutex_);
        command = command_;
    }
    if (dolly_blend_ > 0.0f &&
        dolly_blend_ <= PREMOTION_PHASE_END) {
        command.fill(0.0f);
    }

    homie_buffers_.current_obs[k++] = 2.0f * command[0];
    homie_buffers_.current_obs[k++] = 2.0f * command[1];
    homie_buffers_.current_obs[k++] = 0.5f * command[2];
    homie_buffers_.current_obs[k++] = 0.78f;
    for (float value : gyro_) {
        homie_buffers_.current_obs[k++] = 0.5f * value;
    }
    for (float value : projected_gravity_) {
        homie_buffers_.current_obs[k++] = value;
    }
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        homie_buffers_.current_obs[k++] =
            motor_state_.q[i] - homie_default_joint_positions[i];
    }
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        homie_buffers_.current_obs[k++] = 0.05f * motor_state_.dq[i];
    }
    for (float action : homie_buffers_.previous_action) {
        homie_buffers_.current_obs[k++] = action;
    }

    if (k != NUM_HOMIE_STEP_OBS) {
        throw std::runtime_error("HoMIeRL observation size mismatch");
    }
}

bool LocomotionPolicyController::buildHomieStackedObservation()
{
    while (homie_buffers_.obs_history.size() < NUM_HOMIE_HISTORY) {
        homie_buffers_.obs_history.push_front(homie_buffers_.current_obs);
    }
    int k = 0;
    for (const auto& obs : homie_buffers_.obs_history) {
        for (float value : obs) {
            homie_buffers_.stacked_obs[k++] = value;
        }
    }
    return k == NUM_LOCOMOTION_OBS;
}
#endif

bool LocomotionPolicyController::buildStackedObservation(){
    while (policy_buffers_.obs_history.size() <
           static_cast<std::size_t>(num_obs_history_)) {
        policy_buffers_.obs_history.push_front(policy_buffers_.current_obs);
    }
    int k = 0;
#ifdef MJLAB_DOLLY_TASK
    constexpr std::array<int, 10> term_offsets{
        0, 3, 6, 9, 38, 67, 96, 99, 100, 109};
    for (std::size_t term = 0; term + 1 < term_offsets.size(); ++term) {
        for (const auto& obs : policy_buffers_.obs_history) {
            for (int i = term_offsets[term]; i < term_offsets[term + 1]; ++i) {
                policy_buffers_.stacked_obs[k++] = obs[i];
            }
        }
    }
#else
    for (const auto& obs : policy_buffers_.obs_history) {
        for (float x : obs) {
            policy_buffers_.stacked_obs[k++] = x;
        }
    }
#endif
    return k == static_cast<int>(policy_buffers_.stacked_obs.size());
}

int main(int argc, const char **argv)
{
    std::string model_path = DEFAULT_MODEL_PATH;
    const char* network_interface = nullptr;
    bool enable_arm_sdk = false;
    bool enable_joystick = false;
    int dds_domain_id = -1;
    int positional_argument = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--arm-sdk") {
            enable_arm_sdk = true;
        } else if (argument == "--joystick") {
            enable_joystick = true;
        } else if (argument == "--dds-id") {
            if (++i >= argc) {
                std::cerr << "Missing value for --dds-id" << std::endl;
                return 1;
            }
            try {
                std::size_t parsed_characters = 0;
                dds_domain_id = std::stoi(argv[i], &parsed_characters);
                if (parsed_characters != std::string(argv[i]).size() ||
                    dds_domain_id < 0) {
                    throw std::invalid_argument("invalid DDS domain ID");
                }
            } catch (const std::exception&) {
                std::cerr << "Invalid DDS domain ID: " << argv[i] << std::endl;
                return 1;
            }
        } else if (positional_argument == 0) {
            model_path = argument;
            ++positional_argument;
        } else if (positional_argument == 1) {
            network_interface = argv[i];
            ++positional_argument;
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [model_path] [network_interface] [--arm-sdk]"
                      << " [--joystick] [--dds-id id]"
                      << std::endl;
            return 1;
        }
    }

    if (dds_domain_id < 0) {
        dds_domain_id = network_interface == nullptr ? 1 : 0;
    }
    ChannelFactory::Instance()->Init(
        dds_domain_id,
        network_interface == nullptr ? "lo" : network_interface
    );
    std::cout << "Press enter to start";
    std::cin.get();
    LocomotionPolicyController controller(
        model_path,
        enable_arm_sdk,
        enable_joystick
    );
    std::signal(SIGINT, handleShutdownSignal);
    std::signal(SIGTERM, handleShutdownSignal);
    std::signal(SIGHUP, handleShutdownSignal);
    controller.init();

    while (keep_running)
    {
        usleep(20000);
    }
    controller.shutdown();

    return 0;
}
