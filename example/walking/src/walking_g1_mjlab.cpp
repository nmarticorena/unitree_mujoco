#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <boost/circular_buffer.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <atomic>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/robots/g1/g1.h>
#include <unitree/dds_wrapper/robots/go2/go2.h>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/idl/ros2/String_.hpp>


#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

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
constexpr int NUM_OBS = 106;
constexpr int NUM_LOCOMOTION_OBS = 99;
constexpr const char* DEFAULT_MODEL_PATH = "mjlab_dolly.onnx";
constexpr const char* LOCOMOTION_MODEL_PATH = "mjlab.onnx";
#else
constexpr int NUM_OBS = 99;
constexpr const char* DEFAULT_MODEL_PATH = "mjlab.onnx";
#endif
constexpr int NUM_OBS_HISTORY = 1;
constexpr int NUM_STACKED_OBS = NUM_OBS * NUM_OBS_HISTORY;

constexpr int NUM_ACTIONS = G1_NUM_MOTOR;
constexpr int NUM_POLICY_JOINTS = G1_NUM_MOTOR;
constexpr int DEFAULT_POSE_RAMP_STEPS = 100;
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
constexpr uint16_t JOYSTICK_DEADMAN_RB = 1U << 0;
#ifdef MJLAB_DOLLY_TASK
constexpr uint16_t JOYSTICK_TRANSITION_A = 1U << 8;
constexpr uint16_t JOYSTICK_EXIT_B = 1U << 9;
constexpr float POLICY_TRANSITION_SECONDS = 1.0f;
#endif
static_assert(NUM_ACTIONS == NUM_POLICY_JOINTS);
#ifdef MJLAB_DOLLY_TASK
static_assert(NUM_OBS == 3 + 3 + 3 + 3 * NUM_POLICY_JOINTS + 3 + 1 + 6);
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
        NUM_OBS_HISTORY
    };

    std::array<float, NUM_STACKED_OBS> stacked_obs{};
    std::array<float, NUM_ACTIONS> raw_action{};
    std::array<float, NUM_ACTIONS> previous_action{};
    std::array<float, G1_NUM_MOTOR> target_q{};
};


float clip(float x, float lo, float hi) {
    return std::max(lo, std::min(x, hi));
}

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
    void LowStateMessageHandler(const void *messages);
    void SportModeMessageHandler(const void *messages);
    void ArmSdkMessageHandler(const void *messages);
    void JoystickMessageHandler(const void *messages);
#ifdef MJLAB_DOLLY_TASK
    void DollyObservationMessageHandler(const void *messages);
#endif

    void copyLowStateToMotorState();
    void buildCurrentObservation();
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

    // IMU state
    std::array<float, 3> gyro_{0.0f, 0.0f, 0.0f};
    Eigen::Quaterniond imu_quaternion_{1.0, 0.0, 0.0, 0.0};
    std::array<float, 3> projected_gravity_{0.0f, 0.0f, 0.0f};

    // Velocity tracking
    std::array<float, 3> velocity_{0.0f, 0.0f, 0.0f};
#ifdef MJLAB_DOLLY_TASK
    std::array<float, 7> dolly_observation_{1.0f, 0.0f, 0.0f, 0.0f,
                                            0.0f, 0.0f, 0.0f};
#endif


    std::array<float, G1_NUM_MOTOR> default_q_{default_joint_positions};
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
        ++default_pose_ramp_step_;
        publishCommand();
        return;
    }

    buildCurrentObservation();

    policy_buffers_.obs_history.push_back(policy_buffers_.current_obs);

    if (!buildStackedObservation()) {
        return;
    }

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
        low_cmd.motor_cmd()[i].kp() = Kp[i];
        low_cmd.motor_cmd()[i].kd() = Kd[i];
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

    const char* input_names[] = {"obs"};
    const char* output_names[] = {"actions"};
    auto infer = [&](Ort::Session& policy, int observation_count,
                     std::array<float, NUM_ACTIONS>& actions) {
        std::array<int64_t, 2> input_shape{1, observation_count};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            policy_buffers_.stacked_obs.data(),
            observation_count,
            input_shape.data(),
            input_shape.size()
        );
        auto output_tensors = policy.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
            output_names, 1);
        const float* output_data = output_tensors.front().GetTensorData<float>();
        std::copy_n(output_data, NUM_ACTIONS, actions.begin());
    };

#ifdef MJLAB_DOLLY_TASK
    const float blend_step = static_cast<float>(dt_) / POLICY_TRANSITION_SECONDS;
    dolly_blend_ = clip(
        dolly_blend_ +
            (transition_to_dolly_.load(std::memory_order_relaxed)
                 ? blend_step
                 : -blend_step),
        0.0f,
        1.0f);

    std::array<float, NUM_ACTIONS> locomotion_action{};
    std::array<float, NUM_ACTIONS> dolly_action{};
    if (dolly_blend_ < 1.0f) {
        infer(locomotion_session_, NUM_LOCOMOTION_OBS, locomotion_action);
    }
    if (dolly_blend_ > 0.0f) {
        infer(session_, NUM_STACKED_OBS, dolly_action);
    }
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        policy_buffers_.raw_action[i] =
            (1.0f - dolly_blend_) * locomotion_action[i] +
            dolly_blend_ * dolly_action[i];
    }
#else
    infer(session_, NUM_STACKED_OBS, policy_buffers_.raw_action);
#endif
}

void LocomotionPolicyController::postProcessAction()
{
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
        std::cout << "Joystick enabled: hold A to transition to the dolly policy; hold RB to command motion; press B to exit" << std::endl;
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
    const bool use_dolly_policy =
        (joystick->keys() & JOYSTICK_TRANSITION_A) != 0;
    const bool was_using_dolly = transition_to_dolly_.exchange(
        use_dolly_policy, std::memory_order_relaxed);
    if (use_dolly_policy != was_using_dolly) {
        std::cout << (use_dolly_policy
                          ? "Transitioning to dolly behavior"
                          : "Transitioning to walking behavior")
                  << std::endl;
    }
#endif

    const bool deadman_pressed =
        (joystick->keys() & JOYSTICK_DEADMAN_RB) != 0;
    auto apply_deadzone = [](float value) {
        return std::abs(value) < JOYSTICK_DEADZONE ? 0.0f : value;
    };

    if (deadman_pressed) {
#ifdef MJLAB_DOLLY_TASK
        if (use_dolly_policy) {
            setCommand(
                apply_deadzone(joystick->ly()) * JOYSTICK_MAX_FORWARD_SPEED,
                0.0,
                -apply_deadzone(joystick->rx()) * JOYSTICK_MAX_YAW_SPEED
            );
        } else {
            setCommand(
                apply_deadzone(joystick->ly()),
                apply_deadzone(joystick->lx()) * JOYSTICK_MAX_LATERAL_SPEED,
                -apply_deadzone(joystick->rx()) * JOYSTICK_MAX_YAW_SPEED
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
    std::array<float, 7> values{};
    for (float& value : values) {
        if (!(input >> value)) {
            std::cerr << "Ignoring invalid dolly observation" << std::endl;
            return;
        }
    }
    input >> std::ws;
    if (!input.eof()) {
        std::cerr << "Ignoring invalid dolly observation" << std::endl;
        return;
    }

    std::lock_guard<std::mutex> lock(low_state_mutex_);
    dolly_observation_ = values;
}
#endif

void LocomotionPolicyController::LoadONNX(){
    session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
    // The model input is NUM_OBS values stacked NUM_OBS_HISTORY times.
    std::cout << "Model loaded successfully: " << model_path_ << std::endl;
#ifdef MJLAB_DOLLY_TASK
    locomotion_session_ =
        Ort::Session(env_, LOCOMOTION_MODEL_PATH, session_options_);
    std::cout << "Locomotion model loaded successfully: "
              << LOCOMOTION_MODEL_PATH << std::endl;
#endif

}

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
            motor_state_.q[motor_idx] - default_q_[motor_idx];
    }

    for (int motor_idx : policy_motor_indices) {
        policy_buffers_.current_obs[k++] = motor_state_.dq[motor_idx];
    }

    for (int i = 0; i < NUM_ACTIONS; ++i) {
        policy_buffers_.current_obs[k++] = policy_buffers_.previous_action[i];
    }


    for (int i = 0; i < 3; ++i) {
        policy_buffers_.current_obs[k++] = command[i];
    }

#ifdef MJLAB_DOLLY_TASK
    std::array<float, 7> dolly_observation{};
    {
        std::lock_guard<std::mutex> lock(low_state_mutex_);
        dolly_observation = dolly_observation_;
    }
    for (float value : dolly_observation) {
        policy_buffers_.current_obs[k++] = clip(value, -3.0f, 3.0f);
    }
    policy_buffers_.current_obs[99] = clip(policy_buffers_.current_obs[99], 0.0f, 1.0f);
#endif
    
    if (k != NUM_OBS) {
        throw std::runtime_error("Observation size mismatch");
    }

}

bool LocomotionPolicyController::buildStackedObservation(){
    if (policy_buffers_.obs_history.size() < NUM_OBS_HISTORY) {
        return false;
    }
    int k = 0;
    for (const auto& obs : policy_buffers_.obs_history) {
        for (float x : obs) {
            policy_buffers_.stacked_obs[k++] = x;
        }
    }
    return true;
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
