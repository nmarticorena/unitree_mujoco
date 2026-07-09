#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <string>
#include <boost/circular_buffer.hpp>
#include <Eigen/Dense>

#include <iostream>
#include <stdint.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/robots/g1/g1.h>
#include <unitree/idl/ros2/String_.hpp>


#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

using namespace unitree::common;
using namespace unitree::robot;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_ARM_SDK "rt/arm_sdk"

const int G1_NUM_MOTOR = 29;
constexpr int G1_LOW_CMD_NUM_MOTOR = 35;

constexpr int NUM_OBS = 91;
constexpr int NUM_OBS_HISTORY = 10;
constexpr int NUM_STACKED_OBS = NUM_OBS * NUM_OBS_HISTORY;

constexpr int NUM_ACTIONS = 12;
constexpr int NUM_OBS_JOINTS = 26;
constexpr int DEFAULT_POSE_RAMP_STEPS = 100;
constexpr int POLICY_WARMUP_STEPS = 20;

constexpr float OBS_CLIP = 100.0f;
constexpr float ACTION_CLIP = 100.0f;
constexpr float ACTION_SCALE = 0.25f;



// Stiffness for all G1 Joints
std::array<float, G1_NUM_MOTOR> Kp{
    200, 150, 150, 200, 20, 20,      // left leg
    200, 150, 150, 200, 20, 20,      // right leg
    200, 200, 200,                   // waist
    100, 100, 50, 50, 40, 40, 40,    // left arm
    100, 100, 50, 50, 40, 40, 40     // right arm
};

// Damping for all G1 Joints
std::array<float, G1_NUM_MOTOR> Kd{
    5, 5, 5, 5, 2, 2,     // left leg
    5, 5, 5, 5, 2, 2,     // right leg
    5, 5, 5,              // waist
    2, 2, 2, 2, 2, 2, 2,  // left arm
    2, 2, 2, 2, 2, 2, 2   // right arm
};

enum class Mode {
  PR = 0,  // Series Control for Ptich/Roll Joints
  AB = 1   // Parallel Control for A/B Joints
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

const std::array<int, NUM_ACTIONS> action_motor_indices = {
    LeftHipPitch,
    RightHipPitch,
    LeftHipRoll,
    RightHipRoll,
    LeftHipYaw,
    RightHipYaw,
    LeftKnee,
    RightKnee,
    LeftAnklePitch,
    RightAnklePitch,
    LeftAnkleRoll,
    RightAnkleRoll
};

const std::array<int, G1_NUM_MOTOR> old_action_motor_indices = {
    LeftHipPitch,
    RightHipPitch,
    WaistYaw,
    LeftHipRoll,
    RightHipRoll,
    WaistRoll,
    LeftHipYaw,
    RightHipYaw,
    WaistPitch,
    LeftKnee,
    RightKnee,
    LeftShoulderPitch,
    RightShoulderPitch,
    LeftAnklePitch,
    RightAnklePitch,
    LeftShoulderRoll,
    RightShoulderRoll,
    LeftAnkleRoll,
    RightAnkleRoll,
    LeftShoulderYaw,
    RightShoulderYaw,
    LeftElbow,
    RightElbow,
    LeftWristRoll,
    RightWristRoll,
    LeftWristPitch,
    RightWristPitch,
    LeftWristYaw,
    RightWristYaw
};

const std::array<int, NUM_ACTIONS> action_old_order_indices = {
    0,
    1,
    3,
    4,
    6,
    7,
    9,
    10,
    13,
    14,
    17,
    18
};

const std::array<int, NUM_OBS_JOINTS> obs_motor_indices = {
    // Legs, in policy observation order
    LeftHipPitch,
    RightHipPitch,
    LeftHipRoll,
    RightHipRoll,
    LeftHipYaw,
    RightHipYaw,
    LeftKnee,
    RightKnee,
    LeftAnklePitch,
    RightAnklePitch,
    LeftAnkleRoll,
    RightAnkleRoll,

    // Arms
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
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
    0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f
};

static_assert(ARM_SDK_ENABLE_INDEX < G1_LOW_CMD_NUM_MOTOR);

const std::array<float, G1_NUM_MOTOR> default_joint_positions{
    -0.20f, 0.0f, 0.0f, 0.42f, -0.23f, 0.0f,
    -0.20f, 0.0f, 0.0f, 0.42f, -0.23f, 0.0f,
    0.0f, 0.0f, 0.0f,
    0.35f, 0.18f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f,
    0.35f, -0.18f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f
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

struct MotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
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
    std::array<float, G1_NUM_MOTOR> full_action{};
    std::array<float, G1_NUM_MOTOR> delayed_action{};
    std::array<float, G1_NUM_MOTOR> target_q{};
};


float clip(float x, float lo, float hi) {
    return std::max(lo, std::min(x, hi));
}

void clipArray(std::array<float, NUM_OBS>& obs) {
    for (auto& x : obs) {
        x = clip(x, -OBS_CLIP, OBS_CLIP);
    }
}

class LocomotionPolicyController
{
public:
    explicit LocomotionPolicyController(
        std::string model_path = "policy.onnx",
        double dt = 0.02 // default 50Hz
    );

    void init();
    void update();
    void publishCommand();

private:
    void InitLowCmd();
    void LoadONNX();
    void LowStateMessageHandler(const void *messages);
    void ArmSdkMessageHandler(const void *messages);
    void RunCommandMessageHandler(const void *messages);
    void setCommand(double vx, double vy, double wz, double height);
    void LowCmdWrite();

    void copyLowStateToMotorState();
    void buildCurrentObservation();
    bool buildStackedObservation();
    void runPolicy();
    void postProcessAction();

private:
    std::string model_path_;
    double dt_;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};

    // Robot state
    MotorState motor_state_;
    MotorCommand motor_cmd_;
    unitree_hg::msg::dds_::LowCmd_ low_cmd{};     // default init
    unitree_hg::msg::dds_::LowCmd_ arm_sdk_cmd{}; // default init
    unitree_hg::msg::dds_::LowState_ low_state{}; // default init
    

    std::array<float, 4> command_{
        0.0f, 0.0f, 0.0f, 0.7f
    }; 


    // Controller
    PolicyBuffers policy_buffers_;

    // IMU state
    std::array<float, 3> gyro_{0.0f, 0.0f, 0.0f};
    Eigen::Quaterniond imu_quaternion_{1.0, 0.0, 0.0, 0.0};
    std::array<float, 3> projected_gravity_{0.0f, 0.0f, 0.0f};

    std::array<float, G1_NUM_MOTOR> default_q_{default_joint_positions};
    std::array<float, G1_NUM_MOTOR> default_dq_{};
    std::array<float, G1_NUM_MOTOR> startup_q_{};
    int default_pose_ramp_step_{0};
    int policy_warmup_step_{0};
    bool startup_q_initialized_{false};
    
    
    /*publisher*/
    ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher;
    /*subscriber*/
    ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_hg::msg::dds_::LowCmd_> arm_sdk_subscriber;
    ChannelSubscriberPtr<std_msgs::msg::dds_::String_> velocity_subscriber;

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
        policy_buffers_.delayed_action.fill(0.0f);
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
    unitree_hg::msg::dds_::LowCmd_ arm_sdk_cmd_copy{};
    bool use_arm_sdk_cmd = false;

    if (arm_sdk_sequence_.load(std::memory_order_acquire) > 0) {
        std::lock_guard<std::mutex> lock(arm_sdk_mutex_);
        arm_sdk_cmd_copy = arm_sdk_cmd;
        use_arm_sdk_cmd =
            arm_sdk_cmd_copy.motor_cmd()[ARM_SDK_ENABLE_INDEX].q() >
            ARM_SDK_ENABLE_THRESHOLD;
    }

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        low_cmd.motor_cmd()[i].mode() = 0x01;
        low_cmd.motor_cmd()[i].q() = policy_buffers_.target_q[i];
        low_cmd.motor_cmd()[i].dq() = 0.0f;
        low_cmd.motor_cmd()[i].kp() = Kp[i];
        low_cmd.motor_cmd()[i].kd() = Kd[i];
        low_cmd.motor_cmd()[i].tau() = 0.0f;
    }

    // Keep the leg policy from owning the arm joints.
    for (int i = 0; i < NUM_ARM_SDK_MOTORS; ++i) {
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

    lowcmd_publisher->Write(low_cmd);
}

void LocomotionPolicyController::runPolicy()
{
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    std::array<int64_t, 2> input_shape{1, NUM_STACKED_OBS};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        policy_buffers_.stacked_obs.data(),
        policy_buffers_.stacked_obs.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {"obs"};
    const char* output_names[] = {"actions"};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    for (int i = 0; i < NUM_ACTIONS; ++i) {
        policy_buffers_.raw_action[i] = output_data[i];
    }
}

void LocomotionPolicyController::postProcessAction()
{
    policy_buffers_.full_action.fill(0.0f);

    // Insert policy actions into full action vector.
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        const int motor_idx = action_motor_indices[i];
        policy_buffers_.full_action[motor_idx] = policy_buffers_.raw_action[i];
    }

    std::array<float, G1_NUM_MOTOR> old_order_action{};
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        old_order_action[i] =
            policy_buffers_.full_action[old_action_motor_indices[i]];
    }

    policy_buffers_.delayed_action = old_order_action;

    // Convert normalized action to joint targets.
    policy_buffers_.target_q = default_q_;

    for (int i = 0; i < NUM_ACTIONS; ++i) {
        const int motor_idx = action_motor_indices[i];
        const int old_order_idx = action_old_order_indices[i];

        float action = clip(
            policy_buffers_.delayed_action[old_order_idx],
            -ACTION_CLIP,
            ACTION_CLIP
        );

        policy_buffers_.target_q[motor_idx] =
            default_q_[motor_idx] + ACTION_SCALE * action;
    }

    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        policy_buffers_.target_q[i] = clip(
            policy_buffers_.target_q[i],
            joint_position_min[i],
            joint_position_max[i]
        );
    }
}

LocomotionPolicyController::LocomotionPolicyController(
    std::string model_path,
    double dt
)
    : model_path_(std::move(model_path)),
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
    arm_sdk_subscriber.reset(new ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_>(TOPIC_ARM_SDK));
    arm_sdk_subscriber->InitChannel(std::bind(&LocomotionPolicyController::ArmSdkMessageHandler, this, std::placeholders::_1), 1);
    velocity_subscriber.reset(new ChannelSubscriber<std_msgs::msg::dds_::String_>("rt/run_command/cmd"));
    velocity_subscriber->InitChannel(std::bind(&LocomotionPolicyController::RunCommandMessageHandler, this, std::placeholders::_1), 1);
    
    /*loop publishing thread*/
    lowCmdWriteThreadPtr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, int(dt_ * 1000000), &LocomotionPolicyController::update, this);
}

void LocomotionPolicyController::LoadONNX(){
    session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
    // The model input is NUM_OBS values stacked NUM_OBS_HISTORY times.
    std::cout << "Model loaded successfully: " << model_path_ << std::endl;

}

void LocomotionPolicyController::InitLowCmd()
{
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

void LocomotionPolicyController::RunCommandMessageHandler(const void *message)
{
    auto command =
        (const std_msgs::msg::dds_::String_*)message;
    std::string cmd_str(command->data().c_str());
    std::istringstream iss(cmd_str);
    double vx, vy, wz, height;
    if (iss >> vx >> vy >> wz >> height) {
        setCommand(vx, vy, wz, height);
    } else {
        std::cerr << "Invalid command format: " << cmd_str << std::endl;
    }
}

void LocomotionPolicyController::setCommand(double vx, double vy, double wz, double height)
{
    command_[0] = static_cast<float>(vx);
    command_[1] = static_cast<float>(vy);
    command_[2] = static_cast<float>(wz);
    command_[3] = static_cast<float>(height);
    std::cout << "Received command: vx=" << vx << ", vy=" << vy << ", wz=" << wz << ", height=" << height << std::endl;
}

// void 

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

    for (int i = 0; i < 3; ++i) {
        policy_buffers_.current_obs[k++] = gyro_[i];
    }

    for (int i = 0; i < 3; ++i) {
        policy_buffers_.current_obs[k++] = projected_gravity_[i];
    }

    for (int i = 0; i < 4; ++i) {
        policy_buffers_.current_obs[k++] = command_[i];
    }

    for (int motor_idx : obs_motor_indices) {
        policy_buffers_.current_obs[k++] =
            motor_state_.q[motor_idx] - default_q_[motor_idx];
    }

    for (int motor_idx : obs_motor_indices) {
        policy_buffers_.current_obs[k++] =
            motor_state_.dq[motor_idx] - default_dq_[motor_idx];
    }
    //
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        policy_buffers_.current_obs[k++] = policy_buffers_.delayed_action[i];
    }

    if (k != NUM_OBS) {
        throw std::runtime_error("Observation size mismatch");
    }

    for (float& x : policy_buffers_.current_obs) {
        x = clip(x, -OBS_CLIP, OBS_CLIP);
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
       
    if (argc < 3)
    {
        ChannelFactory::Instance()->Init(1, "lo");
    }
    else
    {
        ChannelFactory::Instance()->Init(0, argv[2]);
    }
    std::cout << "Press enter to start";
    std::cin.get();
    LocomotionPolicyController controller(argv[1]);
    controller.init();

    while (true)
    {
        sleep(1000);
    }

    return 0;
}
