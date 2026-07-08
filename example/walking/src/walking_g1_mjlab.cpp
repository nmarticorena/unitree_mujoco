#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <boost/circular_buffer.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/robots/g1/g1.h>
#include <unitree/dds_wrapper/robots/go2/go2.h>


#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

using namespace unitree::common;
using namespace unitree::robot;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_SPORTMODE "rt/sportmodestate"

const int G1_NUM_MOTOR = 29;

constexpr int NUM_OBS = 99;
constexpr int NUM_OBS_HISTORY = 1;
constexpr int NUM_STACKED_OBS = NUM_OBS * NUM_OBS_HISTORY;

constexpr int NUM_ACTIONS = G1_NUM_MOTOR;
constexpr int NUM_POLICY_JOINTS = G1_NUM_MOTOR;
constexpr int DEFAULT_POSE_RAMP_STEPS = 100;
constexpr int POLICY_WARMUP_STEPS = 20;
static_assert(NUM_ACTIONS == NUM_POLICY_JOINTS);
static_assert(NUM_OBS == 3 + 3 + 3 + 3 * NUM_POLICY_JOINTS + 3);

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
        std::string model_path = "mjlab.onnx",
        double dt = 0.02 // default 50Hz
    );

    void init();
    void setCommand(double vx, double vy, double wz);
    void update();
    void publishCommand();

private:
    void InitLowCmd();
    void LoadONNX();
    void LowStateMessageHandler(const void *messages);
    void SportModeMessageHandler(const void *messages);

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
    unitree_hg::msg::dds_::LowCmd_ low_cmd{};     // default init
    unitree_hg::msg::dds_::LowState_ low_state{}; // default init
    

    std::array<float, 3> command_{
        1.0f, 0.0f, 0.0f 
    }; 


    // Controller
    PolicyBuffers policy_buffers_;

    // IMU state
    std::array<float, 3> gyro_{0.0f, 0.0f, 0.0f};
    Eigen::Quaterniond imu_quaternion_{1.0, 0.0, 0.0, 0.0};
    std::array<float, 3> projected_gravity_{0.0f, 0.0f, 0.0f};

    // Velocity tracking
    std::array<float, 3> velocity_{0.0f, 0.0f, 0.0f};


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

    /*LowCmd write thread*/
    ThreadPtr lowCmdWriteThreadPtr;

    std::mutex low_state_mutex_;
    std::atomic<uint64_t> low_state_sequence_{0};
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
    sportmode_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_SPORTMODE));
    sportmode_subscriber->InitChannel(std::bind(&LocomotionPolicyController::SportModeMessageHandler, this, std::placeholders::_1), 1);
    
    /*loop publishing thread*/
    lowCmdWriteThreadPtr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, int(dt_ * 1000000), &LocomotionPolicyController::update, this);
}

void LocomotionPolicyController::setCommand(double vx, double vy, double wz)
{
    std::lock_guard<std::mutex> lock(low_state_mutex_);
    command_[0] = static_cast<float>(vx);
    command_[1] = static_cast<float>(vy);
    command_[2] = static_cast<float>(wz);
}

void LocomotionPolicyController::LoadONNX(){
    session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
    // The model input is NUM_OBS values stacked NUM_OBS_HISTORY times.
    std::cout << "Model loaded successfully: " << model_path_ << std::endl;

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
