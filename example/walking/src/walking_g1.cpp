#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <string>
#include <boost/circular_buffer.hpp>
#include <Eigen/Dense>

#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/robots/g1/g1.h>


#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

using namespace unitree::common;
using namespace unitree::robot;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_IMUSTATE "rt/secondary_imu"

const int G1_NUM_MOTOR = 29;

constexpr int NUM_OBS = 91;
constexpr int NUM_OBS_HISTORY = 10;
constexpr int NUM_STACKED_OBS = NUM_OBS * NUM_OBS_HISTORY;

constexpr int NUM_ACTIONS = 12;
constexpr int NUM_OBS_JOINTS = 26;
constexpr int ACTION_DELAY = 5;

constexpr float OBS_CLIP = 100.0f;
constexpr float ACTION_CLIP = 100.0f;
constexpr float ACTION_SCALE = 0.25f;



// Stiffness for all G1 Joints
std::array<float, G1_NUM_MOTOR> Kp{
    60, 60, 60, 100, 40, 40,      // legs
    60, 60, 60, 100, 40, 40,      // legs
    60, 40, 40,                   // waist
    40, 40, 40, 40,  40, 40, 40,  // arms
    40, 40, 40, 40,  40, 40, 40   // arms
};

// Damping for all G1 Joints
std::array<float, G1_NUM_MOTOR> Kd{
    1, 1, 1, 2, 1, 1,     // legs
    1, 1, 1, 2, 1, 1,     // legs
    1, 1, 1,              // waist
    1, 1, 1, 1, 1, 1, 1,  // arms
    1, 1, 1, 1, 1, 1, 1   // arms
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

const std::array<int, 26> obs_motor_indices = {
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

std::array<float, G1_NUM_MOTOR> default_q_{
    // legs
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
    std::array<float, 91> current_obs{};
    boost::circular_buffer<std::array<float, 91>> obs_history{
        10
    };

    boost::circular_buffer<std::array<float, 29>> action_delay_buffer{
        29
    };

    std::array<float, 910> stacked_obs{};
    std::array<float, 12> raw_action{};
    std::array<float, 29> full_action{};
    std::array<float, 29> delayed_action{};
    std::array<float, 29> target_q{};
};


float clip(float x, float lo, float hi) {
    return std::max(lo, std::min(x, hi));
}

void clipArray(std::array<float, 91>& obs) {
    for (auto& x : obs) {
        x = clip(x, -100.0f, 100.0f);
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
    void setCommand(double vx, double vy, double wz, double height);
    void update();
    void publishCommand();

private:
    void InitLowCmd();
    void LoadONNX();
    void LowStateMessageHandler(const void *messages);
    void IMUMessageHandler(const void *messages);
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
    unitree_hg::msg::dds_::LowState_ low_state{}; // default init
    

    std::array<float, 4> command_{
        0.0f, 0.0f, 0.0f, 0.0f
    }; 


    // Controller
    PolicyBuffers policy_buffers_;

    // IMU state
    std::array<float, 3> gyro_{0.0f, 0.0f, 0.0f};
    Eigen::Quaterniond imu_quaternion_{1.0, 0.0, 0.0, 0.0};
    std::array<float, 3> projected_gravity_{0.0f, 0.0f, 0.0f};

    std::array<float, G1_NUM_MOTOR> default_q_{};
    std::array<float, G1_NUM_MOTOR> default_dq_{};
    
    
    /*publisher*/
    ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher;
    /*subscriber*/
    ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_hg::msg::dds_::IMUState_> imustate_subscriber;

    /*LowCmd write thread*/
    ThreadPtr lowCmdWriteThreadPtr;

    std::mutex low_state_mutex_;
    std::atomic<bool> low_state_received_{false};

    std::mutex imu_state_mutex_;
};

void LocomotionPolicyController::update()
{
    if (!low_state_received_) {
        return;
    }

    copyLowStateToMotorState();

    buildCurrentObservation();

    policy_buffers_.obs_history.push_back(policy_buffers_.current_obs);

    if (!buildStackedObservation()) {
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
    policy_buffers_.full_action.fill(0.0f);

    // Insert 12 policy actions into 29-DoF action vector.
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        const int motor_idx = action_motor_indices[i];
        policy_buffers_.full_action[motor_idx] = policy_buffers_.raw_action[i];
    }

    // Add to delay buffer.
    policy_buffers_.action_delay_buffer.push_back(policy_buffers_.full_action);

    if (policy_buffers_.action_delay_buffer.size() < ACTION_DELAY) {
        policy_buffers_.delayed_action = policy_buffers_.full_action;
    } else {
        policy_buffers_.delayed_action =
            policy_buffers_.action_delay_buffer.front();
    }

    // Convert normalized action to joint targets.
    policy_buffers_.target_q = default_q_;

    for (int i = 0; i < NUM_ACTIONS; ++i) {
        const int motor_idx = action_motor_indices[i];

        float action = clip(
            policy_buffers_.delayed_action[motor_idx],
            -ACTION_CLIP,
            ACTION_CLIP
        );

        policy_buffers_.target_q[motor_idx] =
            default_q_[motor_idx] + ACTION_SCALE * action;
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
    
    imustate_subscriber.reset(new ChannelSubscriber<unitree_hg::msg::dds_::IMUState_>(TOPIC_IMUSTATE));
    imustate_subscriber->InitChannel(std::bind(&LocomotionPolicyController::IMUMessageHandler, this, std::placeholders::_1), 1);

    /*loop publishing thread*/
    // lowCmdWriteThreadPtr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, int(dt_ * 1000000), &LocomotionPolicyController::LowCmdWrite, this);
}

void LocomotionPolicyController::LoadONNX(){
    session_ = Ort::Session(env_, model_path_.c_str(), session_options_);
    // The model has an input size of 910
    // [angular_vel[3], projected_gravity[3,6], command[3,9], joint_pos[26,35], joint_vel[26,61], last_action[29,90]] * 10
    std::cout << "Model loaded successfully: " << model_path_ << std::endl;

}

void LocomotionPolicyController::InitLowCmd()
{
    // low_cmd.head()[0] = 0xFE;
    // low_cmd.head()[1] = 0xEF;
    // low_cmd.level_flag() = 0xFF;
    // low_cmd.gpio() = 0;
    //
    // for (int i = 0; i < 20; i++)
    // {
    //     low_cmd.motor_cmd()[i].mode() = (0x01); // motor switch to servo (PMSM) mode
    //     low_cmd.motor_cmd()[i].q() = (PosStopF);
    //     low_cmd.motor_cmd()[i].kp() = (0);
    //     low_cmd.motor_cmd()[i].dq() = (VelStopF);
    //     low_cmd.motor_cmd()[i].kd() = (0);
    //     low_cmd.motor_cmd()[i].tau() = (0);
    // }
}

void LocomotionPolicyController::LowStateMessageHandler(const void *message)
{
    auto state =
        (const unitree_hg::msg::dds_::LowState_*)message;
    {
        std::lock_guard<std::mutex> lock(low_state_mutex_);
        low_state = *state;
    }
    low_state_received_ = true;
}

void LocomotionPolicyController::IMUMessageHandler(const void *message)
{
    auto imu = (const unitree_hg::msg::dds_::IMUState_*)message;
    {
        std::lock_guard<std::mutex> lock(imu_state_mutex_);
        gyro_[0] = imu->gyroscope()[0];
        gyro_[1] = imu->gyroscope()[1];
        gyro_[2] = imu->gyroscope()[2];

        imu_quaternion_.w() = imu->quaternion()[0];
        imu_quaternion_.x() = imu->quaternion()[1];
        imu_quaternion_.y() = imu->quaternion()[2];
        imu_quaternion_.z() = imu->quaternion()[3];

        Eigen::Vector3d gravity(0.0, 0.0, -1.0);
        Eigen::Matrix3d rotation_matrix = imu_quaternion_.toRotationMatrix();

        Eigen::Vector3d projected_gravity = rotation_matrix.transpose() * gravity;
        projected_gravity_[0] = projected_gravity[0];
        projected_gravity_[1] = projected_gravity[1];
        projected_gravity_[2] = projected_gravity[2];

    }
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

    if (k != 91) {
        throw std::runtime_error("Observation size mismatch");
    }

    for (float& x : policy_buffers_.current_obs) {
        x = clip(x, -100.0f, 100.0f);
    }
}

bool LocomotionPolicyController::buildStackedObservation(){
    if (policy_buffers_.obs_history.size() < 10) {
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

// void Custom::LowCmdWrite()
// {
//
//     runing_time += dt;
//     if (runing_time < 3.0)
//     {
//         // Stand up in first 3 second
//
//         // Total time for standing up or standing down is about 1.2s
//         phase = tanh(runing_time / 1.2);
//         for (int i = 0; i < 12; i++)
//         {
//             low_cmd.motor_cmd()[i].q() = phase * stand_up_joint_pos[i] + (1 - phase) * stand_down_joint_pos[i];
//             low_cmd.motor_cmd()[i].dq() = 0;
//             low_cmd.motor_cmd()[i].kp() = phase * 50.0 + (1 - phase) * 20.0;
//             low_cmd.motor_cmd()[i].kd() = 3.5;
//             low_cmd.motor_cmd()[i].tau() = 0;
//         }
//     }
//     else
//     {
//         // Then stand down
//         phase = tanh((runing_time - 3.0) / 1.2);
//         for (int i = 0; i < 12; i++)
//         {
//             low_cmd.motor_cmd()[i].q() = phase * stand_down_joint_pos[i] + (1 - phase) * stand_up_joint_pos[i];
//             low_cmd.motor_cmd()[i].dq() = 0;
//             low_cmd.motor_cmd()[i].kp() = 50;
//             low_cmd.motor_cmd()[i].kd() = 3.5;
//             low_cmd.motor_cmd()[i].tau() = 0;
//         }
//     }
//
//     low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
//     lowcmd_publisher->Write(low_cmd);
// }

int main(int argc, const char **argv)
{
       
    if (argc < 2)
    {
        ChannelFactory::Instance()->Init(1, "lo");
    }
    else
    {
        ChannelFactory::Instance()->Init(0, argv[1]);
    }
    std::cout << "Press enter to start";
    std::cin.get();
    LocomotionPolicyController controller;
    controller.init();

    while (true)
    {
        controller.update();
        usleep(20000);  // 50 Hz
    }

    return 0;
}
