#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/dds_wrapper/robots/g1/g1.h>

#include <array>
#include <csignal>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace unitree::robot;

namespace {

constexpr char TOPIC_LOWCMD[] = "rt/lowcmd";
constexpr char TOPIC_ARM_SDK[] = "rt/arm_sdk";
constexpr int ARM_SDK_ENABLE_INDEX = 29;
constexpr float ARM_SDK_ENABLE_THRESHOLD = 0.5f;

volatile std::sig_atomic_t keep_running = 1;

void handleShutdownSignal(int)
{
    keep_running = 0;
}

constexpr std::array<int, 14> ARM_MOTOR_INDICES = {
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28
};

class ArmSdkUpperBodyRelay
{
public:
    void init()
    {
        lowcmd_publisher_.reset(
            new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(TOPIC_LOWCMD)
        );
        lowcmd_publisher_->InitChannel();

        arm_sdk_subscriber_.reset(
            new ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_>(TOPIC_ARM_SDK)
        );
        arm_sdk_subscriber_->InitChannel(
            std::bind(
                &ArmSdkUpperBodyRelay::armSdkMessageHandler,
                this,
                std::placeholders::_1
            ),
            1
        );

        // Clear any command retained by the simulator before this relay started.
        lowcmd_publisher_->Write(unitree_hg::msg::dds_::LowCmd_{});
    }

    void shutdown()
    {
        arm_sdk_subscriber_.reset();

        const unitree_hg::msg::dds_::LowCmd_ release_command{};
        for (int i = 0; i < 5; ++i) {
            lowcmd_publisher_->Write(release_command);
            usleep(20000);
        }
    }

private:
    void armSdkMessageHandler(const void* message)
    {
        const auto* arm_command =
            static_cast<const unitree_hg::msg::dds_::LowCmd_*>(message);
        unitree_hg::msg::dds_::LowCmd_ low_command{};

        if (arm_command->motor_cmd()[ARM_SDK_ENABLE_INDEX].q() >
            ARM_SDK_ENABLE_THRESHOLD) {
            for (const int motor_index : ARM_MOTOR_INDICES) {
                low_command.motor_cmd()[motor_index] =
                    arm_command->motor_cmd()[motor_index];
            }
        }

        lowcmd_publisher_->Write(low_command);
    }

    ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
    ChannelSubscriberPtr<unitree_hg::msg::dds_::LowCmd_> arm_sdk_subscriber_;
};

} // namespace

int main(int argc, const char** argv)
{
    const char* network_interface = nullptr;
    int dds_domain_id = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--dds-id") {
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
        } else if (network_interface == nullptr) {
            network_interface = argv[i];
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [network_interface] [--dds-id id]" << std::endl;
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

    ArmSdkUpperBodyRelay relay;
    std::signal(SIGINT, handleShutdownSignal);
    std::signal(SIGTERM, handleShutdownSignal);
    std::signal(SIGHUP, handleShutdownSignal);
    relay.init();
    std::cout << "Relaying arm_sdk to upper-body arm joints only" << std::endl;

    while (keep_running) {
        sleep(1);
    }
    relay.shutdown();

    return 0;
}
