#pragma once

#include <array>
#include <boost/program_options.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace param
{

struct ObjectPoseConfig
{
    std::string body;
    std::string topic;
    std::string frame;
    int rate_hz;
};

struct SceneResetPoseConfig
{
    std::array<double, 3> position = {1.5, 0.0, 0.45};
    std::array<double, 3> position_margin = {0.0, 0.0, 0.0};
    double yaw = 0.0;
    double yaw_margin = 0.0;
};

inline struct SimulationConfig
{
    std::string robot;
    std::filesystem::path robot_scene;

    int domain_id;
    std::string interface;

    int use_joystick;
    std::string joystick_type;
    std::string joystick_device;
    int joystick_bits;

    int print_scene_information;

    int enable_elastic_band;
    double elastic_band_height = 3.0;
    std::string elastic_band_topic = "rt/elastic_band";
    int band_attached_link = 0;

    int enable_scene_reset = 0;
    std::string scene_reset_topic = "rt/reset_category";
    std::string dolly_reset_topic = "rt/reset_dolly";
    std::string scene_reset_body = "dolly";
    SceneResetPoseConfig scene_reset_pose;

    int publish_object_pose = 0;
    std::string object_pose_body = "dolly";
    std::string object_pose_topic = "rt/object_pose";
    std::string object_pose_frame = "world";
    int object_pose_rate_hz = 50;
    std::vector<ObjectPoseConfig> object_pose_publishers;

    int publish_camera = 0;
    std::string camera_name = "head_camera";
    int camera_width = 640;
    int camera_height = 480;
    int camera_rate_hz = 15;
    int camera_zmq_port = 55555;
    int camera_request_port = 60000;
    int camera_jpeg_quality = 85;

    void load_from_yaml(const std::string &filename)
    {
        auto cfg = YAML::LoadFile(filename);
        try
        {
            robot = cfg["robot"].as<std::string>();
            robot_scene = cfg["robot_scene"].as<std::string>();
            domain_id = cfg["domain_id"].as<int>();
            interface = cfg["interface"].as<std::string>();
            use_joystick = cfg["use_joystick"].as<int>();
            joystick_type = cfg["joystick_type"].as<std::string>();
            joystick_device = cfg["joystick_device"].as<std::string>();
            joystick_bits = cfg["joystick_bits"].as<int>();
            print_scene_information = cfg["print_scene_information"].as<int>();
            enable_elastic_band = cfg["enable_elastic_band"].as<int>();
            if (cfg["elastic_band_height"]) {
                elastic_band_height = cfg["elastic_band_height"].as<double>();
            }
            if (cfg["elastic_band_topic"]) {
                elastic_band_topic = cfg["elastic_band_topic"].as<std::string>();
            }
            if (cfg["enable_scene_reset"]) {
                enable_scene_reset = cfg["enable_scene_reset"].as<int>();
            }
            if (cfg["scene_reset_topic"]) {
                scene_reset_topic = cfg["scene_reset_topic"].as<std::string>();
            }
            if (cfg["dolly_reset_topic"]) {
                dolly_reset_topic = cfg["dolly_reset_topic"].as<std::string>();
            }
            if (cfg["scene_reset_body"]) {
                scene_reset_body = cfg["scene_reset_body"].as<std::string>();
            }
            if (cfg["scene_reset_pose"]) {
                const auto pose = cfg["scene_reset_pose"];
                auto read_vector3 = [](const YAML::Node& node, const std::string& name) {
                    if (!node || !node.IsSequence() || node.size() != 3) {
                        throw std::runtime_error(name + " must contain exactly [x, y, z]");
                    }
                    std::array<double, 3> value = {
                        node[0].as<double>(), node[1].as<double>(), node[2].as<double>()};
                    for (const double component : value) {
                        if (!std::isfinite(component)) {
                            throw std::runtime_error(name + " must contain only finite values");
                        }
                    }
                    return value;
                };

                scene_reset_pose.position = read_vector3(
                    pose["position"], "scene_reset_pose.position");
                if (pose["position_margin"]) {
                    scene_reset_pose.position_margin = read_vector3(
                        pose["position_margin"], "scene_reset_pose.position_margin");
                }
                for (const double margin : scene_reset_pose.position_margin) {
                    if (margin < 0.0) {
                        throw std::runtime_error(
                            "scene_reset_pose.position_margin values must be non-negative");
                    }
                }

                scene_reset_pose.yaw = pose["yaw"] ? pose["yaw"].as<double>() : 0.0;
                scene_reset_pose.yaw_margin =
                    pose["yaw_margin"] ? pose["yaw_margin"].as<double>() : 0.0;
                if (!std::isfinite(scene_reset_pose.yaw) ||
                    !std::isfinite(scene_reset_pose.yaw_margin) ||
                    scene_reset_pose.yaw_margin < 0.0) {
                    throw std::runtime_error(
                        "scene_reset_pose yaw values must be finite and yaw_margin must be non-negative");
                }
            }
            if (cfg["publish_object_pose"]) {
                publish_object_pose = cfg["publish_object_pose"].as<int>();
            }
            if (cfg["object_pose_body"]) {
                object_pose_body = cfg["object_pose_body"].as<std::string>();
            }
            if (cfg["object_pose_topic"]) {
                object_pose_topic = cfg["object_pose_topic"].as<std::string>();
            }
            if (cfg["object_pose_frame"]) {
                object_pose_frame = cfg["object_pose_frame"].as<std::string>();
            }
            if (cfg["object_pose_rate_hz"]) {
                object_pose_rate_hz = cfg["object_pose_rate_hz"].as<int>();
            }
            if (cfg["publish_camera"]) {
                publish_camera = cfg["publish_camera"].as<int>();
            }
            if (cfg["camera_name"]) {
                camera_name = cfg["camera_name"].as<std::string>();
            }
            if (cfg["camera_width"]) {
                camera_width = cfg["camera_width"].as<int>();
            }
            if (cfg["camera_height"]) {
                camera_height = cfg["camera_height"].as<int>();
            }
            if (cfg["camera_rate_hz"]) {
                camera_rate_hz = cfg["camera_rate_hz"].as<int>();
            }
            if (cfg["camera_zmq_port"]) {
                camera_zmq_port = cfg["camera_zmq_port"].as<int>();
            }
            if (cfg["camera_request_port"]) {
                camera_request_port = cfg["camera_request_port"].as<int>();
            }
            if (cfg["camera_jpeg_quality"]) {
                camera_jpeg_quality = cfg["camera_jpeg_quality"].as<int>();
            }
            if (cfg["object_pose_publishers"]) {
                object_pose_publishers.clear();
                for (const auto& node : cfg["object_pose_publishers"]) {
                    ObjectPoseConfig pose;
                    pose.body = node["body"].as<std::string>();
                    pose.topic = node["topic"] ? node["topic"].as<std::string>() : "rt/" + pose.body + "_pose";
                    pose.frame = node["frame"] ? node["frame"].as<std::string>() : object_pose_frame;
                    pose.rate_hz = node["rate_hz"] ? node["rate_hz"].as<int>() : object_pose_rate_hz;
                    object_pose_publishers.push_back(pose);
                }
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }
} config;

/* ---------- Command Line Parameters ---------- */
namespace po = boost::program_options;

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv)
{
    po::options_description desc("Unitree Mujoco");
    desc.add_options()
        ("help,h", "Show help message")
        ("domain_id,i", po::value<int>(&config.domain_id), "DDS domain ID; -i 0")
        ("network,n", po::value<std::string>(&config.interface), "DDS network interface; -n eth0")
        ("robot,r", po::value<std::string>(&config.robot), "Robot type; -r go2")
        ("scene,s", po::value<std::filesystem::path>(&config.robot_scene), "Robot scene file; -s scene_terrain.xml")
        ("publish_object_pose", po::value<int>(&config.publish_object_pose), "Publish a MuJoCo body pose over DDS; 0 or 1")
        ("object_pose_body", po::value<std::string>(&config.object_pose_body), "MuJoCo body name to publish")
        ("object_pose_topic", po::value<std::string>(&config.object_pose_topic), "DDS topic for the object PoseStamped")
        ("object_pose_frame", po::value<std::string>(&config.object_pose_frame), "Frame id for the object PoseStamped")
        ("object_pose_rate_hz", po::value<int>(&config.object_pose_rate_hz), "Object pose publishing rate in Hz")
        ("publish_camera", po::value<int>(&config.publish_camera), "Publish a MuJoCo camera through ZMQ; 0 or 1")
        ("camera_name", po::value<std::string>(&config.camera_name), "MuJoCo camera name to render")
        ("camera_width", po::value<int>(&config.camera_width), "Camera render width")
        ("camera_height", po::value<int>(&config.camera_height), "Camera render height")
        ("camera_rate_hz", po::value<int>(&config.camera_rate_hz), "Camera publishing rate in Hz")
        ("camera_zmq_port", po::value<int>(&config.camera_zmq_port), "ZMQ JPEG PUB port")
        ("camera_request_port", po::value<int>(&config.camera_request_port), "TeleImager configuration REP port")
        ("camera_jpeg_quality", po::value<int>(&config.camera_jpeg_quality), "JPEG quality from 1 to 100")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    return vm;
}

}
