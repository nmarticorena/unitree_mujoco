#pragma once

#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <vector>

namespace param
{

struct ObjectPoseConfig
{
    std::string body;
    std::string topic;
    std::string frame;
    int rate_hz;
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
    int band_attached_link = 0;

    int publish_object_pose = 0;
    std::string object_pose_body = "dolly";
    std::string object_pose_topic = "rt/object_pose";
    std::string object_pose_frame = "world";
    int object_pose_rate_hz = 50;
    std::vector<ObjectPoseConfig> object_pose_publishers;

    int publish_camera = 0;
    std::string camera_name = "head_camera";
    std::string camera_topic = "rt/head_camera/front_video";
    std::string camera_frame = "head_camera";
    int camera_width = 640;
    int camera_height = 480;
    int camera_rate_hz = 15;

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
            if (cfg["camera_topic"]) {
                camera_topic = cfg["camera_topic"].as<std::string>();
            }
            if (cfg["camera_frame"]) {
                camera_frame = cfg["camera_frame"].as<std::string>();
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
        ("publish_camera", po::value<int>(&config.publish_camera), "Publish a MuJoCo camera through DDS; 0 or 1")
        ("camera_name", po::value<std::string>(&config.camera_name), "MuJoCo camera name to render")
        ("camera_topic", po::value<std::string>(&config.camera_topic), "DDS topic for Go2FrontVideoData camera payload")
        ("camera_frame", po::value<std::string>(&config.camera_frame), "Frame id encoded in camera metadata")
        ("camera_width", po::value<int>(&config.camera_width), "Camera render width")
        ("camera_height", po::value<int>(&config.camera_height), "Camera render height")
        ("camera_rate_hz", po::value<int>(&config.camera_rate_hz), "Camera publishing rate in Hz")
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
