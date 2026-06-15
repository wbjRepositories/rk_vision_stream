#pragma once

#include <memory>
#include <functional>
#include <vector>
#include <string>

// AI 推理结果的数据结构
// struct BoundingBox {
//     int x, y, width, height;
//     int class_id;
//     float confidence;
// };

class VisionPipeline {
public:
    VisionPipeline();
    ~VisionPipeline();

    // 禁用拷贝构造，防止底层资源被意外释放
    VisionPipeline(const VisionPipeline&) = delete;
    VisionPipeline& operator=(const VisionPipeline&) = delete;

    // 1. 生命周期管理
    bool init();          // 初始化 Pipeline，但不启动数据流
    bool start();         // 将 Pipeline 设置为 PLAYING 状态
    void stop();          // 停止推流和推理，释放资源

    // 2. 注册 AI 结果回调 (核心：把数据甩给外部)
    // using InferenceCallback = std::function<void(const std::vector<BoundingBox>&)>;
    // void set_inference_callback(InferenceCallback callback);

    // 3. WebRTC 信令交互接口 (供外部信令服务器调用)
    // void set_remote_description(const std::string& type, const std::string& sdp);
    // void add_ice_candidate(const std::string& candidate, int sdp_mline_index);

private:
    // PIMPL 惯用法：前向声明内部实现类
    class Impl;
    // 使用智能指针管理内部实现，外界看不到 Impl 的具体内容
    std::unique_ptr<Impl> pimpl_; 
};