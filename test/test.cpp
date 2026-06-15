#include "vision_pipeline.hpp"
#include <thread>


int main(void) {
    setlocale(LC_ALL, "zh_CN.UTF-8");
    
    std::unique_ptr<VisionPipeline> v = std::make_unique<VisionPipeline>();
    v->init();
    v->start();


    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    

    return 0;
}