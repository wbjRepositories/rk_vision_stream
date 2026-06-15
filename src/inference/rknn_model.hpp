#pragma once

#include <mutex>
#include <queue>
#include <vector>
#include <stdint.h>
#include "rknn_api.h"
#include "postprocess.h"


struct BoundingBox {
    int x1, y1, x2, y2;
    int class_id;
    float confidence;
};


class rknn_model {
private:
    rknn_context ctx = 0;
    rknn_tensor_mem *input_mem = nullptr;
    rknn_tensor_attr input_attr{};
    rknn_input_output_num io_num;
    std::vector<rknn_tensor_attr> output_attrs;
    std::vector<rknn_tensor_mem *> output_mem;
    int channel = 3;
    int width = 0;
    int height = 0;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;

public:
    int init(void);
    int run(void);
    void destroy_model(void);
    int get_dma_fd(void);
};