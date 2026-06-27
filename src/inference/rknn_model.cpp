#include "rknn_model.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "rknn_api.h"
#include "postprocess.h"
#include "config.h"

#include <queue>
#include <mutex>
std::queue<BoundingBox> g_bb_q;
std::mutex g_queue_mutex;


static void print_tensor_attr(const char *prefix, const rknn_tensor_attr *attr) {
    printf("%s index=%u name=%s n_dims=%u dims=[", prefix, attr->index, attr->name, attr->n_dims);
    for (uint32_t i = 0; i < attr->n_dims; ++i) {
        printf("%u%s", attr->dims[i], (i + 1 == attr->n_dims) ? "" : ",");
    }
    printf("] n_elems=%u size=%u size_with_stride=%u fmt=%s type=%s qnt=%s zp=%d scale=%f w_stride=%u h_stride=%u pass_through=%u\n",
           attr->n_elems, attr->size, attr->size_with_stride, get_format_string(attr->fmt),
           get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale,
           attr->w_stride, attr->h_stride, attr->pass_through);
}

static int get_output_grid_size(const rknn_tensor_attr &attr) {
    if (attr.n_dims < 4) {
        return 0;
    }

    if (attr.fmt == RKNN_TENSOR_NCHW) {
        return attr.dims[2] * attr.dims[3];
    }
    if (attr.fmt == RKNN_TENSOR_NHWC) {
        return attr.dims[1] * attr.dims[2];
    }

    return 0;
}

int rknn_model::init(void) {
    int ret = 0;

    // ==========================================
    // 第一步：初始化 (rknn_init)
    // ==========================================

    int model_size = 0;
    
    // 调用 rknn_init，把模型加载进内存并分配给NPU
    ret = rknn_init(&ctx, (char *)MODEL_PATH, model_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }


    // ==========================================
    // 第二步：查询模型信息 (rknn_query)
    // ==========================================
    // 获取输入输出数量
    
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    printf("模型有 %d 个输入, %d 个输出\n", io_num.n_input, io_num.n_output);

    // 获取输入属性（比如分辨率，通道数，数据格式等）
    std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        print_tensor_attr("input", &input_attrs[i]);
    }

    input_attr = input_attrs[0];
    if (input_attr.fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attr.dims[1];
        height = input_attr.dims[2];
        width = input_attr.dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attr.dims[1];
        width = input_attr.dims[2];
        channel = input_attr.dims[3];
    }

    rknn_tensor_attr user_input_attr = input_attr;
    user_input_attr.type = RKNN_TENSOR_UINT8;
    user_input_attr.fmt = RKNN_TENSOR_NHWC;
    user_input_attr.pass_through = 0;
    user_input_attr.w_stride = width;
    user_input_attr.h_stride = height;

    input_mem = rknn_create_mem(ctx, width * height * channel);
    if (input_mem == nullptr) {
        printf("rknn_create_mem (input) failed!\n");
        return -1;
    }
    // 将这块内存设置为模型的输入
    ret = rknn_set_io_mem(ctx, input_mem, &user_input_attr);
    if (ret < 0) {
        printf("rknn_set_io_mem (input) failed! ret=%d\n", ret);
        return -1;
    }

    
    // rknn_tensor_attr output_attrs[io_num.n_output]; 
    output_attrs.resize(io_num.n_output);
    output_mem.resize(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        print_tensor_attr("output", &output_attrs[i]);

        output_mem[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (output_mem[i] == nullptr) {
            printf("rknn_create_mem (output %d) failed!\n", i);
            return -1;
        }

        ret = rknn_set_io_mem(ctx, output_mem[i], &output_attrs[i]);
        if (ret < 0) {
            printf("rknn_set_io_mem (output %d) failed! ret=%d\n", i, ret);
            return -1;
        }
    }

    return 0;
}

int rknn_model::run(void) {
    int ret = 0;
    // ==========================================
    // 第四步：执行推理 (rknn_run)
    // ==========================================
    // NPU 开始干活，默认是阻塞的，跑完才返回

    rknn_mem_sync(ctx, input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);

    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return ret;
    }

    for (int i = 0; i < io_num.n_output; ++i) {
        rknn_mem_sync(ctx, output_mem[i], RKNN_MEMORY_SYNC_FROM_DEVICE);
    }

    // 后处理
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    int output_order[3] = {0, 1, 2};
    if (io_num.n_output >= 3) {
        std::vector<int> order(io_num.n_output);
        for (int i = 0; i < io_num.n_output; ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [this](int lhs, int rhs) {
            return get_output_grid_size(output_attrs[lhs]) > get_output_grid_size(output_attrs[rhs]);
        });
        output_order[0] = order[0];
        output_order[1] = order[1];
        output_order[2] = order[2];
    }
    // printf("post_process output order: stride8<-output%d stride16<-output%d stride32<-output%d\n",
    //        output_order[0], output_order[1], output_order[2]);

    for (int k = 0; k < 3; ++k)
    {
        int i = output_order[k];
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    float scale_w = (float)640.0 / 3840.0;
    float scale_h = (float)640.0 / 2160.0;

    post_process((int8_t *)output_mem[output_order[0]]->virt_addr, (int8_t *)output_mem[output_order[1]]->virt_addr, (int8_t *)output_mem[output_order[2]]->virt_addr, height, width,
    box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    // printf("detect_result_group.count=%d\n",detect_result_group.count);
    // 画框和概率
    char text[256];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        // sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        // printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
        // det_result->box.right, det_result->box.bottom, det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        printf("push：x1=%d,y1=%d\nx2=%d,y2=%d\n",x1,y1,x2,y2);
        struct BoundingBox bb;
        bb.x1 = x1;
        bb.y1 = y1;
        bb.x2 = x2;
        bb.y2 = y2;

        std::lock_guard<std::mutex> lock(g_queue_mutex);
        g_bb_q.push(bb);
    }

    return 0;
}

void rknn_model::destroy_model(void) {
    rknn_destroy_mem(ctx, input_mem);
    for (int i = 0; i < io_num.n_output; i++) {
        rknn_destroy_mem(ctx, output_mem[i]);
    }
    
    // 销毁上下文
    rknn_destroy(ctx);
}

int rknn_model::get_dma_fd(void) {
    if (ctx == 0) {
        printf("rknn 初始化失败!\n");
        return -1;
    }
    return input_mem->fd;
}
