#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rga/im2d.h>
#include <rga/RgaApi.h>

// 辅助函数：根据 RGA 硬件要求对齐 stride
#define RGA_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

int rga_scale_yolo(int src_fd, int dst_fd, int src_w, int src_h) {
    // int src_w = 1920, src_h = 1080;
    int dst_w = 640, dst_h = 640;
    
    // // RGA 通常要求 NV12 的 width 按 16 对齐，height 按 2 (或 8) 对齐
    int src_wstride = RGA_ALIGN(src_w, 16);
    int src_hstride = RGA_ALIGN(src_h, 2);
    
    int dst_wstride = RGA_ALIGN(dst_w, 16);
    int dst_hstride = RGA_ALIGN(dst_h, 2);

    // // 1. 分配虚拟内存 (注意使用 stride 计算实际大小)
    // void *src_ptr = malloc(src_wstride * src_hstride * 3 / 2);
    // void *dst_ptr = malloc(dst_wstride * dst_hstride * 3); // RGB888
    
    // if (!src_ptr || !dst_ptr) {
    //     printf("内存分配失败！\n");
    //     return -1;
    // }

    // (此处省略将摄像头数据拷入 src_ptr 的代码)

    // 2. 将dmabuff fd包装成 RGA Buffer (显式传入 stride)
    rga_buffer_t src = wrapbuffer_fd(src_fd, src_w, src_h, RK_FORMAT_YCbCr_420_SP, src_wstride, src_hstride);
    rga_buffer_t dst = wrapbuffer_fd(dst_fd, dst_w, dst_h, RK_FORMAT_RGB_888, dst_wstride, dst_hstride);

    // 空的矩形区域，表示不进行局部裁剪，处理全图
im_rect src_rect = {0, 0, src_w, src_h};
im_rect dst_rect = {0, 0, dst_w, dst_h};
im_rect pat_rect = {0};

rga_buffer_t pat;
memset(&pat, 0, sizeof(rga_buffer_t));
im_opt_t opt;
memset(&opt, 0, sizeof(opt));

    // 3. 检查 RGA 硬件是否支持该操作
    IM_STATUS check_ret = imcheck(src, dst, src_rect, dst_rect);
    if (check_ret == IM_STATUS_NOERROR) {
        
        // 4. 使用 improcess 或 imcvtcolor 进行处理。
        // 这里显式加上 IM_SYNC 标志，要求同步等待 RGA 硬件处理完成
IM_STATUS status = improcess(
    src,
    dst,
    pat,
    src_rect,
    dst_rect,
    pat_rect,
    IM_SYNC
);
        
        if (status == IM_STATUS_SUCCESS) {
            // printf("RGA 转换并缩放成功！\n");
        } else {
            printf("RGA 失败，错误码：%d, 错误信息：%s\n", status, imStrError((IM_STATUS)status));
            return -1;
        }
    } else {
        printf("RGA 硬件不支持当前的输入输出参数组合！\n");
        printf("imcheck failed: %d, %s\n", check_ret, imStrError(check_ret));
        return -1;
    }

    return 0;
}
