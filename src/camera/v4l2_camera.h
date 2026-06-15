#ifndef _V4L2_CAMERA_h_
#define _V4L2_CAMERA_h_

#include <linux/videodev2.h>

#ifdef __cplusplus
extern "C" {
#endif

// 适配多平面格式，YUV420M 最多 3 个平面 (Y, U, V)
#define MAX_V4L2_PLANES 3

// 单个平面的信息
struct dmabuf_plane {
    int     fd;      // 该平面的 DMA-BUF fd
    size_t  length;  // 该平面的内存大小
    void    *start;  // mmap 的虚拟地址 (如果纯硬件零拷贝，可为 NULL)
};

// 完整的一帧图像 buffer 管理结构
typedef struct dmabuf_buffer {
    int                 index;       // 对应 V4L2 的 buffer index
    int                 num_planes;  // 实际使用的平面数量
    struct dmabuf_plane planes[MAX_V4L2_PLANES]; // 包含的平面数组
}dmabuf;


int         v4l2_open_device(const char *pathname);
int         v4l2_set_format_mplane(int device_fd, unsigned int req_width, unsigned int req_height, unsigned int req_pixelformat);
int         v4l2_get_format_by_plane(int device_fd, int plane_index, unsigned int *stride, unsigned int *sizeimage);
int         v4l2_init_dmabuf(int device_fd, dmabuf *out_buffers, int req_buf_count, size_t buf_size);
int         v4l2_queue_frame_dmabuf(int device_fd, dmabuf *dmabuf_info);
dmabuf*     v4l2_dequeue_frame_dmabuf(int device_fd, dmabuf *dmabuf_info);
int         v4l2_start_capturing(int device_fd, unsigned int count, dmabuf *dmabuf_info);
void        v4l2_stop_capturing(int device_fd);
void        v4l2_close_device(int device_fd);

#ifdef __cplusplus
}
#endif

#endif