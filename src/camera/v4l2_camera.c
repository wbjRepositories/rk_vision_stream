#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include "v4l2_camera.h"
#include "dma_buffer.h"

/**
 * @brief 专家级 ioctl 封装函数（防御性编程）
 * @details 处理因为系统信号打断导致的 EINTR 错误，保证命令必定送达内核
 */
static int xioctl(int fh, int request, void *arg) {
    int r;
    do {
        // 执行 ioctl
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno); // 如果失败且原因是 EINTR，则继续重试
    return r;
}


/**
 * @brief 打开并校验设备节点
 * @param pathname 设备路径，例如 "/dev/video0"
 * @return 成功返回设备文件描述符(fd)，失败返回-1
 */
int v4l2_open_device(const char *pathname) {
    struct stat st = {0};
    int fd = -1;

    // 检查节点是否存在
    if (-1 == stat(pathname, &st)) {
        fprintf(stderr, "无法识别 %s : %d，%s", pathname, errno, strerror(errno));
        goto _ERROR;
    }
    // 检查是否是字符设备(V4L2 设备必须是字符设备)
    if (!S_ISCHR(st.st_mode)) {
        fprintf(stderr, "%s 不是一个字符设备 : %d，%s", pathname, errno, strerror(errno));
        goto _ERROR;
    }

    // 打开设备 (避坑：必须使用 O_RDWR)
    // O_NONBLOCK 表示非阻塞模式。配合 O_NONBLOCK 是最高效的工业级做法，避免进程在内核态死锁。
    fd = open(pathname, O_RDWR | O_NONBLOCK);
    if (-1 == fd) {
        fprintf(stderr, "无法打开 %s : %d，%s", pathname, errno, strerror(errno));
        goto _ERROR;
    }

    // 查询设备能力
    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            fprintf(stderr, "%s 不是一个标准的 V4L2 设备\n", pathname);
            goto _ERROR;
        } else {
            perror("VIDIOC_QUERYCAP 失败");
            goto _ERROR;
        }
    }

    // 打印设备基本信息
    printf("--- 设备信息 ---\n");
    printf("驱动名称: %s\n", cap.driver);
    printf("设备名称: %s\n", cap.card);
    printf("总线信息: %s\n", cap.bus_info);
    printf("V4L2版本: %u.%u.%u\n", (cap.version >> 16) & 0xFF, (cap.version >> 8) & 0xFF, cap.version & 0xFF);

    // 检验设备是否有我们需要的能力
    unsigned int caps = cap.capabilities;
    if (cap.capabilities & V4L2_CAP_DEVICE_CAPS) {
        caps = cap.device_caps;
        printf("检测到现代 V4L2 设备，使用 device_caps 校验\n");
    }

    if (caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
        printf("成功：设备支持 V4L2_CAP_VIDEO_CAPTURE_MPLANE (多平面捕获)\n");
    } else if (caps & V4L2_CAP_VIDEO_CAPTURE) {
        printf("成功：设备支持 V4L2_CAP_VIDEO_CAPTURE (单平面捕获)\n");
    } else {
        fprintf(stderr, "失败：%s 不是一个视频捕获设备 (既不支持单平面也不支持多平面)\n", pathname);
        goto _ERROR;
    }

    // 检查是否支持流式 I/O (STREAMING)
    if (!(caps & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "%s 不支持 Streaming I/O，不支持老旧的 read/write 模式\n", pathname);
        goto _ERROR;
    }

    printf("设备校验通过，支持视频捕获与流式I/O！\n");
    printf("----------------\n");

    return fd;

_ERROR:
    return fd;
}


/**
 * @brief 枚举并打印摄像头支持的所有多平面 (MPLANE) 像素格式
 * @param fd 打开的设备文件描述符
 */
void v4l2_print_formats_mplane(int v4l2_fd) {
    struct v4l2_fmtdesc fmtdesc;
    memset(&fmtdesc, 0, sizeof(fmtdesc));
    
    // 【核心改动 1】：类型必须更改为多平面宏
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmtdesc.index = 0; // 从 0 开始遍历

    printf("--- 摄像头支持的多平面像素格式 ---\n");
    while (xioctl(v4l2_fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
        char fourcc[5] = {0};
        fourcc[0] = (fmtdesc.pixelformat >> 0) & 0xFF;
        fourcc[1] = (fmtdesc.pixelformat >> 8) & 0xFF;
        fourcc[2] = (fmtdesc.pixelformat >> 16) & 0xFF;
        fourcc[3] = (fmtdesc.pixelformat >> 24) & 0xFF;

        printf("索引 [%d]: 描述 = '%s', FourCC = [%s]%s\n", 
               fmtdesc.index, 
               fmtdesc.description, 
               fourcc,
               (fmtdesc.flags & V4L2_FMT_FLAG_COMPRESSED) ? " (压缩格式)" : "");
        
        fmtdesc.index++;
    }
    printf("----------------------------\n");
}

/**
 * @brief 协商并设置多平面视频格式 (分辨率和像素格式)
 * @param device_fd 文件描述符
 * @param req_width 期望的宽度
 * @param req_height 期望的高度
 * @param req_pixelformat 期望的格式 (如 V4L2_PIX_FMT_NV12)
 * @return 成功返回驱动最终分配的平面数量，失败返回-1
 */
int v4l2_set_format_mplane(int device_fd, unsigned int req_width, unsigned int req_height, unsigned int req_pixelformat) {
    struct v4l2_format fmt = {0};
    memset(&fmt, 0, sizeof(fmt));

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = req_width;
    fmt.fmt.pix_mp.height = req_height;
    fmt.fmt.pix_mp.pixelformat = req_pixelformat;
    fmt.fmt.pix_mp.field = V4L2_FIELD_ANY;

    if(-1 == xioctl(device_fd, VIDIOC_S_FMT, &fmt)) {
        perror("VIDIOC_S_FMT (MPLANE) 失败");
        goto _ERROR;
    }
    
    printf("--- MPLANE 格式协商结果 ---\n");
    printf("请求的: %u x %u\n", req_width, req_height);
    printf("驱动最终给的: %u x %u\n", fmt.fmt.pix_mp.width, fmt.fmt.pix_mp.height);

    unsigned char num_planes = fmt.fmt.pix_mp.num_planes;
    printf("返回的平面数量 (num_planes): %u\n", num_planes);

    if (fmt.fmt.pix_mp.width != req_width || fmt.fmt.pix_mp.height != req_height) {
        fprintf(stderr, "警告: 摄像头不支持你请求的分辨率，驱动已自动修改为最接近的可用分辨率！\n");
    }

    if (fmt.fmt.pix_mp.pixelformat != req_pixelformat) {
        char req_fcc[5], real_fcc[5];
        snprintf(req_fcc, 5, "%c%c%c%c", req_pixelformat&0xFF, (req_pixelformat>>8)&0xFF, (req_pixelformat>>16)&0xFF, (req_pixelformat>>24)&0xFF);
        snprintf(real_fcc, 5, "%c%c%c%c", fmt.fmt.pix_mp.pixelformat&0xFF, (fmt.fmt.pix_mp.pixelformat>>8)&0xFF, (fmt.fmt.pix_mp.pixelformat>>16)&0xFF, (fmt.fmt.pix_mp.pixelformat>>24)&0xFF);
        
        fprintf(stderr, "致命错误: 请求像素格式 %s，但驱动强制使用了 %s。程序无法继续。\n", req_fcc, real_fcc);
        goto _ERROR;
    }

    printf("--------------------\n");

    return num_planes;
_ERROR:
    return -1;
}

/**
 * @brief 获取多平面视频格式 (分辨率和像素格式)
 * @param fd 文件描述符
 * @param plane_index 第几个平面
 * @param stride 步长
 * @param sizeimage 总字节数
 * @return 成功返回0，失败返回-1
 */
int v4l2_get_format_by_plane(int device_fd, int plane_index, unsigned int *stride, unsigned int *sizeimage){
    struct v4l2_format fmt = {0};
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (-1 == xioctl(device_fd, VIDIOC_G_FMT, &fmt)) {
        perror("VIDIOC_REQBUFS 失败");
        return -1;
    }

    if(plane_index < fmt.fmt.pix_mp.num_planes) {
        *stride = fmt.fmt.pix_mp.plane_fmt[plane_index].bytesperline;
        *sizeimage = fmt.fmt.pix_mp.plane_fmt[plane_index].sizeimage;
    } else {
        return -1;
    }
    return 0;
}


/**
 * @brief 初始化 DMABUF 和 V4L2 队列
 * @param device_fd         视频设备节点
 * @param out_buffers       输出：我们维护的管理结构体数组
 * @param req_buf_count     请求的 buffer 数量 (如 4, 8)
 * @param buf_size          每个dmabuf的大小
 * @return 成功返回0，失败返回-1
 */
int v4l2_init_dmabuf(int device_fd, dmabuf *out_buffers, int req_buf_count, size_t buf_size) {
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));

    req.count = req_buf_count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_DMABUF;

    if (-1 == xioctl(device_fd, VIDIOC_REQBUFS, &req)) {
        perror("VIDIOC_REQBUFS 失败");
        goto _ERROR;
    }

    struct v4l2_format fmt = {0};
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (-1 == xioctl(device_fd, VIDIOC_G_FMT, &fmt)) {
        perror("VIDIOC_REQBUFS 失败");
        goto _ERROR;
    }

    for(int i = 0; i < req_buf_count; i++) {
        out_buffers[i].index = i;
        out_buffers[i].num_planes = fmt.fmt.pix_mp.num_planes;
        
        for (int p = 0; p < out_buffers[i].num_planes; p++) {
            int dma_fd = dmabuf_alloc(buf_size);
            if(dma_fd < 0) {
                perror("dmabuff分配失败！");
                goto _ERROR;
            }

            out_buffers[i].planes[p].fd = dma_fd;
            out_buffers[i].planes[p].length = buf_size;
            out_buffers[i].planes[p].start = NULL;
        }
    }
    
    return 0;

_ERROR:
    return -1;
}

/**
 * @brief 将指定的 buffer 入队 (归还给 V4L2)
 * @param device_fd     视频节点
 * @param dmabuf_info   要归还的 buffer 结构体指针
 * @return 成功返回0，  失败返回-1
 */
int v4l2_queue_frame_dmabuf(int device_fd, dmabuf *dmabuf_info) {
    struct v4l2_buffer buf = {0};
    struct v4l2_plane planes[MAX_V4L2_PLANES];
    memset(planes, 0, sizeof(planes));

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_DMABUF;
    buf.index = dmabuf_info->index;
    buf.length = dmabuf_info->num_planes;
    buf.m.planes = planes;

    for(size_t i = 0; i < buf.length; i++) {
        buf.m.planes[i].length = dmabuf_info->planes[i].length;
        buf.m.planes[i].m.fd = dmabuf_info->planes[i].fd;
    }

    if(-1 == xioctl(device_fd, VIDIOC_QBUF, &buf)) {
        perror("VIDIOC_QBUF 失败");
        return -1;
    }

    return 0;
}

/**
 * @brief 将指定的 buffer 入队 (归还给 V4L2)
 * @param device_fd     视频节点
 * @param dmabuf_info   dmabuffer 结构体指针
 * @return 成功返回dmabuf，失败返回NULL，失败也有可能是errno == EAGAIN， 非阻塞模式下暂时没数据，需要重试。
 */
dmabuf* v4l2_dequeue_frame_dmabuf(int device_fd, dmabuf *dmabuf_info) {
    struct v4l2_buffer buf = {0};
    struct v4l2_plane planes[MAX_V4L2_PLANES];
    memset(planes, 0, sizeof(planes));

    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_DMABUF;
    buf.m.planes = planes;
    // 这里填planes 数组最多能装几个 plane
    buf.length   = MAX_V4L2_PLANES; 

    if(-1 == xioctl(device_fd, VIDIOC_DQBUF, &buf)) {
        if (errno == EAGAIN) return NULL;
        perror("VIDIOC_DQBUF 失败"); 
        return NULL;
    }

    return &dmabuf_info[buf.index];
}

/**
 * @brief 开启视频流
 * @param device_fd     设备文件描述符
 * @param count         我们申请的缓冲区总数
 * @param dmabuf_info   dmabuffer 结构体指针
 * @return              成功返回0，失败返回-1
 */
int v4l2_start_capturing(int device_fd, unsigned int count, dmabuf *dmabuf_info) {
    // 在开启流之前，必须把所有缓冲区塞入内核的"空闲队列"
    for (unsigned int i = 0; i < count; ++i) {
        // 入队
        v4l2_queue_frame_dmabuf(device_fd, &dmabuf_info[i]);
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (-1 == xioctl(device_fd, VIDIOC_STREAMON, &type)) {
        perror("VIDIOC_STREAMON 失败");
        return -1;
    }
    printf("视频流已启动 (STREAMON)\n");
    return 0;
}

/**
 * @brief 关闭视频流
 * @param fd 设备文件描述符
 */
void v4l2_stop_capturing(int device_fd) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

    if (-1 == xioctl(device_fd, VIDIOC_STREAMOFF, &type)) {
        perror("VIDIOC_STREAMOFF 失败");
    }
    printf("视频流已停止 (STREAMOFF)\n");
}

// 释放设备的简单封装
void v4l2_close_device(int device_fd) {
    if(-1 == close(device_fd)) {
        perror("关闭设备失败");
    }
}

