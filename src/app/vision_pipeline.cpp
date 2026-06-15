#include "vision_pipeline.hpp"

// 在 cpp 文件中尽情引入底层库
#include <gst/gst.h>
// #include <gst/app/gstappsink.h>
#include <gst/allocators/gstdmabuf.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <queue>
#include <vector>
#include "v4l2_camera.h"
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <gst/app/gstappsink.h>
#include <poll.h>
#include "gst_rga.h"
#include "rknn_model.hpp"
#include <rga/im2d.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/dma-buf.h>
#include "config.h"
#include "draw_lines_by_cpu.hpp"


extern std::queue<BoundingBox> g_bb_q;
extern std::mutex g_queue_mutex;


struct dma_buff_fd_context {
    GstSample* sample;
    int fd;
};

using dma_ctx = dma_buff_fd_context;

struct GstBufferReleaseContext {
    int device_fd;
    dmabuf_buffer *v4l2_buf;
};


// 定义隐藏的实现类
class VisionPipeline::Impl {
private:
    int     device_fd{-1};
    int     planes_num{-1};
    __u32   stride{0};
    __u32   sizeimage{0};
    dmabuf  out_buffers[4]{0};
    int     buffers_num{4};
    guint64 frame_count{0};

    // GStreamer 资源
    GstElement* pipeline{nullptr};
    GstElement* appsrc{nullptr};
    GstElement* appsink{nullptr};
    GstElement* webrtcbin{nullptr};
    GMainLoop*  main_loop{nullptr};
    GstAllocator* allocator{nullptr};

    // appsink 到 inference 线程的图像队列和锁
    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    // 假设用一个简单的结构体保存图像地址，实际应用中这里传的是 DMA-BUF fd 或 mmap 的内存指针
    std::queue<dma_ctx> dma_fd_queue; 

    // 线程管理
    std::thread appsrc_thread;
    std::thread loop_thread;         // 专门跑 GMainLoop 处理异步事件
    std::thread inference_thread;    // 专门跑 YOLO 推理
    std::atomic<bool> is_running{false};

public:
    Impl() = default;

    ~Impl() {
        stop();
    }

    bool init() {
        if (pipeline || device_fd >= 0) {
            return true;
        }

        device_fd = v4l2_open_device(DEVICE_PATH);
        if (device_fd < 0) {
            GST_ERROR("V4L2 打开设备错误！");
            return false;
        }
        planes_num = v4l2_set_format_mplane(device_fd, VIDEO_WIDTH, VIDEO_HEIGHT, V4L2_PIX_FMT_NV12);
        if(planes_num < 0) {
            GST_ERROR("V4L2 设置格式错误！");
            return false;
        }

        if(v4l2_get_format_by_plane(device_fd, 0, &stride, &sizeimage) < 0) {
            GST_ERROR("V4L2 获取格式错误！");
            return false;
        }

        if(v4l2_init_dmabuf(device_fd, out_buffers, buffers_num, sizeimage) < 0) {
            GST_ERROR("初始化 DMABUF 和 V4L2 队列失败！");
            return false;
        }


        gst_init(NULL, NULL);

        pipeline = gst_pipeline_new("vision-pipeline");
        allocator = gst_dmabuf_allocator_new();

        if (!pipeline) {
            GST_ERROR("创建 GStreamer pipeline 失败！");
            return false;
        }
        if (!allocator) {
            GST_ERROR("创建 DMABUF allocator 失败！");
            return false;
        }

        appsrc                  = gst_element_factory_make("appsrc",        "v4l2out");
        GstElement *tee         = gst_element_factory_make("tee",           "tee");
        
        // 分支1：rga缩放给算法
        GstElement *queue_rga   = gst_element_factory_make("queue",         "q_rga");
        appsink                 = gst_element_factory_make("appsink",       "appsink");

        // 分支2：远程传输
        GstElement *queue_trans = gst_element_factory_make("queue",         "q_trans");
        GstElement *mpph264enc  = gst_element_factory_make("mpph264enc",    "mpph264enc");
        GstElement *h264parse   = gst_element_factory_make("h264parse",    "h264parse");
        GstElement *rtph264pay  = gst_element_factory_make("rtph264pay",    "rtph264pay");
        GstElement *udpsink     = gst_element_factory_make("udpsink",       "udpsink");

        if (!appsrc || !tee || !queue_rga || !appsink || !queue_trans || !mpph264enc || !h264parse || !rtph264pay || !udpsink) {
            GST_ERROR("有元件创建失败，请检查插件是否安装！\n");
            return false;
        }

        GstCaps *caps = gst_caps_new_simple("video/x-raw",
            "width",     G_TYPE_INT,  VIDEO_WIDTH,
            "height",    G_TYPE_INT,  VIDEO_HEIGHT,
            "format",    G_TYPE_STRING, "NV12",
            "framerate", GST_TYPE_FRACTION, VIDEO_FPS, 1,
            NULL);

        if (nullptr == caps) {
            GST_ERROR("gst_caps_new_simple 失败！\n");
            return false;
        }

        g_object_set(appsrc, "caps", caps, NULL);
        gst_caps_unref(caps);

        g_object_set(appsrc,
            "is-live", TRUE,
            "format",  GST_FORMAT_TIME,
            NULL);

        g_object_set(rtph264pay, "config-interval", 1, NULL);
        g_object_set(udpsink, "host", REMOTE_ADDRESS, "port", REMOTE_PORT, NULL);

        gst_bin_add_many(GST_BIN(pipeline),
                        appsrc, tee, queue_rga, appsink, queue_trans, mpph264enc, h264parse, rtph264pay, udpsink, NULL);

        // 分支1
        GstPad *tee_src0 = gst_element_request_pad_simple(tee, "src_%u");
        GstPad *pad_q_rga = gst_element_get_static_pad(queue_rga, "sink");

        // 分支2
        GstPad *tee_src1 = gst_element_request_pad_simple(tee, "src_%u");
        GstPad *pad_q_trans = gst_element_get_static_pad(queue_trans, "sink");

        if (!gst_element_link(appsrc, tee)) {
            g_printerr("元件连接失败！可能是数据格式不兼容。\n");
            return false;
        }
                        
        if (!gst_element_link_many(queue_rga, appsink, NULL)) {
            g_printerr("元件连接失败！可能是数据格式不兼容。\n");
            return false;
        }

        if (!gst_element_link_many(queue_trans, mpph264enc, h264parse,
                                    rtph264pay, udpsink, NULL)) {
            g_printerr("元件连接失败！可能是数据格式不兼容。\n");
            return false;
        }

        if(!tee_src0 || !pad_q_rga || !tee_src1 || !pad_q_trans) {
            GST_ERROR("获取Pad时出现错误！\n");
        }

        if (GST_PAD_LINK_OK != gst_pad_link(tee_src0, pad_q_rga)) {
            GST_ERROR("tee1 连接失败！\n");
        }

        if (GST_PAD_LINK_OK != gst_pad_link(tee_src1, pad_q_trans)) {
            GST_ERROR("tee2 连接失败！\n");
        }

        
        GstPad *mpph264enc_sink_pad = gst_element_get_static_pad(mpph264enc, "sink");

        gst_pad_add_probe(
            mpph264enc_sink_pad, 
            GST_PAD_PROBE_TYPE_BUFFER, 
            (GstPadProbeCallback)osd_probe_callback, 
            NULL, // 传给回调的用户数据
            NULL  // 销毁回调时的清理函数
        );


        gst_object_unref(pad_q_rga);
        gst_object_unref(pad_q_trans);
        gst_object_unref(tee_src0);
        gst_object_unref(tee_src1);

        g_object_set(appsink, "emit-signals", TRUE, NULL);
        g_object_set(appsink, "max-buffers", 1, "drop", TRUE, NULL);
        g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample_static), this);

        GstBus *bus = gst_element_get_bus(pipeline);
        if (!bus) {
            GST_ERROR("从pipeline中获取bus失败！");
            return false;
        }
        gst_bus_add_watch(bus, bus_callback_wrapper, this);
        gst_object_unref(bus);

        return true;
    }

    bool start() {
        if (is_running) return true;
        is_running = true;

        /* ---- 启动 V4L2 流 ---- */
        v4l2_start_capturing(device_fd, buffers_num, out_buffers);

        /* ---- 启动 pipeline ---- */
        GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            g_printerr("无法将管道设置为 PLAYING 状态！\n");
            gst_object_unref(pipeline);
            // return -1;
        }
        g_print("管道已进入 PLAYING 状态，开始推流...\n");

        appsrc_thread = std::thread(&Impl::appsrc_push_task,this);


        // 启动 GMainLoop 线程 (必须有这个，否则 WebRTC 信令和总线消息不工作)
        main_loop = g_main_loop_new(nullptr, FALSE); // 监听总线
        loop_thread = std::thread([this]() {
            g_main_loop_run(main_loop);
        });

        // 3. 启动旁路推理线程
        inference_thread = std::thread(&Impl::inference_task, this);
        return true;
    }

    void stop() {
        if (!is_running) return;
        is_running = false;

        // 通知推理线程退出
        frame_cv_.notify_all();
        if (inference_thread.joinable()) {
            inference_thread.join();
        }

        // 停止 GStreamer 和 GMainLoop
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
        }
        if (main_loop) {
            g_main_loop_quit(main_loop);
            if (loop_thread.joinable()) {
                loop_thread.join();
            }
            g_main_loop_unref(main_loop);
            main_loop = nullptr;
        }
        // ... 清理其他对象
    }

private:
    // 静态中转函数（符合 C 语言签名）
    static gboolean bus_callback_wrapper(GstBus *bus, GstMessage *msg, gpointer user_data) {
        // 3. 将 user_data 强转回 Impl 对象的指针 (this)
        Impl* self = static_cast<Impl*>(user_data);
        // 4. 调用真正的普通成员函数
        return self->bus_call_back(bus, msg); 
    }

    /* ============================================================================
    * bus_call_back — 总线消息回调
    * ============================================================================ */
    gboolean bus_call_back(GstBus *bus, GstMessage *msg) {

        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError *err = NULL;
                gchar *debug_info = NULL;
                gst_message_parse_error(msg, &err, &debug_info);
                g_printerr("致命错误: 来自元件 %s: %s\n",
                        GST_OBJECT_NAME(msg->src), err->message);
                g_printerr("调试信息: %s\n", debug_info ? debug_info : "无");
                g_clear_error(&err);
                g_free(debug_info);
                g_main_loop_quit(main_loop);
                break;
            }
            case GST_MESSAGE_EOS:
                g_print("End-Of-Stream (EOS): 视频播放结束。\n");
                g_main_loop_quit(main_loop);
                break;
            case GST_MESSAGE_STATE_CHANGED: {
                if (GST_MESSAGE_SRC(msg) == GST_OBJECT(pipeline)) {
                    GstState old_state, new_state, pending_state;
                    gst_message_parse_state_changed(msg, &old_state, &new_state,
                                                    &pending_state);
                    g_print("管道状态从 %s 变为 %s\n",
                            gst_element_state_get_name(old_state),
                            gst_element_state_get_name(new_state));
                }
                break;
            }
            default:
                break;
        }
        return TRUE;
    }


    // 这是 appsink 的 C 风格回调，我们通过 user_data 把 C++ 的 this 指针传进来
    static GstFlowReturn on_new_sample_static(GstElement* sink, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        return self->on_new_sample(sink);
    }

    GstFlowReturn on_new_sample(GstElement* appsink) {
        GstBuffer *buffer;
        GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
        if (!sample) {
            g_printerr("获取sample错误！\n");
        }

        buffer = gst_sample_get_buffer(sample);
        if (!buffer) {
            g_printerr("获取buffer错误！\n");
        }

        // 2. 获取 Buffer 中的第一个内存块 (通常 DMABuf 只有一个 memory block)
        GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
        
        // 3. 检查该内存块是否真的是 DMABUF 类型
        if (mem && gst_is_dmabuf_memory(mem)) {
            
            // 4. 提取文件描述符 FD
            // gint fd = gst_dmabuf_memory_get_fd(mem);
            gst_sample_ref(sample);
            dma_ctx dctx;
            dctx.fd = gst_dmabuf_memory_get_fd(mem);
            dctx.sample = sample;
            // g_print("成功提取 DMABUF FD: %d\n", fd);
            
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                // 如果队列积压太多（推理太慢），丢弃旧帧保证实时性
                if (dma_fd_queue.size() > 2) {
                    dma_fd_queue.pop(); 
                }
                dma_fd_queue.push(dctx);
            }
            frame_cv_.notify_one(); // 唤醒推理线程

        } else {
            g_print("警告: 当前 Buffer 不是 DMABUF 内存\n");
        }

        gst_sample_unref(sample);

        return GST_FLOW_OK;
    }


    // 独立的推理线程
    void inference_task() {
        // 在这里初始化 RKNN 模型
        std::shared_ptr<rknn_model> model = std::make_shared<rknn_model>();
        model->init();

        while (is_running) {
            dma_ctx dctx = {0};
            {
                std::unique_lock<std::mutex> lock(frame_mutex_);
                frame_cv_.wait(lock, [this]{ return !dma_fd_queue.empty() || !is_running; });
                if (!is_running) break;

                dctx = dma_fd_queue.front();
                dma_fd_queue.pop();
            }

            GstCaps *caps = gst_sample_get_caps(dctx.sample);
            if (!caps) {
                g_printerr("获取caps失败！\n");
                return;
            }
            GstVideoInfo video_info;
            if (!gst_video_info_from_caps(&video_info, caps)) {
                g_printerr("Failed to parse caps into video info\n");
                return;
            }
            
            int dst_fd = model->get_dma_fd();

            if (rga_scale_yolo(dctx.fd, dst_fd, video_info.width, video_info.height) != 0) {
                g_printerr("RGA preprocess failed, skip this frame\n");
                gst_sample_unref(dctx.sample);
                continue;
            }


            // 1. 调用 RKNN 执行 YOLO 推理
            // std::vector<BoundingBox> results = run_rknn(frame_data);
            model->run();
            gst_sample_unref(dctx.sample);
            
            // 测试用的伪数据
            std::vector<BoundingBox> results = {{10, 10, 50, 50, 0, 0.95}};

            // 2. 如果外部注册了回调，把结果甩出去 (甩给 ROS 2)
            // if (inference_cb_) {
            //     inference_cb_(results);
            // }
        }
    }




    void appsrc_push_task(void) {
        struct pollfd poll_fd[1];
        poll_fd[0].fd = device_fd;
        poll_fd[0].events = POLLIN;
        struct dmabuf_buffer *buf = nullptr;
        while (is_running)
        {
            int ret = poll(poll_fd, 1, 500);
            if (ret < 0) {
                perror("poll error");
                break;
            }

            if (poll_fd[0].revents & POLLIN) {
                /* 非阻塞出队 */
                buf = v4l2_dequeue_frame_dmabuf(device_fd, out_buffers);

                if (buf) {
                    GstFlowReturn ret = push_one_frame(buf);

                    if (ret == GST_FLOW_FLUSHING || ret == GST_FLOW_ERROR) {
                        g_print("管道已停止，结束推流。\n");
                        g_main_loop_quit(main_loop);
                        is_running = false;
                    }
                }
                /* 如果 buf == NULL（EAGAIN），什么都不做，等下一轮超时 */
            }
        }
    }


    /* ============================================================================
    * push_one_frame — 把一帧 DMA-BUF 数据封装为 GstBuffer，推入 appsrc
    *
    * 这是推模式的核心：V4L2 只给出一个 dma-buf fd，
    * 我们必须通过 GstDmaBufAllocator 把它包装成 GStreamer 认识的 GstMemory，
    * 再加上 GstVideoMeta（告诉编码器 stride / offset / 格式），
    * 最后打上 PTS，调用 gst_app_src_push_buffer() 推入管道。
    * ============================================================================ */
    GstFlowReturn push_one_frame(struct dmabuf_buffer *v4l2_buf)
    {
        /* ---- 1. 复制 fd（GStreamer 会接管所有权）---- */
        int fd_dup = dup(v4l2_buf->planes[0].fd);
        if (fd_dup < 0) {
            g_printerr("dup DMA-BUF fd 失败！\n");
            v4l2_queue_frame_dmabuf(device_fd, v4l2_buf);
            return GST_FLOW_ERROR;
        }

        /* ---- 2. fd → GstMemory ---- */
        GstMemory *mem = gst_dmabuf_allocator_alloc(allocator,
                                                    fd_dup, sizeimage);
        if (!mem) {
            g_printerr("从 dmabuf fd 创建 GstMemory 失败！\n");
            close(fd_dup);
            v4l2_queue_frame_dmabuf(device_fd, v4l2_buf);
            return GST_FLOW_ERROR;
        }

        GstBuffer *buffer = gst_buffer_new();
        gst_buffer_append_memory(buffer, mem);

        auto *release_ctx = new GstBufferReleaseContext{device_fd, v4l2_buf};
        static GQuark release_quark = g_quark_from_static_string("gst-core-v4l2-buffer-release");
        gst_mini_object_set_qdata(
            GST_MINI_OBJECT(buffer),
            release_quark,
            release_ctx,
            release_v4l2_buffer_when_gst_done);

        /* ---- 3. 添加 GstVideoMeta（告知编码器 NV12 的 stride 和 UV 偏移）---- */
        // NV12: Y 平面在前，UV 交错平面在后
        gint   strides[4] = { static_cast<gint>(stride), static_cast<gint>(stride), 0, 0 };
        gsize  offsets[4] = { 0, (gsize)sizeimage * 2 / 3, 0, 0 };

        gst_buffer_add_video_meta_full(
            buffer,
            GST_VIDEO_FRAME_FLAG_NONE,
            GST_VIDEO_FORMAT_NV12,
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
            2,          // NV12 有 2 个平面
            offsets,
            strides);

        /* ---- 4. 打时间戳 ---- */
        GstClockTime pts = frame_count *
                        (GST_SECOND / VIDEO_FPS);
        GST_BUFFER_PTS(buffer)      = pts;
        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND,
                                                                VIDEO_FPS);
        frame_count++;

        /* ---- 5. 推入 appsrc（推模式唯一入口）---- */
        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc),
                                                    buffer);
        if (ret != GST_FLOW_OK) {
            g_printerr("appsrc push buffer 失败: %s (%d)\n",
                    gst_flow_get_name(ret), ret);
        }
        return ret;
    }

    static void release_v4l2_buffer_when_gst_done(gpointer data)
    {
        auto *ctx = static_cast<GstBufferReleaseContext *>(data);
        if (ctx && ctx->v4l2_buf) {
            v4l2_queue_frame_dmabuf(ctx->device_fd, ctx->v4l2_buf);
        }
        delete ctx;
    }


    /* 【核心】：探针回调函数 */
    static GstPadProbeReturn osd_probe_callback(
        GstPad *pad,
        GstPadProbeInfo *info,
        gpointer user_data)
    {
        GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
        if (!buffer) {
            return GST_PAD_PROBE_OK;
        }

        GstMemory *mem = gst_buffer_peek_memory(buffer, 0);

        // g_print("memory count = %u\n", gst_buffer_n_memory(buffer));
        // g_print("is dmabuf = %d\n", gst_is_dmabuf_memory(mem));

        if (!mem || !gst_is_dmabuf_memory(mem)) {
            g_printerr("不是 DMABUF memory\n");
            return GST_PAD_PROBE_OK;
        }

        int fd = gst_dmabuf_memory_get_fd(mem);
        gsize mem_offset = 0;
        gsize max_size = 0;
        gsize mem_size = gst_memory_get_sizes(mem, &mem_offset, &max_size);
        gsize map_size = max_size > 0 ? max_size : mem_size;

        GstCaps *caps = gst_pad_get_current_caps(pad);
        if (!caps) {
            g_printerr("无法获取 caps\n");
            return GST_PAD_PROBE_OK;
        }

        GstStructure *s = gst_caps_get_structure(caps, 0);

        int width = 0;
        int height = 0;
        const char *format = gst_structure_get_string(s, "format");

        if (!gst_structure_get_int(s, "width", &width) ||
            !gst_structure_get_int(s, "height", &height) ||
            !format) {
            g_printerr("caps 中没有 width/height/format\n");
            gst_caps_unref(caps);
            return GST_PAD_PROBE_OK;
        }

        int rga_format = 0;
        bool is_nv12 = false;

        if (g_strcmp0(format, "NV12") == 0) {
            rga_format = RK_FORMAT_YCbCr_420_SP;
            is_nv12 = true;
        } else if (g_strcmp0(format, "RGB") == 0) {
            rga_format = RK_FORMAT_RGB_888;
        } else if (g_strcmp0(format, "BGR") == 0) {
            rga_format = RK_FORMAT_BGR_888;
        } else if (g_strcmp0(format, "RGBA") == 0) {
            rga_format = RK_FORMAT_RGBA_8888;
        } else {
            g_printerr("RGA 暂不支持当前 format: %s\n", format);
            gst_caps_unref(caps);
            return GST_PAD_PROBE_OK;
        }

        gst_caps_unref(caps);

        int wstride = width;
        int hstride = height;
        int uv_stride = width;
        gsize uv_offset = (gsize)width * height;
        GstVideoMeta *vmeta = gst_buffer_get_video_meta(buffer);
        if (vmeta && vmeta->n_planes > 0) {
            wstride = vmeta->stride[0];
            if (is_nv12 && vmeta->n_planes > 1 && vmeta->stride[0] > 0) {
                uv_stride = vmeta->stride[1];
                uv_offset = vmeta->offset[1];
                hstride = vmeta->offset[1] / vmeta->stride[0];
            }
        } else if (is_nv12) {
            wstride = cpu_draw::align_up(width, 16);
            hstride = cpu_draw::align_up(height, 2);
            uv_stride = wstride;
            uv_offset = (gsize)wstride * hstride;
        }

        std::vector<BoundingBox> boxes;
        {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            while (!g_bb_q.empty()) {
                boxes.push_back(g_bb_q.front());
                g_bb_q.pop();
            }
        }

        for (auto &bb : boxes) {
            printf("front：x1=%d,y1=%d\nx2=%d,y2=%d\n",bb.x1,bb.y1,bb.x2,bb.y2);
            cpu_draw::clip_box(bb.x1, bb.y1, bb.x2, bb.y2, width, height);
            if (is_nv12) {
                // 采用cpu画框
                cpu_draw::Nv12Dmabuf nv12_buffer{
                    fd,
                    static_cast<std::size_t>(mem_offset),
                    static_cast<std::size_t>(mem_size),
                    static_cast<std::size_t>(map_size),
                    width,
                    height,
                    wstride,
                    uv_stride,
                    static_cast<std::size_t>(uv_offset)
                };
                cpu_draw::draw_rect_nv12(nv12_buffer, {bb.x1, bb.y1, bb.x2, bb.y2}, 8);
            } else {
                rga_buffer_t img = wrapbuffer_fd(
                    fd,
                    width,
                    height,
                    rga_format,
                    wstride,
                    hstride
                );
                im_rect rect{bb.x1, bb.y1, bb.x2 - bb.x1, bb.y2 - bb.y1};
                imrectangle(img, rect, 0xEB8080, 8);
            }
        }

        return GST_PAD_PROBE_OK;
    }
};




VisionPipeline::VisionPipeline() : pimpl_(std::make_unique<Impl>()) {}
VisionPipeline::~VisionPipeline() = default;

bool VisionPipeline::init()
{
    return pimpl_->init();
}

bool VisionPipeline::start()
{
    return pimpl_->start();
}

void VisionPipeline::stop()
{
    pimpl_->stop();
}
