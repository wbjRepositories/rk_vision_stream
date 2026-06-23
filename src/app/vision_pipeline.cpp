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
#include <string>
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
#include <gst/webrtc/webrtc.h>
#include <json-glib/json-glib.h>
#include <libsoup/soup.h>

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
    SoupServer *server{nullptr};
    SoupWebsocketConnection *ws_conn{nullptr};
    std::mutex ws_mutex_;
    std::string ws_peer_ip_;

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
    std::atomic<bool> offer_in_progress{false};
    std::atomic<bool> remote_answer_set{false};
    std::atomic<guint64> rtp_packet_count{0};

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
        webrtcbin     = gst_element_factory_make("webrtcbin",       "webrtcbin");
        GstElement *rtp_capsfilter = gst_element_factory_make("capsfilter", "rtp_caps");

        if (!appsrc || !tee || !queue_rga || !appsink || !queue_trans || !mpph264enc || !h264parse || !rtph264pay || !webrtcbin || !rtp_capsfilter) {
            GST_ERROR("有元件创建失败，请检查插件是否安装！\n");
            return false;
        }

        GstCaps *appsrc_caps = gst_caps_new_simple("video/x-raw",
            "width",     G_TYPE_INT,  VIDEO_WIDTH,
            "height",    G_TYPE_INT,  VIDEO_HEIGHT,
            "format",    G_TYPE_STRING, "NV12",
            "framerate", GST_TYPE_FRACTION, VIDEO_FPS, 1,
            NULL);

        if (nullptr == appsrc_caps) {
            GST_ERROR("appsrc_caps gst_caps_new_simple 失败！\n");
            return false;
        }

        g_object_set(appsrc, "caps", appsrc_caps, NULL);
        gst_caps_unref(appsrc_caps);

        g_object_set(appsrc,
            "is-live", TRUE,
            "format",  GST_FORMAT_TIME,
            NULL);

        
        GstCaps *rtp_caps = gst_caps_new_simple("application/x-rtp",
            "media",            G_TYPE_STRING,  "video",
            "encoding-name",    G_TYPE_STRING,  "H264",
            "payload",          G_TYPE_INT,     96,
            "clock-rate",       G_TYPE_INT,     90000,
            NULL);

        if (nullptr == rtp_caps) {
            GST_ERROR("rtp_caps gst_caps_new_simple 失败！\n");
            return false;
        }
        g_object_set(rtp_capsfilter, "caps", rtp_caps, NULL);
        gst_caps_unref(rtp_caps);

        g_object_set(mpph264enc,
            "profile", 66,          // baseline; 默认 high(100) 会让部分浏览器拒绝 WebRTC H264
            "level",   51,          // 5.1，适配 3840x2160@30fps
            "gop",     VIDEO_FPS,
            "bps",     12000000,
            NULL);

        g_object_set(h264parse, "config-interval", 1, NULL);
        g_object_set(rtph264pay, "config-interval", 1, "pt", 96, NULL);
        // g_object_set(udpsink, "host", REMOTE_ADDRESS, "port", REMOTE_PORT, NULL);

        g_object_set(webrtcbin, "bundle-policy", GST_WEBRTC_BUNDLE_POLICY_MAX_BUNDLE, NULL);

        gst_bin_add_many(GST_BIN(pipeline),
                        appsrc, tee, queue_rga, appsink, queue_trans, mpph264enc, 
                        h264parse, rtph264pay, rtp_capsfilter, webrtcbin, NULL);

        // 分支1
        GstPad *tee_src0 = gst_element_request_pad_simple(tee, "src_%u");
        GstPad *pad_q_rga = gst_element_get_static_pad(queue_rga, "sink");

        // 分支2
        GstPad *tee_src1 = gst_element_request_pad_simple(tee, "src_%u");
        GstPad *pad_q_trans = gst_element_get_static_pad(queue_trans, "sink");

        if (!gst_element_link(appsrc, tee)) {
            g_printerr("appsrc, tee元件连接失败！可能是数据格式不兼容。\n");
            return false;
        }
                        
        if (!gst_element_link_many(queue_rga, appsink, NULL)) {
            g_printerr("queue_rga, appsink元件连接失败！可能是数据格式不兼容。\n");
            return false;
        }

        if (!gst_element_link_many(queue_trans, mpph264enc, h264parse,
                                    rtph264pay, rtp_capsfilter, NULL)) {
            g_printerr("queue_trans, mpph264enc, h264parse, rtph264pay, rtp_capsfilter元件连接失败！可能是数据格式不兼容。\n");
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


        GstPad *rtp_srcpad = gst_element_get_static_pad(rtp_capsfilter, "src");
        GstPad *webrtc_sinkpad = gst_element_request_pad_simple(webrtcbin, "sink_%u");

        if (gst_pad_link(rtp_srcpad, webrtc_sinkpad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link RTP branch to webrtcbin\n");
        }
        gst_object_unref(rtp_srcpad);
        gst_object_unref(webrtc_sinkpad);

        g_signal_connect(webrtcbin, "on-negotiation-needed", G_CALLBACK(on_negotiation_needed_cb_static), this);
        g_signal_connect (webrtcbin, "on-ice-candidate", G_CALLBACK (on_ice_candidate_cb_static), this);
        g_signal_connect(webrtcbin, "notify::ice-connection-state", G_CALLBACK(on_webrtc_notify_cb_static), this);
        g_signal_connect(webrtcbin, "notify::connection-state", G_CALLBACK(on_webrtc_notify_cb_static), this);
        g_signal_connect(webrtcbin, "notify::ice-gathering-state", G_CALLBACK(on_webrtc_notify_cb_static), this);
        g_signal_connect(webrtcbin, "notify::signaling-state", G_CALLBACK(on_webrtc_notify_cb_static), this);

        server = soup_server_new (NULL, NULL);

        soup_server_add_handler(server, NULL, on_http_request_cb_static, this, NULL);
        soup_server_add_websocket_handler (server, "/ws", NULL, NULL,
                                     on_web_connected_cb_static, this, NULL);

        GError *listen_error = NULL;
        if (!soup_server_listen_all(server, 8080, static_cast<SoupServerListenOptions>(0), &listen_error)) {
            g_printerr("libsoup 监听 8080 端口失败: %s\n",
                       listen_error ? listen_error->message : "unknown error");
            g_clear_error(&listen_error);
            return false;
        }
        g_print("HTTP/WebSocket 信令服务已启动: http://0.0.0.0:8080/  ws://0.0.0.0:8080/ws\n");
        
        GstPad *mpph264enc_sink_pad = gst_element_get_static_pad(mpph264enc, "sink");

        gst_pad_add_probe(
            mpph264enc_sink_pad, 
            GST_PAD_PROBE_TYPE_BUFFER, 
            (GstPadProbeCallback)osd_probe_callback, 
            NULL, // 传给回调的用户数据
            NULL  // 销毁回调时的清理函数
        );
        gst_object_unref(mpph264enc_sink_pad);

        GstPad *rtph264pay_src_pad = gst_element_get_static_pad(rtph264pay, "src");
        gst_pad_add_probe(
            rtph264pay_src_pad,
            GST_PAD_PROBE_TYPE_BUFFER,
            on_rtp_buffer_probe_static,
            this,
            NULL
        );
        gst_object_unref(rtph264pay_src_pad);


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

    bool send_ws_text(const gchar *text) {
        SoupWebsocketConnection *connection = nullptr;

        {
            std::lock_guard<std::mutex> lock(ws_mutex_);
            if (!ws_conn || !SOUP_IS_WEBSOCKET_CONNECTION(ws_conn)) {
                return false;
            }
            connection = SOUP_WEBSOCKET_CONNECTION(g_object_ref(ws_conn));
        }

        bool sent = false;
        if (soup_websocket_connection_get_state(connection) == SOUP_WEBSOCKET_STATE_OPEN) {
            soup_websocket_connection_send_text(connection, text);
            sent = true;
        }

        g_object_unref(connection);
        return sent;
    }

    bool has_open_ws_connection() {
        SoupWebsocketConnection *connection = nullptr;

        {
            std::lock_guard<std::mutex> lock(ws_mutex_);
            if (!ws_conn || !SOUP_IS_WEBSOCKET_CONNECTION(ws_conn)) {
                return false;
            }
            connection = SOUP_WEBSOCKET_CONNECTION(g_object_ref(ws_conn));
        }

        bool is_open = soup_websocket_connection_get_state(connection) == SOUP_WEBSOCKET_STATE_OPEN;
        g_object_unref(connection);
        return is_open;
    }

    static void on_webrtc_notify_cb_static(GObject *object, GParamSpec *pspec, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        self->on_webrtc_notify_cb(object, pspec);
    }

    void on_webrtc_notify_cb(GObject *object, GParamSpec *pspec) {
        GValue value = G_VALUE_INIT;
        g_value_init(&value, pspec->value_type);
        g_object_get_property(object, pspec->name, &value);

        if (G_VALUE_HOLDS_ENUM(&value)) {
            gint enum_value_int = g_value_get_enum(&value);
            GEnumClass *enum_class = G_ENUM_CLASS(g_type_class_ref(pspec->value_type));
            GEnumValue *enum_value = g_enum_get_value(enum_class, enum_value_int);
            g_print("webrtcbin %s: %s\n", pspec->name,
                    enum_value ? enum_value->value_nick : "unknown");
            g_type_class_unref(enum_class);
        }

        g_value_unset(&value);
    }

    static GstPadProbeReturn on_rtp_buffer_probe_static(GstPad *pad, GstPadProbeInfo *info,
                                                        gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        return self->on_rtp_buffer_probe(pad, info);
    }

    GstPadProbeReturn on_rtp_buffer_probe(GstPad *pad, GstPadProbeInfo *info) {
        guint64 count = ++rtp_packet_count;
        if (count == 1 || count % 300 == 0) {
            g_print("RTP H264 包已送到 webrtcbin 前: %" G_GUINT64_FORMAT "\n", count);
        }
        return GST_PAD_PROBE_OK;
    }

    void update_ws_peer_ip(SoupClientContext *client) {
        ws_peer_ip_.clear();

        GSocketAddress *remote_address = soup_client_context_get_remote_address(client);
        if (!remote_address || !G_IS_INET_SOCKET_ADDRESS(remote_address)) {
            g_print("无法获取 WebSocket 对端 IP，遇到 .local ICE candidate 时可能无法连通。\n");
            return;
        }

        GInetAddress *inet_address = g_inet_socket_address_get_address(G_INET_SOCKET_ADDRESS(remote_address));
        gchar *ip = g_inet_address_to_string(inet_address);
        if (ip) {
            ws_peer_ip_ = ip;
            g_print("WebSocket 对端 IP: %s\n", ws_peer_ip_.c_str());
            g_free(ip);
        }
    }

    std::string rewrite_mdns_candidate_if_needed(const gchar *candidate) {
        if (!candidate) {
            return std::string();
        }
        if (g_strstr_len(candidate, -1, ".local") == NULL) {
            return std::string(candidate);
        }
        if (ws_peer_ip_.empty()) {
            g_print("收到 .local ICE candidate，但没有 WebSocket 对端 IP，无法改写。\n");
            return std::string(candidate);
        }

        gchar **tokens = g_strsplit(candidate, " ", 0);
        if (!tokens || !tokens[0] || !tokens[1] || !tokens[2] || !tokens[3] || !tokens[4] || !tokens[5]) {
            g_strfreev(tokens);
            return std::string(candidate);
        }

        if (g_str_has_suffix(tokens[4], ".local")) {
            g_print("将浏览器 .local ICE 地址改写为 WebSocket 对端 IP: %s -> %s\n",
                    tokens[4], ws_peer_ip_.c_str());
            g_free(tokens[4]);
            tokens[4] = g_strdup(ws_peer_ip_.c_str());
        }

        gchar *rewritten = g_strjoinv(" ", tokens);
        std::string result = rewritten ? rewritten : candidate;
        g_free(rewritten);
        g_strfreev(tokens);
        return result;
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

    static void on_negotiation_needed_cb_static(GstElement* webrtcbin, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        self->on_negotiation_needed_cb(webrtcbin);
    }

    // ⚖️ 当 WebRTC 准备好进行媒体协商时，这个函数会被触发
    void on_negotiation_needed_cb (GstElement * webrtcbin)
    {
        if (!has_open_ws_connection()) {
            g_print("WebSocket 未连接，暂不创建 Offer，等待浏览器连接。\n");
            return;
        }

        if (remote_answer_set) {
            g_print("已收到 Answer，忽略后续 on-negotiation-needed，避免重复协商。\n");
            return;
        }

        bool already_in_progress = offer_in_progress.exchange(true);
        if (already_in_progress) {
            g_print("已有 Offer 正在等待 Answer，忽略重复 on-negotiation-needed。\n");
            return;
        }

        g_print ("正在向 webrtcbin 发送 create-offer 指令...\n");
        // 创建一个 Promise，并指定当 Offer 生成完毕后，去执行 on_offer_created_cb 函数
        // 同时继续把 app 传递下去
        GstPromise *promise = gst_promise_new_with_change_func (on_offer_created_cb_static, this, NULL);

        // 发射指令：创建 Offer！
        g_signal_emit_by_name (webrtcbin, "create-offer", NULL, promise);
    }

    static void on_ice_candidate_cb_static(GstElement * webrtcbin, guint mlineindex, gchar * candidate, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        self->on_ice_candidate_cb(webrtcbin, mlineindex, candidate);
    }

    // 🧊 当底层引擎找到一条本地网络路径时，这个函数会被触发
    void on_ice_candidate_cb (GstElement * webrtcbin, guint mlineindex, gchar * candidate)
    {
        JsonBuilder *builder = json_builder_new ();
        json_builder_begin_object (builder);

        json_builder_set_member_name (builder, "type");
        json_builder_add_string_value (builder, "ice");

        json_builder_set_member_name (builder, "sdpMLineIndex");
        json_builder_add_int_value (builder, mlineindex);

        json_builder_set_member_name (builder, "candidate");
        json_builder_add_string_value (builder, candidate);

        json_builder_end_object (builder);

        JsonNode *root = json_builder_get_root (builder);
        JsonGenerator *gen = json_generator_new ();
        json_generator_set_root (gen, root);

        gchar *json_str = json_generator_to_data (gen, NULL);
        
        g_print ("--> 发送板子本地 ICE 给网页: %s\n", candidate);

        // 通过 WebSocket 推给网页！
        if (!send_ws_text(json_str)) {
            g_print ("WebSocket 未连接或已关闭，跳过发送本地 ICE。\n");
        }

        g_free (json_str);
        g_object_unref (gen);
        json_node_free (root);
        g_object_unref (builder);
    }

    static void on_offer_created_cb_static(GstPromise * promise, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        self->on_offer_created_cb(promise);
    }

    void on_offer_created_cb (GstPromise * promise) {
        GstWebRTCSessionDescription *offer = NULL;
        
        // 1. 获取 Promise 的返回值（里面装载了刚刚生成的 Offer）
        const GstStructure *reply = gst_promise_get_reply (promise);
        
        // 2. 从返回值中提取出名为 "offer" 的对象
        gst_structure_get (reply, "offer",
                            GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);

        // 此时，你可以把 offer 里的纯文本打印出来看看长啥样
        gchar *sdp_text = gst_sdp_message_as_text (offer->sdp);
        g_print ("生成的 Offer SDP:\n%s\n", sdp_text);

        // 3. 关键动作：签发“本地描述”
        g_print ("正在将生成的 Offer 设置为 Local Description...\n");
        GstPromise *local_desc_promise = gst_promise_new (); // 设置本地描述也需要一个空的 promise
        g_signal_emit_by_name (webrtcbin, "set-local-description", offer, local_desc_promise);
        
        // 释放资源
        gst_promise_interrupt (local_desc_promise);
        gst_promise_unref (local_desc_promise);
        gst_webrtc_session_description_free (offer);
        gst_promise_unref (promise);
        
        // ==========================================
        // 💡 下一步：在这里，我们要用 json-glib 把 sdp_text 打包，
        // 然后用 libsoup 发送给信令服务器！
        // ==========================================
        JsonBuilder *builder = json_builder_new ();

        // 2. 开始构建一个 JSON 对象 { ... }
        json_builder_begin_object (builder);

        // 添加 "type" 字段
        json_builder_set_member_name (builder, "type");
        json_builder_add_string_value (builder, "offer");

        // 添加 "sdp" 字段，并把刚才获取的 sdp_text 填进去
        json_builder_set_member_name (builder, "sdp");
        json_builder_add_string_value (builder, sdp_text);

        // 结束构建该对象
        json_builder_end_object (builder);

        // 3. 将构建好的积木转化为节点 (Node)
        JsonNode *root = json_builder_get_root (builder);

        // 4. 使用生成器 (Generator) 把节点序列化为纯文本字符串
        JsonGenerator *gen = json_generator_new ();
        json_generator_set_root (gen, root);
        
        // 获取最终要通过网络发送的 JSON 字符串
        gchar *json_string = json_generator_to_data (gen, NULL);

        g_print ("准备发送给远端的 JSON:\n%s\n", json_string);

        // ==========================================
        // 下一步：调用 libsoup-2.4 的发送函数，把 json_string 发出去！
        if (!send_ws_text(json_string)) {
            g_print ("WebSocket 未连接或已关闭，跳过发送 Offer。\n");
            offer_in_progress = false;
        }
        // ==========================================

        // 5. 释放使用过的 json 对象和字符串内存，防止内存泄漏
        g_free (json_string);
        g_object_unref (gen);
        json_node_free (root);
        g_object_unref (builder);

        g_free (sdp_text);
    }


    static void on_web_connected_cb_static(SoupServer *server, SoupWebsocketConnection *connection,
                                            const char *path, SoupClientContext *client,
                                            gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        self->on_web_connected_cb(server, connection, path, client);
    }


    void on_web_connected_cb (SoupServer *server, SoupWebsocketConnection *connection,
                                const char *path, SoupClientContext *client ) {

        g_print ("🎉 叮咚！检测到网页客户端连入 WebSocket!\n");
        update_ws_peer_ip(client);

        SoupWebsocketConnection *old_conn = nullptr;
        {
            std::lock_guard<std::mutex> lock(ws_mutex_);
            if (ws_conn != NULL && ws_conn != connection) {
                old_conn = ws_conn;
            }
            ws_conn = SOUP_WEBSOCKET_CONNECTION(g_object_ref(connection));
        }
        offer_in_progress = false;
        remote_answer_set = false;

        if (old_conn != nullptr) {
            g_print ("正在断开旧的网页连接...\n");
            soup_websocket_connection_close(old_conn, SOUP_WEBSOCKET_CLOSE_GOING_AWAY, "new client connected");
            g_object_unref(old_conn);
        }

        // 2. 绑定“收到网页消息”的信号（用来收网页发回来的 Answer 和 ICE）
        g_signal_connect (connection, "message", G_CALLBACK (on_ws_message_cb_static), this);
        
        // 3. 绑定“网页关掉浏览器”的信号
        g_signal_connect (connection, "closed", G_CALLBACK (on_ws_closed_cb_static), this);

        // 4. 🚀🚀🚀 全场最核心的一步：点燃导火索！
        g_print ("正在唤醒 GStreamer 管道开始推流...\n");
        
        gst_element_set_state (pipeline, GST_STATE_PLAYING);
        on_negotiation_needed_cb(webrtcbin);
    }

    static void on_http_request_cb_static(SoupServer *server, SoupMessage *msg,
                                          const char *path, GHashTable *query,
                                          SoupClientContext *client, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        self->on_http_request_cb(server, msg, path, query, client);
    }

    void on_http_request_cb(SoupServer *server, SoupMessage *msg,
                            const char *path, GHashTable *query,
                            SoupClientContext *client) {
        const char *relative_path = NULL;
        const char *content_type = NULL;

        if (g_strcmp0(msg->method, SOUP_METHOD_GET) != 0) {
            soup_message_set_status(msg, SOUP_STATUS_METHOD_NOT_ALLOWED);
            return;
        }

        if (g_strcmp0(path, "/") == 0 || g_strcmp0(path, "/index.html") == 0) {
            relative_path = "index.html";
            content_type = "text/html; charset=utf-8";
        } else if (g_strcmp0(path, "/client.js") == 0) {
            relative_path = "client.js";
            content_type = "application/javascript; charset=utf-8";
        } else {
            soup_message_set_status(msg, SOUP_STATUS_NOT_FOUND);
            soup_message_set_response(msg, "text/plain; charset=utf-8",
                                      SOUP_MEMORY_STATIC, "404 Not Found\n", 14);
            return;
        }

        gchar *file_path = g_build_filename(BROWSER_CLIENT_DIR, relative_path, NULL);
        gchar *contents = NULL;
        gsize length = 0;
        GError *error = NULL;

        if (!g_file_get_contents(file_path, &contents, &length, &error)) {
            g_printerr("读取浏览器页面文件失败 %s: %s\n",
                       file_path, error ? error->message : "unknown error");
            g_clear_error(&error);
            g_free(file_path);
            soup_message_set_status(msg, SOUP_STATUS_NOT_FOUND);
            soup_message_set_response(msg, "text/plain; charset=utf-8",
                                      SOUP_MEMORY_STATIC, "404 Not Found\n", 14);
            return;
        }

        g_free(file_path);
        soup_message_set_status(msg, SOUP_STATUS_OK);
        soup_message_headers_append(msg->response_headers, "Cache-Control", "no-cache");
        soup_message_set_response(msg, content_type, SOUP_MEMORY_TAKE, contents, length);
    }


    static void on_ws_message_cb_static(SoupWebsocketConnection *connection, gint type, GBytes *message, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        self->on_ws_message_cb(connection, type, message);
    }

    void on_ws_message_cb (SoupWebsocketConnection *connection, gint type, GBytes *message) {
        gsize size;
        const gchar *data;
        GError *error = NULL;

        // 1. WebSocket 规范：过滤掉非文本的二进制杂音
        if (type != SOUP_WEBSOCKET_DATA_TEXT) {
            return;
        }

        // 从 GBytes 中把原始字符串连同长度一起拔出来
        data = static_cast<const gchar *>(g_bytes_get_data(message, &size));
        g_print ("\n📥 收到网页端 JSON (长度 %" G_GSIZE_FORMAT " 字节):\n%s\n", size, data);

        // 2. 召唤 json-glib 解析器
        JsonParser *parser = json_parser_new ();
        if (!json_parser_load_from_data (parser, data, size, &error)) {
            g_printerr ("糟糕，网页发的 JSON 格式烂了: %s\n", error->message);
            g_clear_error (&error);
            g_object_unref (parser);
            return;
        }

        // 获取 JSON 的根 Object
        JsonObject *root_obj = json_node_get_object (json_parser_get_root (parser));
        if (!json_object_has_member (root_obj, "type")) {
            g_printerr ("收到不明包裹（缺少 'type' 字段）\n");
            g_object_unref (parser);
            return;
        }

        const gchar *msg_type = json_object_get_string_member (root_obj, "type");

        // ==================== 分支 A：处理远端 Answer ====================
        if (g_strcmp0 (msg_type, "answer") == 0) {
            const gchar *sdp_str = json_object_get_string_member (root_obj, "sdp");
            GstSDPMessage *sdp = NULL;
            GstWebRTCSessionDescription *answer = NULL;

            if (g_strstr_len(sdp_str, -1, "m=video 0") != NULL) {
                g_printerr("浏览器拒绝了视频 m-line：answer 里出现 m=video 0。"
                           "通常是 H264 profile/level 不被浏览器接受；当前 Offer 里的 profile-level-id 需要调成 baseline/constrained-baseline，或降低分辨率/level。\n");
                offer_in_progress = false;
                remote_answer_set = true;
                g_object_unref (parser);
                return;
            }
            if (g_strstr_len(sdp_str, -1, ".local") != NULL) {
                g_printerr("Answer 里的 ICE candidate 使用了 .local mDNS 地址。"
                           "如果 webrtcbin 的 ice-connection-state 一直停在 checking/failed，"
                           "请在浏览器关闭 WebRTC mDNS 隐藏本地 IP，或使用可用的 STUN/TURN。\n");
            }

            g_print ("--> 正在解析远端 Answer SDP...\n");
            if (gst_sdp_message_new_from_text (sdp_str, &sdp) == GST_SDP_OK) {
            
            // 把原始 SDP 文本包装成 WebRTC 专用的 Answer 结构体
            answer = gst_webrtc_session_description_new (GST_WEBRTC_SDP_TYPE_ANSWER, sdp);
            
            // 动作指令：设置远端描述！(同样需要一个占位的 promise)
            GstPromise *promise = gst_promise_new ();
            g_signal_emit_by_name (webrtcbin, "set-remote-description", answer, promise);

            gst_promise_interrupt (promise);
            gst_promise_unref (promise);
            gst_webrtc_session_description_free (answer);
            offer_in_progress = false;
            remote_answer_set = true;
            
            g_print ("✅ 远端 Answer 设置成功！握手进度 50%%\n");
            } else {
            g_printerr ("无法将 Answer 文本转为 GstSDPMessage！\n");
            offer_in_progress = false;
            }
        } 
        // ==================== 分支 B：处理远端 ICE ====================
        else if (g_strcmp0 (msg_type, "ice") == 0) {
            gint mline_index = json_object_get_int_member (root_obj, "sdpMLineIndex");
            const gchar *candidate_str = json_object_get_string_member (root_obj, "candidate");
            std::string candidate = rewrite_mdns_candidate_if_needed(candidate_str);

            g_print ("--> 收到远端网络路径 (ICE): index=%d, %s\n", mline_index, candidate.c_str());

            // 动作指令：喂给 GStreamer 底层引擎去打洞！
            g_signal_emit_by_name (webrtcbin, "add-ice-candidate", mline_index, candidate.c_str());
        } 
        else {
            g_print ("未知的包裹类型: %s\n", msg_type);
        }

        // 释放解析器，绝不漏一滴内存
        g_object_unref (parser);
    }

    static void on_ws_closed_cb_static(SoupWebsocketConnection *connection, gpointer user_data) {
        auto* self = static_cast<Impl*>(user_data);
        self->on_ws_closed_cb(connection);
    }

    void on_ws_closed_cb (SoupWebsocketConnection *connection) {
        g_print ("🛑 网页端关闭了连接，暂停底层流媒体推流。\n");

        bool is_current_connection = false;
        {
            std::lock_guard<std::mutex> lock(ws_mutex_);
            if (connection == ws_conn) {
                g_clear_object(&ws_conn);
                is_current_connection = true;
            }
        }
        
        if (is_current_connection) {
            // 网页跑了，把管道设回 NULL 节约板子 CPU，等下一个人进来再重新启动
            gst_element_set_state (pipeline, GST_STATE_NULL);
            offer_in_progress = false;
            remote_answer_set = false;
            ws_peer_ip_.clear();
        }
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
