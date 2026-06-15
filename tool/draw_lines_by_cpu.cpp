#include "draw_lines_by_cpu.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace cpu_draw {
namespace {

void fill_luma(std::uint8_t *data,
               std::size_t buffer_size,
               int x,
               int y,
               int width,
               int height,
               int frame_width,
               int frame_height,
               int stride,
               std::uint8_t value)
{
    if (!data || buffer_size == 0 || stride <= 0 || width <= 0 || height <= 0) {
        return;
    }

    const int start_x = clamp(x, 0, frame_width);
    const int start_y = clamp(y, 0, frame_height);
    const int end_x = clamp(x + width, 0, frame_width);
    const int end_y = clamp(y + height, 0, frame_height);

    for (int row = start_y; row < end_y; ++row) {
        const std::size_t offset = static_cast<std::size_t>(row) * stride + start_x;
        const std::size_t count = static_cast<std::size_t>(end_x - start_x);
        if (offset >= buffer_size) {
            continue;
        }
        std::memset(data + offset, value, std::min(count, buffer_size - offset));
    }
}

void fill_chroma(std::uint8_t *data,
                 std::size_t buffer_size,
                 int x,
                 int y,
                 int width,
                 int height,
                 int frame_width,
                 int frame_height,
                 int uv_stride,
                 std::size_t uv_offset,
                 std::uint8_t u,
                 std::uint8_t v)
{
    if (!data || buffer_size == 0 || uv_stride <= 0 || width <= 0 || height <= 0) {
        return;
    }

    int start_x = clamp(x, 0, frame_width);
    int start_y = clamp(y, 0, frame_height);
    int end_x = clamp(x + width, 0, frame_width);
    int end_y = clamp(y + height, 0, frame_height);

    start_x &= ~1;
    start_y &= ~1;
    end_x = (end_x + 1) & ~1;
    end_y = (end_y + 1) & ~1;

    for (int row = start_y / 2; row < end_y / 2; ++row) {
        for (int col = start_x; col < end_x; col += 2) {
            const std::size_t offset = uv_offset + static_cast<std::size_t>(row) * uv_stride + col;
            if (offset + 1 >= buffer_size) {
                continue;
            }
            data[offset] = u;
            data[offset + 1] = v;
        }
    }
}

void fill_nv12_rect(std::uint8_t *data,
                    std::size_t buffer_size,
                    int x,
                    int y,
                    int width,
                    int height,
                    int frame_width,
                    int frame_height,
                    int y_stride,
                    int uv_stride,
                    std::size_t uv_offset,
                    Nv12Color color)
{
    fill_luma(data, buffer_size, x, y, width, height,
              frame_width, frame_height, y_stride, color.y);
    fill_chroma(data, buffer_size, x, y, width, height,
                frame_width, frame_height, uv_stride, uv_offset, color.u, color.v);
}

bool sync_dmabuf(int fd, std::uint64_t flags)
{
    dma_buf_sync sync{};
    sync.flags = flags;
    if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) == 0) {
        return true;
    }

    std::fprintf(stderr, "DMA-BUF sync failed: %s\n", std::strerror(errno));
    return false;
}

}  // namespace

int clamp(int value, int low, int high)
{
    return std::max(low, std::min(value, high));
}

int align_up(int value, int alignment)
{
    if (alignment <= 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

void clip_box(Rect &rect, int width, int height)
{
    if (width <= 0 || height <= 0) {
        rect = {};
        return;
    }

    rect.x1 = clamp(rect.x1, 0, width - 1);
    rect.y1 = clamp(rect.y1, 0, height - 1);
    rect.x2 = clamp(rect.x2, 0, width - 1);
    rect.y2 = clamp(rect.y2, 0, height - 1);
}

void clip_box(int &x1, int &y1, int &x2, int &y2, int width, int height)
{
    Rect rect{x1, y1, x2, y2};
    clip_box(rect, width, height);
    x1 = rect.x1;
    y1 = rect.y1;
    x2 = rect.x2;
    y2 = rect.y2;
}

int draw_rect_nv12(const Nv12Dmabuf &buffer,
                   Rect rect,
                   int thickness,
                   Nv12Color color)
{
    if (buffer.fd < 0 || buffer.map_size == 0 || buffer.buffer_size == 0 ||
        buffer.memory_offset > buffer.map_size ||
        buffer.buffer_size > buffer.map_size - buffer.memory_offset) {
        std::fprintf(stderr, "invalid NV12 dmabuf\n");
        return -1;
    }

    clip_box(rect, buffer.width, buffer.height);
    if (rect.x2 <= rect.x1 || rect.y2 <= rect.y1) {
        std::fprintf(stderr, "invalid box: %d %d %d %d\n", rect.x1, rect.y1, rect.x2, rect.y2);
        return -1;
    }

    const int rect_width = rect.x2 - rect.x1;
    const int rect_height = rect.y2 - rect.y1;
    thickness = clamp(thickness, 1, std::min(rect_width, rect_height));

    void *mapped = mmap(nullptr, buffer.map_size, PROT_READ | PROT_WRITE, MAP_SHARED, buffer.fd, 0);
    if (mapped == MAP_FAILED) {
        std::fprintf(stderr, "NV12 dmabuf mmap failed: %s\n", std::strerror(errno));
        return -1;
    }

    sync_dmabuf(buffer.fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);

    auto *data = static_cast<std::uint8_t *>(mapped) + buffer.memory_offset;
    fill_nv12_rect(data, buffer.buffer_size, rect.x1, rect.y1, rect_width, thickness,
                   buffer.width, buffer.height, buffer.y_stride, buffer.uv_stride,
                   buffer.uv_offset, color);
    fill_nv12_rect(data, buffer.buffer_size, rect.x1, rect.y2 - thickness, rect_width, thickness,
                   buffer.width, buffer.height, buffer.y_stride, buffer.uv_stride,
                   buffer.uv_offset, color);
    fill_nv12_rect(data, buffer.buffer_size, rect.x1, rect.y1, thickness, rect_height,
                   buffer.width, buffer.height, buffer.y_stride, buffer.uv_stride,
                   buffer.uv_offset, color);
    fill_nv12_rect(data, buffer.buffer_size, rect.x2 - thickness, rect.y1, thickness, rect_height,
                   buffer.width, buffer.height, buffer.y_stride, buffer.uv_stride,
                   buffer.uv_offset, color);

    sync_dmabuf(buffer.fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);
    munmap(mapped, buffer.map_size);
    return 0;
}

}  // namespace cpu_draw
