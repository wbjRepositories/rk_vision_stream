#pragma once

#include <cstddef>
#include <cstdint>

namespace cpu_draw {

struct Rect {
    int x1{0};
    int y1{0};
    int x2{0};
    int y2{0};
};

struct Nv12Color {
    std::uint8_t y{76};
    std::uint8_t u{84};
    std::uint8_t v{255};
};

struct Nv12Dmabuf {
    int fd{-1};
    std::size_t memory_offset{0};
    std::size_t buffer_size{0};
    std::size_t map_size{0};
    int width{0};
    int height{0};
    int y_stride{0};
    int uv_stride{0};
    std::size_t uv_offset{0};
};

int clamp(int value, int low, int high);
int align_up(int value, int alignment);
void clip_box(Rect &rect, int width, int height);
void clip_box(int &x1, int &y1, int &x2, int &y2, int width, int height);

int draw_rect_nv12(const Nv12Dmabuf &buffer,
                   Rect rect,
                   int thickness,
                   Nv12Color color = {});

}  // namespace cpu_draw
