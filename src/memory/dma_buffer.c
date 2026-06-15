#define _GNU_SOURCE

#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>  // 解决 Cache 一致性宏
#include <linux/dma-heap.h> // DMA-Heap 宏
#include "dma_buffer.h"

/**
 * @brief 辅助函数：从 Linux DMA-Heap 分配一块物理连续内存
 * @param size 需要的内存大小 (通常是上一步 format 协商出来的 sizeimage)
 * @return 成功返回内存的 fd，失败返回 -1
 */
int dmabuf_alloc(size_t size) {
    // RK3588 通常有 /dev/dma_heap/system 或 cma，视设备树而定
    int heap_fd = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        perror("打开 dma_heap 失败 (请检查RK3588内核是否开启了dma-heap)");
        return -1;
    }

    struct dma_heap_allocation_data heap_data;
    memset(&heap_data, 0, sizeof(heap_data));
    heap_data.len = size;
    // O_CLOEXEC: 避免子进程继承; O_RDWR: 允许读写
    heap_data.fd_flags = O_RDWR | O_CLOEXEC; 

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &heap_data) < 0) {
        perror("DMA-Heap 内存分配失败");
        close(heap_fd);
        return -1;
    }

    close(heap_fd); // 可以关闭堆节点，分配出来的 buffer fd 依然有效
    return heap_data.fd;
}

