GCU 是燃原的 AI 计算加速设备。TopsCC 是基于 GCU 的编程平台，TopsCC 包含一套工具集和 runtime 库，支持 C/C++ 编程，可以生成在设备端和主机端运行的程序。本文件主要介绍如何在 GCU 上通过 TopsCC 进行算子编程。
表 1‑2 词汇表：

| 术语      | 描述                                   |
|-----------|----------------------------------------|
| GCU       | 燃原 AI 加速卡                         |
| TopsCC    | 燃原编程平台                           |
| clang     | C/C++ 编译器                           |
| llvm      | 编译器套件                             |
| Kernel    | 核函数，运行在设备端的程序             |
| Fatbin    | 包含了设备端和主机端运行程序的二进制文件 |
| DTE       | 数据搬运引擎                           |
| RTC       | 运行时编译                             |
| SIMT      | 单指令多线程                           |
| SIP       | 硬件计算核心                           |
| i20       | 第二代 GCU 推理卡                      |
| GCU210    | 第二代 GCU 推理卡，和 i20 等价         |
| TopsRider | 燃原 SDK 开发套件                     |
---
# 2 简介
人工智能领域对计算性能的需求极高。GCU 作为强大的计算引擎，提供了必要的算力支撑。而 TopsCC 则扮演了编程平台的角色，它通过优化编程环境，使得 GCU 的计算潜能得到更充分的释放。TopsCC 通过扩展 C++ 语言，使得开发者能够以接近 C++ 的编程方式，高效地为 GCU 编写程序。
GCU 具备多计算核心和多级存储的设计，这些架构特性会反映到具体的编程模型上。
（图 2‑1 GCU 架构图）
## 2.1 计算核心
SIP 计算核心是燃原科技打造面向云端数据中心的人工智能训练一体芯片采用全新的通用计算单元 GCU‑CARE 架构，为深度学习提供强大的算力支持。计算核心支持标量、向量和张量计算。通过燃原科技自有知识产权的软硬件架构 TopsRider，可以广泛地支持视觉、语音、NLP、推荐、LLM 等各技术方向的模型训练与推理。i20 一共含 24 个计算核心。
## 2.2 多级存储
GCU 包含 L1–L3 三级存储。

(图 2‑2 三级存储)

L1: 每个计算核心内部包含了一个私有存储（L1）。i20 中每一个计算核心具有 1 M 的 L1。
L2: 12 个计算核心可以组成一个计算簇，同一个 GCU 内包含多个计算簇。每个计算簇内的计算核心可以共享一个簇内的 24 M L2 共享存储。
L3: GCU 拥有一个全局设备存储（L3），对所有计算簇可见，i20 的 L3 大小为 16 GB。
L1、L2 的介质是 SRAM，L3 介质一般是 HBM 或者 GDDR。

---

# 3 TopsCC 编程模型

## 3.1 概述

### 3.1.1 函数类型限定符

TopsCC 支持设备端和主机端混合编程，使用 __device__ 和 __global__ 标记设备端程序，使用 __host__ 标记主机端程序。没有标记的函数默认为主机端程序。

示例代码：
```cpp
#include <tops/tops_runtime.h>

__global__ void foo() {
    // TODO: add code here
}

int main(int argc, char **argv) {
    foo<<<1/*grid dim*/, 1/*block dim*/>>>();
    assert(topsGetLastError() == topsSuccess);

    foo<<<dim3(1,1,1), dim3(1,1,1)>>>();
    assert(topsGetLastError() == topsSuccess);
    return 0;
}
```

表 3‑1 函数类型限定符：

| 函数类型限定符 | 执行                     | 调用                                                                 |
|----------------|--------------------------|-----------------------------------------------------------------------|
| __device__     | 在设备端运行的函数       | 可以被 __global__ 函数调用，但不能被 __host__ 函数调用                 |
| __global__     | 在设备端运行的入口函数   | 只能由 __host__ 函数启动，不能被 __device__ 或其他 __global__ 函数调用 |
| __host__       | 在主机端运行的函数       | 只能被 __host__ 函数调用                                             |

说明：一个函数可以同时标记为 __device__ 和 __host__，这种函数在设备端和主机端都能被调用。没有标记的函数默认都是 __host__ 函数。

### 3.1.2 线程模型

TopsCC 支持类似 SIMT 的编程模型，多个 Thread 可同时执行同一份代码。Thread 的层次结构分为 Thread、Block 和 Grid 三层，每层可用 x、y、z 三个维度指定程序运行的层次结构。

- **Thread**：对应一个标量或向量执行程序，物理上映射到一个计算核心 SIP 上执行。在线程内部可以通过 `threadIdx.x`、`threadIdx.y`、`threadIdx.z` 获取当前 thread 的坐标。
- **Block**：包含一组 Thread，物理上映射到一组 SIP。在线程内部可以通过 `blockIdx.x`、`blockIdx.y`、`blockIdx.z` 获取当前 block 的坐标。可以使用 `blockDim.x`、`blockDim.y`、`blockDim.z` 指定 block 的维度大小，作为内核调用符 `<<<>>>` 的参数。
  - 硬件特性：在 gcu210 上 block dims 的乘积最大为 12，即 `blockDim.x * blockDim.y * blockDim.z <= 12`。
- **Grid**：包含一组 Block。可以使用 `gridDim.x`、`gridDim.y`、`gridDim.z` 指定 grid 的维度大小作为内核调用符 `<<<>>>` 的参数。

图 3‑1 thread 模型：线程按照 Thread → Block → Grid 三级结构组织，并映射到芯片上多个 SIP 计算核心和存储层级。

TopsCC 支持 cooperative 模式。当一个 __global__ 函数（kernel）被标记为 cooperative 模式时，这个 kernel 会被运行时使用 `topsLaunchCooperativeKernel` 启动。在这种模式下 grid 中所有的 block 会同时运行。内核中可以使用 `__syncblocks()` 调用进行 block 间的同步。常规 kernel 不受“乘积≤2”限制；仅对 cooperative kernel，grid 的总块数不得超过设备的 multiProcessorCount（i20 为 2），详见 3.2.3 的“资源限制”。

示例代码：
```cpp
__global__ __cooperative__ void foo() {
    __syncblocks();
}
```

### 3.1.3 存储模型

i20 GCU 的存储系统结构如下图所示（图 3‑2）。用户视角可见以下地址空间：

- `__device__`：全局的地址空间，所有 block 可见。对应硬件的全局设备存储（L3）。

  ```c
  __device__ int data[100];
  ```

说明：在 GCU 2.0 中，kernel/SIP 不直接访问 L3，需要由 DTE 将数据搬运到 L1 后访问。除 __constant__ 只读区可被直接访问（实现上由架构/编译器映射），其余 L2/L3 需经 DTE。可直访：L1(Private) 与 __constant__；其余（L2/L3 对应 Shared/Global）一律经 DTE 先搬到 L1 再算。声明修饰仅用于 DTE/编译器识别地址空间，并不改变直访能力。

* `__constant__`：全局的常量地址空间，所有 thread 可见。对应硬件的全局设备存储（L3）。

  ```c
  __constant__ int a = 2;
  ```

* `__shared__`：block 内共享地址空间，block 内的 thread 可见。对应硬件的簇内共享存储（L2）。

  ```c
  __shared__ int data[100];        // static size shared memory
  extern __shared__ int data2[];   // dynamic size shared memory
  ```

  说明：在 GCU 2.0 中，不允许直接访问 L2，只能用于 DTE 数据搬运操作。动态大小的 shared memory 每个 **global** 函数只能使用一个。每个计算簇最大支持 24 MB 的 shared memory。计算只能直访 L1 与 __constant__；__device__/__shared__ 仅作 DTE 搬运的来源/去向”

* 无修饰符变量：位于 thread 的私有地址空间，对应硬件的计算核心私有存储（L1）。

  ```c
  int data;
  float data2[100];
  ```

* `__valigned__`：对于私有空间的存储，如果会被向量操作使用，需要使用 `__valigned__` 进行对齐。

  ```c
  __valigned__ int data[100];
  ```

### 3.1.4 内建的宏

* `__TOPS_DEVICE_COMPILE__`：编译设备端代码时会被定义。
* `__GCU_ARCH__`：由 3 位数字组成；i20 的值为 210。

## 3.2 编程接口

TopsCC 通过扩展 C++，提供设备端和主机端的运行时库来支持基于 C++ 的编程。编译方式包括两种：离线编译和运行时编译,比赛是离线编译。


### 图 3‑3 TopsCC 离线编译

离线编译流程中，C/C++ 源代码与 kernel 库、host 库一起经由编译器生成 fatbin 文件，运行时由 CPU 和 GCU 装载执行。本次竞赛采用离线编译方式。

### 3.2.3 一个简单的程序例子

示例程序如下。它在 GCU 上启动一个空的 kernel 函数 `foo`，`dim3(1,1,1)` 与 `1` 等价。

```c++
#include <tops/tops_runtime.h>

__global__ void foo() { }

int main(int argc, char **argv) {
    // 启动一个 block，block 中有一个 thread 执行 kernel 函数 foo
    foo<<<1/*grid dim*/, 1/*block dim*/>>>();
    assert(topsGetLastError() == topsSuccess);

    foo<<<dim3(1,1,1), dim3(1,1,1)>>>();
    assert(topsGetLastError() == topsSuccess);
    return 0;
}
```

**内核调用运算符 `<<<grid_dim, block_dim, share_memory_sz, stream>>>`** 接受 4 个参数：

1. `grid_dim`：Grid 的尺寸，可参照线程层次结构选择。
2. `block_dim`：Block 的尺寸，同样参照线程层次结构。
3. `share_memory_sz`：kernel 在 block 中申请的动态共享数组字节数（如 `__shared__ int data[]`）。默认值为 0 Byte；如果申请大小超过硬件共享内存限制，程序会出错。
4. `stream`：流对象，默认为空；用于异步调用。

建议在 kernel 调用后使用 `topsGetLastError` 检查执行是否成功。

**资源限制：**最大 grid 尺寸可以通过 `topsGetDeviceProperties` 查询。gcu210 硬件限制为 `gridDim.x ≤ 65536`、`gridDim.y ≤ 256`、`gridDim.z ≤ 256`。对于 cooperative 模式，grid 的总大小 `gridDim.x * gridDim.y * gridDim.z` 不得超过设备的 `multiProcessorCount`（gcu210 为 2）。block 的总线程数不得超过设备 `maxThreadsPerMultiProcessor`（gcu210 为 12）。下面代码展示如何查询这些属性并设置 block 大小：

```c++
topsDeviceProp_t prop;
int deviceId = 0;
topsGetDeviceProperties(&prop, deviceId);

dim3 blockDims;
blockDims.x = prop.maxThreadsPerMultiProcessor;
blockDims.y = 1;
blockDims.z = 1;

foo<<<1, blockDims>>>();
```

### 3.2.4 设备端编程

#### 打印和断言

在 GCU 2.0 中 kernel 直接访问的存储地址只支持私有地址空间（L1）和 `__constant__` 地址空间；对于 `__shared__` 和 `__device__` 地址空间，需要通过 DTE 将数据搬运到 L1 后访问。使用 `printf` 和 `assert` 有以下限制：

* kernel 函数中可以使用 `printf`，但格式字符串必须是常量，且不支持 `%p` 和 `%.` 等格式。
* `assert` 默认在 O3 优化级别关闭，如需开启可在编译时定义 `-DTOPS_ENABLE_ASSERT`。
* 使用 printf/assert 时会触发调试同步模式（同一 stream 串行），详见 3.4 编程限制汇总。

#### 3.2.4.1 数据流编程

TopsCC 使用 DTE（Data Transfer Engine）接口进行数据搬运，允许计算与搬运并行。
一次数据搬运，通常包含以下流程：  
1. 声明DTE上下文
2. 使用mdspan给memory加入信息
3. 使用ctx操作接口

一个DTE编程示例如下，将数据从设备端指针from线性拷贝到设备端指针to（from和to所指向的均为L3上的内存）：
```c
__global__ void copy_d2d(int *from, int *to, size_t N) {
  __private_dte__tops_dte_ctx_t ctx;  // Declare a DTE Context
  tops::dte_scope s(ctx);             // Initalize DTE Context

  tops::mdspan src(tops::Global, from, N);   // Add shape info for source
  tops::mdspan dst(tops::Global, to, N);      // Add shape info for dest

  tops::memcpy(ctx, dst, src);        // Copy data from source to dest
}
```

##### 支持的标量数据类型

* 基本类型：`bool`、`char`、`unsigned char`、`short`、`unsigned short`、`int`、`unsigned int`、`float`。
* 扩展浮点类型：`tops::half` 和 `tops::bfloat`。例如可通过 `#include <tops/half.h>` 和 `#include <tops/bfloat.h>` 引入。

代码：tops::half 和tops::bfloat 的定义方式示例
```c
#include<tops/half.h>
#include<tops/bfloat.h>

__device__ void test(){
  tops::half value1(0.2);
  tops::bfloat value2(2.4);
}
```

##### DTE Context（DTE 上下文）

在 `__shared__` 或 `__device__` 地址空间上的数据需通过 DTE 搬运。使用 DTE 前需声明 DTE 上下文，TopsCC 支持三种类型：

* **Block 级共享 DTE Context**：`__shared_dte__ tops_dte_ctx_t ctx[n];`——block 内线程共享 DTE 上下文，只能支持 Global 和 Shared 之间的数据传输。
* **Block 级私有 DTE Context**：`__private_dte__ tops_dte_ctx_t ctx;`——Block 级的 DTE 资源，只能支持 Global 和 Shared 之间的数据传输。
* **Thread 级 DTE Context**：`tops_dte_ctx_t ctx;`——线程私有 DTE 资源，可支持 Global、Shared 和 Private 之间的数据传输。
仅 Thread 级 DTE ctx 支持涉及 tops::Private（L1）的搬运（Global/Shared/Private 任意组合）。
Block 级 ctx（共享/私有） 仅用于 Global↔Shared（L3↔L2） 的数据传输，不直接触达 L1。

##### mdspan

TopsCC使用一种名为`mdspan`的数据结构来给设备地址附加额外的信息（如维度、形状、所属内存空间、总大小等），其构造函数参数包括：

1. 所属地址空间（可选）：`tops::Global`、`tops::Shared`、`tops::Private`，对应 L3/L2/L1。
2. 起始地址指针。
3. 形状维度大小列表。

DTE相关结构都使用`mdspan`作为配置参数。

示例：声明 `mdspan` 的两种方式：

```c++
// method 1
tops::mdspan src1(tops::Global, from, N, H, W, C);
tops::mdspan src2(from, N, H, W, C);

// method 2
auto shape = {N, H, W, C};
tops::mdspan src3(tops::Global, from, shape);
tops::mdspan src4(from, shape);
```

其中 `from` 为某数据类型的起始地址，DTE 操作一般支持九种数据类型（`int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`tops::bfloat`、`tops::half` 和 `float`）。`shape`是数据的形状，`shape[0]` 为数据最高维度的大小，通常对应内存中步幅最大（最不连续）的那一维。维度约定：shape 从高维到低维依次列出，最后一个元素对应X 轴（最快变化维），倒数第二个元素对应 Y 轴。本文 mirror_tb 以 X 轴翻转，mirror_lr 以 Y 轴翻转。

##### DTE 支持的操作模式

DTE 支持两种使用模式：

1. **配置与启动分离模式**：适用于计算与搬运流水并行的场景。先调用配置接口，再调用启动接口。支持同步和异步两种启动方式。

   * **同步启动**（`trigger_and_wait`）：

     ```c++
     tops_dte_ctx_t ctx;
     ctx.init();
     ctx.config_memcpy(dst, src);
     ctx.trigger_and_wait();
     ctx.destroy();
     ```

   * **异步启动**（`trigger`）：返回 `tops::event`，可用 `tops::wait` 等待。

     ```c++
     tops_dte_ctx_t ctx;
     ctx.init();
     ctx.config_memcpy(dst, src);
     tops::event ev = ctx.trigger();
     tops::wait(ev);
     ctx.destroy();
     ```

2. **配置和启动合并模式**：代码更简洁，同时支持同步和异步两种方式。

   * **同步启动**：

     ```c++
     tops_dte_ctx_t ctx;
     ctx.init();
     tops::memcpy(ctx, dst, src);
     ctx.destroy();
     ```

   * **异步启动**：调用以 `_async` 结尾的函数返回 `tops::event`，可用 `tops::wait` 等待。

     ```c++
     tops_dte_ctx_t ctx;
     ctx.init();
     auto ev = tops::memcpy_async(ctx, dst, src);
     tops::wait(ev);
     ctx.destroy();
     ```
注意异步接口 _async 返回 tops::event；同步版本返回 void。

##### 全量配置接口

全量配置需要完整设置 DTE 的参数。多数搬运接口含有两个 `mdspan` 参数，第一个为目标地址对象，第二个为源地址对象。常用接口见下表：

| 接口                                    | 描述                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| `ctx.config_memcpy(dst, src)`         | 以 `src` 总大小拷贝 `src` 到 `dst`，用户需确保 `dst` 足够大。                                   |
| `ctx.config_memset(dst, const_value)` | 将 `dst` 指定的内存设置为 `const_value`。                                                |
| `ctx.config_slice(dst, src, offset)`  | 按 `dst` 指定的形状和 `offset` 指定的位置从 `src` 中拷贝数据到 `dst`。若 `dst` 大小或偏置超出 `src`，将自动填充。**切片操作：** 在GCU的应用中，通常处理的数据量很大，而GCU的执行单元 SIP 所能访问的SIP memory比较小，所以在数据处理上，需要将大块数据切片搬运到 SIP memory，供SIP处理加工，这种操作称为slice，你想的操作称为deslice。根据数据处理需求，可选择以下切片方式：(1)**线性切片**：依次将相应的数据片段搬运到SIP memory中。一个典型的例子：如果要处理的是一个2维数组，而SIP每次可以处理数组中的完整一行，则可以对该数组进行线性切片。  (2)**等维切片**：对多维数组同时沿各维切割成多个形状相同的块（将一个 N 维数组切为多个小的 N 维数组）。一个典型的例子：将一个3维数组切分成多个小形状的3维数组，可以理解为把一个大的立方体切分成多个小立方体。 (3)**多维切片**：仅在指定的部分维度上进行切片。可以把线性切片和等维切片理解为多维切片的两个特例|
| `ctx.config_deslice(dst, src, offset)`                            | 把 `src` 指定的数据拷贝并覆盖到 `dst` 的 `offset` 位置。                 |
| `ctx.config_transpose(dst, src, layout)`                          | 按 `layout` 对 `src` 进行转置并拷贝至 `dst`，layout的数据排列定义为形如{0,1,2,3}。                       |
| `ctx.config_slice_transpose(dst, src, offset, layout)`            | slice和transpose的组合，先对 `src` 切片，再按 `layout` 转置切片并放入 `dst`。                   |
| `ctx.config_transpose_deslice(dst, src, offset, layout)`          | transpose和deslice的组合，先对 `src` 按 `layout` 转置，再拷贝并覆盖到 `dst` 指定位置。               |
| `ctx.config_pad(dst, src, pad_low, pad_high, pad_mid, pad_value)` | 垫片操作，把src指定的数据，按照dst所指定的形状和大小，用pad_value的值设置到src的首部（pad_low有效），尾部（pad_high有效），或者中间（pad_mid有效），并把结果移动到dst所指的位置。 |
| `ctx.config_mirror_tb(dst, src)`                                  | 按第一维（X 轴，shape的最后一个元素）翻转 `src`，结果放入 `dst`。                        |
| `ctx.config_mirror_lr(dst, src)`                                  | 按第二维（Y 轴，shape的倒数第二个元素）翻转 `src`，结果放入 `dst`。                        |
| `ctx.config_broadcast(dst, src)`                                  | 根据 `src` 到 `dst` 维度变化扩展 `src` 数据，结果放入 `dst`。             |

##### 增量配置接口

增量配置接口是指在完成一次全量 DTE 配置后，后续仅对发生变化的部分调用相应的API进行配置，可以减少DTE配置时间。增量配置接口可以在不修改DTE数据搬运操作的类型的情况下改变 DTE context 的配置。

* **基础增量接口：**

| 接口                       | 描述          |
| ------------------------ | ----------- |
| `ctx.set_dst_addr(addr)` | 设置新的 dst 地址 |
| `ctx.set_src_addr(addr)` | 设置新的 src 地址 |
| `ctx.set_dst_offset(dim, offset)`               | 设置 `dst` 在维度 `dim` 的偏移 |
| `ctx.set_src_offset(dim, offset)`               | 设置 `src` 在维度 `dim` 的偏移 |
| `ctx.set_dst_dim_size(dim, size)`               | 设置 `dst` 在维度 `dim` 的大小 |
| `ctx.set_src_dim_size(dim, size)`               | 设置 `src` 在维度 `dim` 的大小 |
| `ctx.set_total_size(size)`                      | 设置整体大小，常用于 memcpy      |
| `ctx.set_transpose_layout(layout)`              | 设置转置的 `layout`         |
| `ctx.set_pad_config(pad_low, pad_high, ad_mid)` | 设置 pad 参数              |

##### 启动和同步接口

* **启动接口（配置与启动分离时使用）：**

| 接口                       | 描述                     |
| ------------------------ | ---------------------- |
| `ctx.trigger()`          | 触发 DTE 操作，返回一个 `event` |
| `ctx.trigger_and_wait()` | 触发 DTE 操作并等待完成         |

* **同步接口：**

| 接口            | 描述              |
| ------------- | --------------- |
| `wait(event)` | 等待指定 `event` 完成 |

##### 配置和启动合并接口

在接口中，ctx参数是指定DTE上下文，dst是mdspan指定的数据搬运的输出位置，src是mdspan指定的数据搬运的输入位置。
使用下列函数可同时完成配置和启动，均支持九种数据类型，并提供 `_async` 后缀的异步版本：

| 接口                                                                | 描述                                       |
| ----------------------------------------------------------------- | ---------------------------------------- |
| `tops::memcpy(ctx, dst, src)`                                     | 以 `src` 总大小拷贝 `src` 到 `dst`              |
| `tops::memset(ctx, dst, const_value)`                             | 将 `dst` 指定的内存设置为 `const_value`           |
| `tops::slice(ctx, dst, src, offset)`                              | 按 `dst` 形状和 `offset` 从 `src` 拷贝数据到 `dst` |
| `tops::deslice(ctx, dst, src, offset)`                            | 把 `src` 数据拷贝并覆盖到 `dst` 指定位置              |
| `tops::transpose(ctx, dst, src, layout)`                          | 按 `layout` 转置 `src` 并拷贝至 `dst`           |
| `tops::slice_transpose(ctx, dst, src, offset, layout)`            | 先切片再转置再拷贝至 `dst`                         |
| `tops::transpose_deslice(ctx, dst, src, offset, layout)`          | 先转置再覆盖到 `dst`                            |
| `tops::pad(ctx, dst, src, pad_low, pad_high, pad_mid, pad_value)` | 按形状填补并拷贝至 `dst`                          |
| `tops::mirror_tb(ctx, dst, src)`                                  | 第一维翻转并拷贝至 `dst`                          |
| `tops::mirror_lr(ctx, dst, src)`                                  | 第二维翻转并拷贝至 `dst`                          |
| `tops::broadcast(ctx, dst, src)`                                  | 按维度变化扩展 `src` 并拷贝至 `dst`                 |

##### DTE 软件流水编程

下面给出“双缓冲 + 事件”的软件流水模板，可直接复用。
```c
    tops_dte_ctx_t ctxs[2][2];
    tops::event evs[2][2];

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ctxs[i][j].init();

    __valigned__ int input_buffer[2][tile_size];
    __valigned__ int output_buffer[2][tile_size];

    tops::mdspan input(tops::Global, in, tile_size);  // in 为 L3 指针
    tops::mdspan output(tops::Global, out, tile_size); // out 为 L3 指针

    for (int i = 0; i < 2; i++)
        ctxs[0][i].config_memcpy(tops::mdspan(tops::Private,
                               input_buffer[i], tile), input, tile_size);

    for (int i = 0; i < 2; i++)
        ctxs[1][i].config_memcpy(output,
                               tops::mdspan(tops::Private, output_buffer[i], tile), tile_size);

    evs[0][0] = ctxs[0][0].trigger();
    int iter = 0;

    for (int i = 0; i < size; i += tile_size) {
        evs[0][iter%2].wait();
        if (i + tile_size < size) {
            ctxs[0][(iter+1)%2].set_src_addr(in + i);
            evs[0][(iter+1)%2] = ctxs[0][(iter+1)%2].trigger();
        }

        // do computation
        foo(input_buffer[iter%2], output_buffer[iter%2]);

        if (i != 0) {
            evs[1][(iter-1)%2].wait();
        }
        ctxs[1][iter%2].set_dst_addr(output + i);
        evs[1][iter%2] = ctxs[1][iter%2].trigger();
        if (i + tile_size >= size) {
            evs[1][iter%2].wait();
        }
        iter++;
    }

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            ctxs[i][j].destroy();
```

默认情况下，DTE 的非法行为不会报错，加上宏 -DTOPS_ENABLE_DTE_CHECK 会检查非法行为。

#### 3.2.4.2 计算流程编程

一般地，topscc 支持标量计算。计算只能发生在线程内部，且从 L1 中读取数据。对于 L1 的地址，可以直接使用下标索引数据。

```c
    __valigned__ int inp[128];
    __valigned__ int out[128];
    for (size_t i = 0; i < 128; ++i) {
        out[i] = inp[i] * inp[i];
    }
```

此外，TopsCC 提供了向量接口和矩阵计算接口，以利用 GCU 的 1D 和 2D 算力。

##### 1D 计算流编程

一个 `vector` 类型默认长度为 **128** 字节，支持的 `vector` 类型如下所示。

###### 表 3-7 支持的 `vector` 类型

| 类型        | 说明                    | 默认的元素个数                                   |
| --------- | --------------------- | ----------------------------------------- |
| `vchar`   | `char` 向量类型           | 一个 `vchar` 向量包含 **128** 个 `int8_t`        |
| `vuchar`  | `unsigned char` 向量类型  | 一个 `vuchar` 向量包含 **128** 个 `uint8_t`      |
| `vshort`  | `short` 向量类型          | 一个 `vshort` 向量包含 **64** 个 `int16_t`       |
| `vushort` | `unsigned short` 向量类型 | 一个 `vushort` 向量包含 **64** 个 `uint16_t`     |
| `vint`    | `int` 向量类型            | 一个 `vint` 向量包含 **32** 个 `int`             |
| `vuint`   | `unsigned int` 向量类型   | 一个 `vuint` 向量包含 **32** 个 `unsigned int`   |
| `vfloat`  | `float` 向量类型          | 一个 `vfloat` 向量包含 **32** 个 `float`         |
| `vhalf`   | `half` 向量类型           | 一个 `vhalf` 向量包含 **64** 个 `tops::half`     |
| `vbfloat` | `bfloat` 向量类型         | 一个 `vbfloat` 向量包含 **64** 个 `tops::bfloat` |

> 支持的 `vector` 操作包括：

###### 表 3-8 支持的 `vector` 操作（I）

| 接口           | 描述                                                                                       |
| ------------ | ---------------------------------------------------------------------------------------- |
| `vload`      | 从指定地址开始读取一个向量数据。`vload` 访问的地址需要对齐，即需要用 `__valigned__` 修饰。例如：`auto v = vload<vint>(addr);` |
| `vstore`     | 存储一个向量数据到某个指定地址。例如：`vstore(value, addr);`                                                |
| `vlength`    | 根据给定的数据类型，返回对应的 `vector` 计算所支持的向量长度。例如：`__valigned__ int buf[tops::vlength<vint>()]`      |
| `vzero`      | 返回一个向量，所有值都设置为 `0`                                                                       |
| `vadd`       | 返回两个向量的和。例如：`vint sum = tops::vadd(lhs, rhs)`                                            |
| `vsub`       | 返回两个向量的差。例如：`vint diff = tops::vsub(lhs, rhs)`                                           |
| `vmul`       | 返回两个向量的乘积。例如：`vint prdt = tops::vmul(lhs, rhs)`                                          |
| `vdiv`       | 返回两个向量的商。例如：`vint quot = tops::vdiv(lhs, rhs)`                                           |
| `vmod`       | 返回两个向量的模。例如：`vint md = tops::vmod(lhs, rhs)`                                             |
| `vrem`       | 返回两个向量的余数。例如：`vint rm = tops::vrem(lhs, rhs)`                                            |
| `vsign`      | 返回一个向量中每个元素的“符号”型，正数返回 1，负数返回 −1。例如：`vint sgn = tops::vsign(v)`                          |
| `vbroadcast` | 将一个标量的值赋给向量的所有成员。例如：`vint brd = tops::vbroadcast(int2)`                                  |
| `vcast`      | 因为向量类型不支持隐式转换，所以可以用这个函数进行**显示类型转换**                                                      |
| `vbitcast`   | 将一个向量强制转换为另外一个相同大小的向量。例如：`vchar cv = tops::vbitcast(iv)`                                 |
| `vsin`       | 返回一个向量每个元素的正弦，仅支持 `vfloat` 类型。例如：`auto vsn = tops::vsin(fv)`                             |
| `vasin`      | 返回一个向量每个元素的反正弦，仅支持 `vfloat` 类型。例如：`auto vasn = tops::vasin(fv)`                          |
| `vsinh`  | 返回一个向量每个元素的双曲正弦，仅支持 `vfloat` 类型。例如：`auto vhs = tops::vsinh(fv)`                                               |
| `vasinh` | 返回一个向量每个元素的反双曲正弦，仅支持 `vfloat` 类型。例如：`auto vahs = tops::vasinh(fv)`                                            |
| `vcos`   | 返回一个向量每个元素的余弦，仅支持 `vfloat` 类型。例如：`auto vcs = tops::vcos(fv)`                                                  |
| `vcosh`  | 返回一个向量每个元素的双曲余弦，仅支持 `vfloat` 类型。例如：`auto vhcs = tops::vcosh(fv)`                                              |
| `vacos`  | 返回一个向量每个元素的反余弦，仅支持 `vfloat` 类型。例如：`auto vacs = tops::vacos(fv)`                                               |
| `vacosh` | 返回一个向量每个元素的反双曲余弦，仅支持 `vfloat` 类型。例如：`auto vhacs = tops::vacosh(fv)`                                           |
| `vabs`   | 返回一个向量每个元素的绝对值，仅支持 `vfloat` 类型。例如：`auto vabso = tops::vabs(fv)`                                               |
| `vcbrt`  | 返回一个向量每个元素的立方根，仅支持 `vfloat` 类型。例如：`auto vcbr = tops::vcbrt(fv)`                                               |
| `vtan`   | 返回一个向量每个元素的正切，仅支持 `vfloat` 类型。例如：`auto vtn = tops::vtan(fv)`                                                  |
| `vatan`  | 返回一个向量每个元素的反正切，仅支持 `vfloat` 类型。例如：`auto vatn = tops::vatan(fv)`                                               |
| `vatan2` | 将两个向量每个元素分别相除，再对结果进行反正切，仅支持 `vfloat` 类型。例如：`auto vatan2 = tops::vatan2(fv1, fv2)`                             |
| `vneg`   | 返回一个向量的符号相反的值，支持所有符号类型。例如：`auto vng = tops::vneg(fv)`                                                         |
| `vsqrt`  | 返回一个向量每个元素的平方根，仅支持 `vfloat` 类型。例如：`auto vsqt = tops::vsqrt(fv)`                                               |
| `vrsqrt` | 返回一个向量每个元素的**反平方根**，仅支持 `vfloat` 类型。例如：`auto vrsqt = tops::vrsqrt(fv)`                                        |
| `vfloor` | 返回一个向量每个元素的**向下取整**（最接近且不大于自身的整数），仅支持 `vfloat` 类型。例如：`auto vflr = tops::vfloor(fv)`                           |
| `vceil`  | 返回一个向量每个元素的**向上取整**（最接近且不小于自身的整数），仅支持 `vfloat` 类型。例如：`auto vcl = tops::vceil(fv)`                             |
| `vround` | 返回一个向量每个元素**最接近**的整数，仅支持 `vfloat` 类型。例如：`auto vrnd = tops::vround(fv)`                                        |
| `vtrunc` | 按截断规则 `trunc(x) = x >= 0 ? floor(x) : ceil(x)` 处理向量每个元素返回，仅支持 `vfloat` 类型。例如：`auto vtrc = tops::vtrunc(fv)`   |
| `vrint`  | 和 `vround` 很像，例如把 `x.5` 的浮点数，`round` 会处理成 `x+1`，`rint` 是 `x`；仅支持 `vfloat` 类型。例如：`auto vri = tops::vrint(fv)` |
| `vexp`   | 按输入向量的每个元素作为数学常数 `e` 的指数计算后，返回一个**相同类型**向量，仅支持 `vfloat` 类型。例如：`auto vxp = tops::vexp(fv)`                     |
| `vexpm1` | 按输入向量的每个元素作为指数的 **`e^x - 1`** 计算后，返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vxpm = tops::vexpm1(fv)`                |
| `vexp2`  | 按输入向量的每个元素为 2 的指数计算后，返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vxp2 = tops::vexp2(fv)`                              |
| `vlog`   | 按输入向量的每个元素为数学常数 `e` 的对数计算后，返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vlg = tops::vlog(fv)`                          |
| `vlog1p`    | 按输入向量的每个元素加 1 后作自然对数计算（`log(1+x)`），返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vlgp = tops::vlog1p(fv)`                     |
| `vlog2`     | 按输入向量的每个元素为 2 的对数计算，返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vlg2 = tops::vlog2(fv)`                                     |
| `vlog10`    | 按输入向量的每个元素为 10 的对数计算，返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vlg10 = tops::vlog10(fv)`                                  |
| `vlogb`     | 按输入向量的每个元素以 10 为底的对数，只保留**整数部分**，并返回**相同类型**向量，仅支持 `vfloat` 类型。例如：`auto v lgb = tops::vlogb(fv)`                    |
| `vilogb`    | 按输入向量的每个元素以 10 为底的对数，只保留结果的**整数部分**，并返回一个**整数型**向量，仅支持 `vfloat` 类型。例如：`auto vilgb = tops::vilogb(fv)`               |
| `vpower`    | 按第一个输入向量的每个元素为底数、第二个输入向量的对应元素为指数计算，并返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vpw = tops::vpower(fv1, fv2)`               |
| `vgelu`     | 计算输入向量每个元素的高斯误差线性单元，并返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vglu = tops::vgelu(fv)`                                    |
| `vsoftplus` | 按照规则 `vlog(vexp(v)+1)` 计算后，返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vstp = tops::vsoftplus(fv)`                          |
| `vsigmoid`  | 计算输入向量每个元素的 Sigmoid 函数，并返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vsm = tops::vsigmoid(fv)`                               |
| `vdim`      | 计算第一个输入向量和第二个向量的差值，如果差值是个负数则返回 `0`，并返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vsm = tops::vdim(fv1, fv2)`                 |
| `vhypot`    | 以第一个向量每个元素的直角边，以及第二个向量的对应元素作为第二直角边，计算相应的斜边，并返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vhpt = tops::vhypot(fv1, fv2)`    |
| `vcopysign` | 以第二个向量的每个元素的符号，作为第一个向量对应元素的符号，并返回一个相同类型向量，仅支持 `vfloat` 类型。例如：`auto vhpt = tops::vcopysign(fv1, fv2)`                |
| `visnan`    | 返回一个 `vuint` 的向量，它的元素的值为 `0` 代表输入向量的对应元素不是 `NaN`，其它位代表 `NaN`，仅支持 `vfloat` 类型。例如：`auto vnn = tops::visnan(fv)`       |
| `visfinite` | 返回一个 `vuint` 的向量，它的元素的值为 `0` 代表输入向量的对应元素是 `NaN` 或者 `INF`，其它代表不是，仅支持 `vfloat` 类型。例如：`auto vnn = tops::visfinite(fv)` |
| `vmax`      | 返回一个向量，该向量中每个元素为两个输入向量相应位置的**最大**值组成，支持所有类型。例如：`auto vmx = tops::vmax(v1, v2)`                                      |
| `vmin`      | 返回一个向量，该向量中每个元素为两个输入向量相应位置的**最小**值组成，支持所有类型。例如：`auto vmn = tops::vmin(v1, v2)`                                      |
| `vand`      | 按照两个输入向量的位计算“与”结果，并返回相同类型的向量，支持所有类型。例如：`auto vrt = tops::vand(v1, v2)`                                              |
| `vor`       | 按照两个输入向量的位计算“或”结果，并返回相同类型的向量，支持所有类型。例如：`auto vrt = tops::vor(v1, v2)`                                               |
| `vxor`      | 按照两个输入向量的位计算“异或”结果，并返回相同类型的向量，支持所有类型。例如：`auto vrt = tops::vxor(v1, v2)`                                             |
| `vnot`    | 按照两个输入向量的位计算“非”结果，并返回相同类型的向量，支持所有类型。例如：`auto vrt = tops::vnot(v1, v2)`                                                    |
| `vshl`    | 按照向量 `v` 中的每个元素指定的位数，按位向左移动向量 `in` 中对应元素的数值（**保留符号**），并返回相同类型的向量，不支持无符号类型。例如：`auto vrt = tops::vshl(v1, v2)`          |
| `vshr`    | 按照向量 `v` 中的每个元素指定的位数，按位向右移动向量 `in` 中对应元素的数值（**保留符号**），并返回相同类型的向量，不支持无符号类型。例如：`auto vrt = tops::vshr(v1, v2)`          |
| `vshli`   | 按照参数 `is` 指定的位数，按位向左移动向量 `in` 中对应元素的数值（**不保留符号**），并返回相同类型的向量，仅支持无符号类型。例如：`auto vrt = tops::vshli(iv, is)`                 |
| `vshri`   | 按照参数 `is` 指定的位数，按位向右移动向量 `in` 中对应元素的数值（**不保留符号**），并返回相同类型的向量，仅支持无符号类型。例如：`auto vrt = tops::vshri(iv, is)`                 |
| `vselect` | 按照第一个向量每个元素的条件（`0` 代表否），选择第二个向量对应元素（条件为是）或第三个（条件为否），并返回和后两个向量相同的类型的向量，支持所有类型。例如：`auto vsel = tops::vselect(vcnd, v1, v2)` |

#### 3.2.4.2 2D 计算流编程

（在 2D 计算相关题目里，会在题目说明中提供 2DAPI 使用方法。）

#### 3.2.4.3 同步

TopsCC 支持 **Block** 内的所有 **Thread** 同步，以及整个 **Grid** 的全局同步。

* `__syncthreads`：Block 内所有 Thread 做一次同步。
* `__syncblocks`：Grid 内所有 Thread 做一次同步。

> 说明：使用 `__syncblocks` 的 `kernel` 需必须声明为 `__cooperative__`。

### 3.2.5 主机端编程

> 注：本次竞赛主要考察设备端编程，主机端编程部分作为参考，帮助参赛者理解主机侧。

主机端运行时实现依赖 `TopsRT` 库中，基于 TopsCC 开发的应用程序会动态链接到 `libtopsrt.so`。运行时所有接口都以 `tops` 为命名前缀。运行时主要负责以下类别的管理：

* 执行环境：描述了主机运行时的设备管理和初始化过程。
* 存储系统：描述了运行时感知的存储管理系统。
* 异步并行：描述了在不同层面上如何通过运行时接口实现异步并行。
* 多设备：描述了跨多个设备编程时的相关接口行为。

#### 3.2.5.1 互斥限制

使用 `shared` 类型时，一个 **Block** 中只能有一个 `thread`（对应一个 **SIP**）执行 L3->L2 或L2->L3 搬运，内存复制。

#### 3.2.5.2 存储系统

TopsCC 编程模型下假设系统由主机端和设备端组成，二者有独立的存储：**主存**和**设备内存**。Kernel 主要在设备内存中工作，主机运行时需要负责设备内存的**分配、释放、拷贝，以及在主存和设备内存间的数据搬运**。

#### 3.2.5.3 设备内存

当前设备内存为**线性内存**，内存句柄中仅包含地址信息，不包含维度解释、切片（tiling）等信息。
当前设备地址空间为设备物理地址，因此和主机地址空间没有统一。设备地址空间的位宽如下所示：

**表 3-9 设备内存**

| 设备地址空间位宽          | i20           | 最大 40bits |

线性内存分配在设备地址空间中，并**映射式**地映射到主机地址空间中。每个分配的内存对象可以在主机端通过指针来引用，主机端的指针被包装为运行时的内存对象句柄。而在设备端 **Kernel** 通过设备地址引用内存对象，其表现形式仍为指针。在主机端启动 Kernel 时，会将主机端指针转换为设备端指针，让工作在两个地址空间中的代码可以协同。

内存对象通常通过 **topsMalloc()** 分配和 **topsFree()** 释放，数据搬运使用 **topsMemcpy** 接口（如前所述目前 **topsMalloc3D** 和 **topsMallocPitch** 类接口均不支持）。下面代码所示：

```cpp
#include <stdio.h>

#include <tops/tops_runtime.h>
#include <tops.h>

__global__ void vec_add(int *from, int *to, size_t N)
{
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);
    __valigned__ int buffer[128];

    tops::mdspan buf(tops::Private, &buffer, 128);

    for (size_t i = 0; i < N; i += 128) {
        tops::mdspan src(tops::Global, from + i, 128);
        tops::mdspan dst(tops::Global, to + i, 128);
        tops::memcpy(ctx, buf, src);

        for (size_t j = 0; j < 128; j += tops::vlength<vint>()) {
            const auto &v = tops::vload<vint>(buffer + j);
            tops::vstore(tops::vadd<vint>(v, v), buffer + j);
        }

        tops::memcpy(ctx, dst, buf);
    }
}

int main(int argc, char *argv[])
{
    int *A_d, *C_d;
    int *A_h, *C_h;
    size_t N = 512;
    size_t Nbytes = N * sizeof(int);

    A_h = (int*)malloc(Nbytes);
    C_h = (int*)malloc(Nbytes);

    // Initialize the data.
    ...

    topsMalloc(&A_d, Nbytes);
    topsMalloc(&C_d, Nbytes);

    topsMemcpy(A_d, A_h, Nbytes, topsMemcpyHostToDevice);

    vec_add<<<1, 1>>>(A_d, C_d, N);

    topsMemcpy(C_h, C_d, Nbytes, topsMemcpyDeviceToHost);

    topsFree(A_d);
    topsFree(C_d);

    free(A_h);
    free(C_h);

    return 0;
}
```

#### 3.2.5.4 访问主存

除设备内存之外，设备也可以访问系统主存，用户需要通过 ``topsMallocHost()`` 分配或 ``topsHostRegister()`` 接口注册分配的系统内存指针。主存同样会被映射到两个地址空间中，并且锁定在物理内存中（**pinned pages**）。

设备端对其访问的性能会较低，但会有如下优点：

* 可以实现设备端发起的异步数据拷贝从而和 ``Kernel`` 的执行并行。
* 映射到设备地址空间后，设备端可以直接访问少量内存拷贝。
* 目前在主存和设备内存之间自动迁移的内存对象尚不支持，即 ``topsMallocManaged()`` 当前不可用。

#### 3.2.5.5 全局变量

主机运行时还可以访问程序中的设备空间全局变量，示例如下：

```cpp
#include <tops/tops_runtime.h>
#include <tops/tops_runtime_api.h>
#include <tops.h>

__device__ int globalIn[256];
__device__ int globalOut[256];

int main(int argc, char *argv[])
{
    int data[256] = {0};
    int* ptr;

    topsMalloc(&ptr, 256 * sizeof(int));

    topsMemcpyFromSymbol(data, globalIn, 256 * sizeof(int));

    topsMemcpy(ptr, data, 256 * sizeof(int), topsMemcpyHostToDevice);

    topsMemcpyToSymbol(globalOut, ptr, 256 * sizeof(int));

    topsFree(ptr);

    return 0;
}
```

``topsGetSymbolAddress()`` 可以获取全局变量的内存句柄，``topsGetSymbolSize()`` 可以获取内存对象的大小。

#### 3.2.5.6 异步并行

TopsRider 提供了一系列 **API**，为各种层级的计算和存储运并行提供支持：

* 在主机端的计算
* 在设备端的计算
* 从主机端向设备搬运数据
* 从设备端向主机搬运数据
* 在指定设备内搬运数据
* 在设备之间搬运数据

上述这些任务可以在不同层面并行。

##### 主机端和设备端并行

主机端通过异步接口将任务下发到设备的队列中，设备执行完毕后会通知主机端（**event**），在此期间设备端可以执行其他任务而不是阻塞等待。设备端支持如下的异步任务：

* 启动内核
* 内存拷贝
* 内存赋值

上述任务同样支持对应的同步任务 API。

#### 3.2.5.7 多核并行执行

在同一个设备上，不同的进程、上下文、线程都可以使用并行下发的方式异步启动内核任务。多个内核使用的资源充足时，它们就会并行调度执行。

##### 数据流与计算并行

数据通道需要从主存搬运数据到设备内存，经由计算后将结果从设备内存搬回主存，这个过程可以通过**输入、计算、输出**三级流水，在内核逻辑里也有类似的流水并行优化，但是主机端运行时不感知，三者也可以并行执行。

##### 数据传输并行

在硬件上主数据搬运的带宽通常大于一个，因此输入、输出和设备内的数据搬运经常可以并行，受限于总线带宽和读写口数量，并发数据传输并不总是会获得更好的性能，部分场景下会有较大收益。

##### Stream 任务流

上述描述的所有并行场景都是通过一种称为 **stream** 的任务流来实现的。**stream** 是一段命令协议包（**command packet**）的序列，命令序列会被设备端按照顺序执行。不同的 stream 中的命令序列的执行顺序则是彼此独立的，可以在多个 stream 之间显式的添加依赖来控制它们执行顺序的关系。同步等待一个 stream 可以保证之前已下发的所有命令全部完成。

**1. 创建和销毁**

stream 的创建包括构造一个任务流对象以及添加任务流中的任务，例如启动内核、主存与设备内存之间的数据拷贝。下面代码例子中创建了两个 stream 对象并分配了一个映射到设备端的主存中的数组。

```cpp
topsStream_t stream[2];

for (int i = 0; i < 2; ++i)
    topsStreamCreate(&stream[i]);

float* hostPtr;
topsMallocHost(&hostPtr, 2 * size);
```

每个stream对象负责一个主存到设备内存的数据搬运、一个启动内核操作、一次设备内存到主存的数据搬运

```cpp
for (int i = 0; i < 2; ++i) {
    topsMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, topsMemcpyHostToDevice, stream[i]);

    MyKernel<<<1, 1, 0, stream[i]>>>(
        outputDevPtr + i * size, inputDevPtr + i * size, size);

    topsMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, topsMemcpyDeviceToHost, stream[i]);
}
```

两个 stream 都会拷贝自己的一段输入数组 **hostPtr** 到设备内存的 **inputDevPtr** 中，然后调用 **MyKernel()** 处理 **inputDevPtr**，再将结果 **outputDevPtr** 从设备内存中拷贝回 **hostPtr** 的主存里。根据设备的能力，两个 stream 交替或同时执行。

用户需要主动销毁 stream 对象。

```cpp
for (int i = 0; i < 2; ++i)
    topsStreamDestroy(stream[i]);
```

为避免用户阻塞正在执行的 stream，**topsStreamDestroy()** 接口会立即返回，但是 stream 对象和关联的资源会在设备端完成 stream 的执行后才会释放。

**2. 默认 stream**

用户调用异步任务接口时通常需要传递 **stream** 参数指定任务流。如果用户不指定或者传递空 **stream** 指针，则任务会被发送到**默认 stream** 上，并且按顺序下发顺序保证顺序执行。每个设备拥有一个默认 stream，所有线程在该设备上共享同一个默认 stream。尚不支持多线程下每个线程拥有独立的默认 stream。

**3. 显式同步**

用户可以主动同步 stream：

* `topsDeviceSynchronize()` 会等待当前所有线程的所有 **stream** 全部执行完成。
* `topsStreamSynchronize()` 接受一个 **stream** 对象作为参数，等待该 stream 对象上的所有任务都完成。
* `topsStreamWaitEvent()` 接受一个 **stream** 和一个 **event** 作为参数，在任务流上构建一个异步等待任务，所有该任务之后下发的任务都会等待 **event** 对应的事件发生后才会继续执行。
* `topsStreamQuery()` 供应用程序查询某个 **stream** 里的任务是否已经完成。

**4. 隐式同步**

不同 **stream** 的命令通常可以并行，暂时没有操作会触发隐含的同步行为。

**5. 并发行为**

多个 stream 上的命令，其并发行为取决于各自命令所在序列顺序，以及设备对各种类型任务支持的最大并发数量。
例如，在设备上如果某个时钟的数据搬运任务的最大并发度是 **1**，那么两个 stream 上的内存拷贝操作将会结构性冒险（**structural hazard**），进而串行执行。运行时未来将提供接口可查询各类型任务当前执行环境下的最大并发度。
两个 stream 上的不同类型的任务可以在设备端并发执行。

**6. 主机端回调**

运行时提供了 `topsStreamAddCallback()` 接口，可以向 stream 中插入一个异步的主机端回调任务，在这个任务之前的所有任务执行完毕后，该任务才会执行。下面例子中 **MyCallback** 函数会在设备内存到主存的数据搬运结束后被执行。

```cpp
void topsStreamCallback_t MyCallback(topsStream_t stream, topsError_t status, void *data) {
    printf("inside callback %d\n", (size_t)data);
}

for (size_t i = 0; i < 2; ++i) {
    topsMemcpyAsync(devPtrIn[i], hostPtr[i], size, topsMemcpyHostToDevice, stream[i]);

    MyKernel<<<1, 1, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i], size);

    topsMemcpyAsync(hostPtr[i], devPtrOut[i], size, topsMemcpyDeviceToHost, stream[i]);

    topsStreamAddCallback(stream[i], MyCallback, (void*)i);
}
```

在主机端回调任务之后下发到 stream 上的任务不会等待回调函数结束后才执行，而是直接顺序执行。因此如果需要同步阻塞等待的场景，需要主机端使用同步接口例如 `topsStreamSynchronize()` 来实现。

**7. Stream 优先级**

当前 **stream 不支持优先级** 调度。

##### 事件

**event** 事件可以用于跟踪设备端异步任务的执行进度，显式同步设备端的多个 **stream**，同步主机端和设备端的任务。事件可以记录在 stream 上；当一个事件完成时，该 stream 上所有处于这个 **event** 前的任务都已经执行完成。默认 **stream** 上的事件发生时，所有 stream 上在这个事件记录之前下发的所有任务都已经执行完成。

**创建和销毁：**

```cpp
topsEvent_t start, stop;

topsEventCreate(&start);
topsEventCreate(&stop);

topsEventDestroy(start);
topsEventDestroy(stop);
```

##### 同步任务调用

有一些任务接口是**同步**的，在设备端将任务执行完之前，接口不会返回。可以通过 `topsSetDeviceFlags()` 接口来控制主机端线程此时是让步（yield）、阻塞或是忙等。

#### 3.2.5.8 统一地址空间

目前 TopsCC 程序尚未实现完整的统一地址空间机制。通过存储管理接口分配的内存对象句柄均为主机端指针，用户程序可以直接对其读写访问。当句柄被传递到接口中使用时，会根据需要将其转换到设备地址空间的指针，用户程序可以直接使用。

```cpp
__global__ void test(int *ptr)
{
    printf("%lx\n", (uint64_t)ptr); // device address pointer
}

int main(int argc, char *argv[])
{
    int *data;

    topsMalloc(&data, sizeof(int));
    *data = 0; // host address pointer

    test<<<1, 1>>>(data);

    topsFree(data);

    return 0;
}
```

```cpp
#include <tops/tops_runtime.h>

__device__ extern void foo();

__global__ void bar() {
    foo();
}

int main() {
    bar<<<1,1>>>();
    return 0;
}
```


### 3.4 编程限制汇总

#### 3.4.1 编程相关

1. **GCU 2.0 不支持全局寻址**。kernel 不能直接访问 `__shared__` 和 `__device__` 地址空间。需要通过 **DTE** 将 `__shared__` 和 `__device__` 地址的内容搬运至 **L1** 进行计算。
2. 关于打印功能，在 kernel function 中 **printf** 只支持打印**整型**，不支持 **%s** 和 **%p**。关于地址打印，可以将指针对强制转换成 **long long** 类型。
3. 在 **GCU210** 上 **Block dims** 的乘积最大为 **12**，即一个 Block 内**最多开 12 个 Thread**。
4. 在 **GCU210** 上 **Grid dims** 的乘积最大为 **2**。
5. **GCU 2.0 不允许直接访问 shared memory（L2）**，只能用于 **DTE** 数据搬运操作。动态大小的 shared memory，每个 `__global__` 函数只能使用一个。
6. 有关硬件支持 shared memory 的大小限制，**i20** 上每个 Block **最大 shared memory 24MB**。

### 3.4.2 DTE 数据搬运

1. 用 `__shared_dte__` 和 `__private_dte__` 声明的 **DTE Context** 只能支持 **Global** 和 **Shared** 之间的数据传输。不加任何修饰符声明的 **DTE Context** 是 **Thread** 私有的 DTE 上下文，支持 **Global、Shared 和 Private** 之间的数据传输。

### 3.5 性能调优指南

尽量使用 **DTE 软件流水** 使得**数据搬运**和**计算**可以并行。
向量化要点”：
vload/vstore 仅在数组 __valigned__ 且访问起始地址、长度均为 128B 对齐时使用；否则走标量尾部。
tops::vlength<T>() 获取向量宽；向量路径 + “标量尾”双路径，保证边界一致性。
参与向量操作的 L1 缓冲必须 __valigned__。

---

# 4 FAQ

## 4.1 使用报错信息

**Q：** 使用 `mdspan` 时地址空间类型设置错误，可能会引发程序 *hang*。
**A：** 检查地址空间类型设置，**L1 内存**地址设置为 `tops::Private`，**L2 内存**地址设置为 `tops::Shared`，**L3 内存**地址设置为 `tops::Global`。
示例程序如下：

```cpp
__global__ void foo(int *arr, int size) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);

    int buf[size];

    tops::mdspan L3(tops::Private, arr, size); // L3 should be set to tops::Global but wrongly set to tops::Private
    tops::mdspan L1(tops::Global, buf, size);  // L1 should be set to tops::Private but wrongly set to tops::Global

    tops::memcpy(ctx, L1, L3);                 // at this point the program may hang
}
```

---

**Q：** DTE 未经初始化直接使用，可能会引发程序 *hang*。
**A：** 使用 `tops::dte_scope` 或者显式调用 `init` 函数，其中 `tops::dte_scope` 会自动完成 DTE 的初始化操作以及销毁操作；如果显式调用 `init` 函数，在 DTE 使用完成之后**注意调用** `destroy` 函数释放 DTE 资源。
示例程序如下：

```cpp
__global__ void foo() {
    int a[32];
    int b[32];

    tops::mdspan src(tops::Private, a, 32);
    tops::mdspan dst(tops::Private, b, 32);

    tops_dte_ctx_t ctx;

    tops::dte_scope s(ctx);       // use tops::dte_scope to initialize dte context
    // ctx.init();                // or use init() to initialize dte context

    tops::memcpy(ctx, dst, src);  // if dte is uninitialized, the program may hang or abort

    // ctx.destroy();             // if use init(), remember to use destroy() to free dte context
}
```
注意DTE 事件等待与地址复用不严，写回被覆盖或遗漏 → 输出出现 0

**GCU210（i20）**，忽略 GCU200。并修正术语对齐到 v2（例如对齐修饰符统一为 `__valigned__`）。

## 0. 适用范围与术语对齐
- **芯片/平台**：GCU210（i20）。
- **地址空间修饰**（与 v2 保持一致）：
  - `tops::Global` ↔ L3（设备全局存储）
  - `tops::Shared` ↔ L2（簇共享存储，**仅 DTE 通道**）
  - `tops::Private` ↔ L1（SIP 私有存储，**可计算**）
- **对齐修饰**：统一使用 `__valigned__`（v1 中出现的 `__valigned__` 系误写，按 v2 规范更正为 `__valigned__`）。
---

## 1. v2 未覆盖/强调不够的 **高层算子框架 API**
> 这些 API 在 v1 被反复提及，但 v2 未系统收录；可显著简化典型“搬运→计算→写回”的模板代码。如 SDK 不含这些头，请忽略本节或改用 3.2.4 的手写模板。

### 1.1 Elementwise（逐元素）框架
- **场景**：在 L3 张量上执行逐元素函数（自动以 tile 方式搬运至 L1，再执行）。
- **核心接口**（设备端）：
  ```cpp
  #include <tops/elemwise.h>

  // 在 Kernel 内部直接调用
  tops::elemwise_kernel(
      [] __device__(auto &out, auto &in) {
          out = in * in;  // 在 L1 上的逐元素操作
      },
      N,                          // 总元素个数
      tops::Input(0), in_ptr,     // 输入
      tops::Output(0), out_ptr    // 输出
  );
  ```
- **变体/控制**：`elemwise_tiles`（按 tile 粒度自定义）、`elemwise_local`（已在 L1 的缓冲上直接算）。
- **优势**：自动封装 DTE 切片/回写与对齐处理，减少手写样板代码。

### 1.2 Reduction（归约）框架
- **场景**：对张量做加/最大/最小等归约（可从 L3 直接发起或在 L1 上本地归约）。
- **核心接口**（设备端，示例）：
  ```cpp
  #include <tops/reduction.h>

  // kernel 级：从 L3 发起（内部自动搬运）
  tops::reduction_kernel(
      [] __device__(auto &acc, auto &x) {
          acc = __reduction_add(acc, x);
      },
      out_ptr, out_shape,          // 归约输出
      in_ptr,  in_shape,           // 归约输入
      /*identity*/ 0               // 加法幺元
  );

  // local 级：对已在 L1 的缓冲做归约
  tops::reduction_local(
      [] __device__(auto &acc, auto &x) {
          acc = __reduction_max(acc, x);
      },
      out_L1, in_L1
  );
  ```
- **内置运算符**：`__reduction_add / __reduction_max / __reduction_min`。
- **注意**：默认归约维度常为“中间维”，需与 `in_shape/out_shape` 对齐。

### 1.3 Select / Broadcast 等辅助算子
- **条件选择**：
  ```cpp
  #include <tops/select.h>
  tops::select_kernel(
      [] __device__(auto &o, auto &lhs, auto &rhs, auto &cond) {
          o = cond ? lhs : rhs;
      },
      size, tops::Input(0), lhs, tops::Input(1), rhs, tops::Input(2), cond,
      tops::Output(0), out
  );
  ```
- **按维广播**：
  ```cpp
  #include <tops/broadcast.h>
  tops::broadcast_in_dim(out, in, dim0, dim1, /*broadcast_dim*/1, /*bsize*/k);
  ```

> 这些高层 API 有助于**标准化**常见套路；v2 可在“3.2.4 设备端编程”后追加“高层封装”小节引入。

---

## 2. DTE 进阶：链式/异步流水的细节补全
v2 已介绍同步/异步与软件流水；v1 另强调了“**多 DTE 上下文链式**”与若干 **易错点**：

### 2.1 dte_chain（多上下文串联）
```cpp
// 伪头文件名，实际以你环境为准：
#include <tops/dte_chain.h>

tops_dte_ctx_t ctxA, ctxB;
ctxA.init(); ctxB.init();

auto chain = tops::dte_chain(ctxA, ctxB);
chain.connect(...);               // 配置 A→B 的数据流
chain.trigger();                  // 触发
chain.wait();                     // 等待完成

ctxB.destroy(); ctxA.destroy();
```
> 适合 **L3→L1→L3** 的双向搬运在不同 ctx 上交错、做更深流水。若你当前环境无 `dte_chain` 头（SDK 版本差异），可用**手动双 ctx + event** 等价实现（v2 已给出）。

### 2.2 `slice_async` 签名易错
- **正确**：必须给 **offset**（至少 4 参起），例如：
  ```cpp
  auto ev = tops::slice_async(ctx, dst_md, src_md, /*offset*/ {x0, y0, z0});
  tops::wait(ev);
  ```
- **错误**：`slice_async(ctx, dst, src)`（少 offset）会编译/链接失败。

### 2.3 `_async` 返回 `tops::event`
- 只有带 `_async` 的接口返回 `tops::event`；**无后缀**版本为 `void`。
- 典型易错：
  ```cpp
  // 错误：同步接口赋给 event
  tops::event ev = tops::transpose_deslice(...); // ❌ 返回 void
  // 正确：
  tops::event ev = tops::transpose_deslice_async(...);
  tops::wait(ev);
  ```

### 2.4 开发期健壮性
- 建议全程开启：`-DTOPS_ENABLE_DTE_CHECK`（越界/地址空间不一致更早暴露）。
- **地址空间配置**一旦写错（如把 L3 标成 `tops::Private`），现象多为 **hang** 而非报错。

---

## 3. 向量工具与类型映射（补充）
v2 列了大量向量算子，但 **类型映射/广播**在 v1 有更集中提示：

- **固定向量宽**：128B；`vfloat`=32×`float`，`vhalf`=64×`tops::half`，`vbfloat`=64×`tops::bfloat`，…
- **从标量到向量类型**：
  ```cpp
  // 由标量类型 T 查到对应的 vector 类型：
  using V = typename tops::scalar2vector<float>::type; // -> vfloat
  ```
- **标量广播到向量**：
  ```cpp
  auto v = tops::vbroadcast(3.14f);      // vfloat
  auto h = tops::vbroadcast(tops::half(1));
  ```
- **对齐与边界**：仅在**完全对齐且长度是 vlength<T> 的倍数**时用 `vload/vstore`，否则走标量尾。

---

## 4. 调试/运行时补遗（v1 独有要点）
- **printf/assert 导致同步**：kernel 内使用 `printf/assert` 会让 runtime 进入**调试同步模式**（同一 stream 强制同步），用于排错可以，但性能测试前务必移除。v2 有提及，但建议在“性能调优”再次**加粗提醒**。
- **不存在 API 的“想当然命名”**：
  - 如 `tops::reduce_add / tops::vreduce_add` **不存在**；应使用前述 **reduction** 框架或自己展开。
- **评测/比赛环境可能禁用**某些 API：如把 `topsMalloc/topsFree` 宏重定向为 `_topsMalloc_disabled`。**算子实现不要私自设备端分配临时 L3**，尽量用 L1 缓冲或由上层传入。

---

## 5. 典型模板（v1 风格，按 v2 术语修正）

### 5.1 元素算子（完整可嵌入）
```cpp
#include <tops/elemwise.h>

__global__ void square_kernel(float *out, const float *in, int N) {
    // 自动按 tile 从 L3 → L1，L1 上逐元素操作，再写回
    tops::elemwise_kernel(
        [] __device__(auto &o, auto &x) {
            o = x * x;
        },
        N,
        tops::Input(0),  in,
        tops::Output(0), out
    );
}
```

### 5.2 手写 Tile + SIMD（与 v2 一致但给出“边界回退”套路）
```cpp
#include <tops/tops_runtime.h>
#include <tops.h>

__global__ void vec_add(float *a, float *b, float *c, int N) {
    tops_dte_ctx_t ctx;
    tops::dte_scope s(ctx);

    constexpr int TILE = 128;                         // 对齐 128B
    __valigned__ float buf_a[TILE], buf_b[TILE], buf_c[TILE];

    tops::mdspan A_L1(tops::Private, buf_a, TILE);
    tops::mdspan B_L1(tops::Private, buf_b, TILE);
    tops::mdspan C_L1(tops::Private, buf_c, TILE);

    for (int i = 0; i < N; i += TILE) {
        int n = min(TILE, N - i);

        tops::memcpy(ctx, A_L1, tops::mdspan(tops::Global, a + i, n));
        tops::memcpy(ctx, B_L1, tops::mdspan(tops::Global, b + i, n));

        int j = 0;
        // 向量快路径
        for (; j + tops::vlength<vfloat>() <= n; j += tops::vlength<vfloat>()) {
            auto va = tops::vload<vfloat>(buf_a + j);
            auto vb = tops::vload<vfloat>(buf_b + j);
            auto vc = tops::vadd(va, vb);
            tops::vstore(vc, buf_c + j);
        }
        // 边界标量路径
        for (; j < n; ++j) buf_c[j] = buf_a[j] + buf_b[j];

        tops::memcpy(ctx, tops::mdspan(tops::Global, c + i, n), C_L1);
    }
}
```

### 5.3 归约（加和）示例
```cpp
#include <tops/reduction.h>

__global__ void sum_kernel(float *out, const float *in, int N) {
    // 对 1D 数组做加和，identity=0
    tops::reduction_kernel(
        [] __device__(auto &acc, auto &x) { acc = __reduction_add(acc, x); },
        out, /*out_shape*/ N,
        in,  /*in_shape*/  N,
        0
    );
}
```

### 5.4 GEMM 的“稳妥输出路径”提示
- 若使用 `tops::transpose_deslice_async` 复杂组合在大规模/动态 tile 下出现偶发误差，**保守做法**：
  1) 在 L1 计算出 `C`；
  2) 需要转置的场景，**手工在 L1 做转置**到 `C_T`；
  3) 用普通 `deslice` 写回 L3。  
  实测该路径最稳，代价是多一次 L1 遍历。

---

## 6. GCU210 资源/并发提醒（与 v2 对齐但再次明确）
- **Block 线程总数 ≤ 12**（`blockDim.x * blockDim.y * blockDim.z ≤ 12`）。
- **Cooperative Kernel**：grid 总块数 ≤ `multiProcessorCount`（i20 为 **2**）。
- **向量化前提**：所有参与 `vload/vstore` 的 L1 缓冲必须 `__valigned__`，且访问地址/长度满足 128B 自然对齐与 `vlength<T>` 整倍数。

---

## 7. MLP/激活实现的“一致性与稳定性”建议（补充条）
> 与 v2 的一般指南不冲突，这里收拢为 **可复制到算子说明** 的检查清单：

- **公式一致化**：向量与标量路径使用**同一**数学表达式（如 SiLU 用 `x/(1+exp(-x))`，向量用 `vexp/vdiv`，标量尾用 `exp/`相同公式）。
- **边界一致性**：仅“完全对齐”时走向量路径；其余统一走**标量尾**，避免出现微妙的数值分歧。
- **累加稳态化**：长链路加法用 **Kahan** 补偿或配对求和，固定加法顺序；批量/并行度变化后需重跑精度回归。
- **可切换模式**：保留 `--precise / --fast` 两种路径开关，便于在评测与上线间切换。

---

## 8. 常见陷阱对照表（v1 特有案例）
| 症状 | 可能根因 | 解决方案 |
|---|---|---|
| kernel 无响应（hang） | DTE 未 init；mdspan 地址空间写错；对齐不足的 vload | 用 `tops::dte_scope`；开启 `-DTOPS_ENABLE_DTE_CHECK`；边界走标量 |
| “把 void 当 event” 编译错 | 使用了同步 API 却当作 `_async` 用 | 仅 `_async` 返回 `tops::event`；用 `tops::wait` 同步 |
| `slice_async` 参数不匹配 | 遗漏 `offset` | 使用 `slice_async(ctx, dst, src, {offset...})` |
| 运行慢/卡住 | kernel 内 `printf/assert` | 调试阶段可用，性能测试前务必移除 |
| 链接/编译异常 | 使用被评测环境禁用的分配 API | 不在设备端私自分配 L3；通过 L1 缓冲或调用方传入 |

---

## 9. 集成方式建议（如何合入 v2）
- 在 **3.2.4 设备端编程** 后增设小节 **“高层算子封装（Elementwise / Reduction / Select / Broadcast）”**。
- 在 **DTE 软件流水** 小节追加 **dte_chain/多 ctx 提示** 与 `_async`/`void` 返回值区别示例。
- 在 **性能与正确性** 小节集中强调：`__valigned__`、边界标量尾、一致公式、`-DTOPS_ENABLE_DTE_CHECK`。

---


### 1) 内建编译期宏（**新**）
- `__TOPS_DEVICE_COMPILE__`：设备端编译时定义（便于区分 host/device 代码路径）。  
- `__GCU_ARCH__`：三位数字的架构码：`S60=300, T20=200, i20=210`。  
  典型用法（按架构走不同实现）：
```cpp
#if defined(__TOPS_DEVICE_COMPILE__)
  #if __GCU_ARCH__ >= 210
    // i20 (GCU210) 专用路径
  #else
    // 其他架构路径
  #endif
#endif
```

### 2) 设备/并发属性查询与 cooperative 约束（**更细化**）
- `topsGetDeviceProperties` 可取：  
  - `multiProcessorCount`：**GCU210 = 2**  
  - `maxThreadsPerMultiProcessor`：**GCU210 = 12**（即 Block 维度乘积最大 12）
- cooperative kernel：**GCU210 上 Grid 维度乘积 ≤ 2**；且 Grid 总大小 ≤ `multiProcessorCount`。
```cpp
topsDeviceProp_t prop; int dev=0;
topsGetDeviceProperties(&prop, dev);
// 验证 GCU210 限制：
assert(prop.multiProcessorCount == 2);
dim3 block(prop.maxThreadsPerMultiProcessor, 1, 1); // 12
// cooperative 启动前自检：
auto gridProd = 2u; // 例如 dim3(2,1,1)
assert(gridProd <= 2 && gridProd <= (unsigned)prop.multiProcessorCount);
```

### 3) 存储/寻址与容量（**细化 i20 数据**）
- **GCU 2.0 不支持全局寻址**：L2/L3 仅能经 DTE 搬运；计算只能直接访问 L1（私有）与 `__constant__`。  
- **共享内存上限**：**i20 = 24 MB**（动态 shared 每个 `__global__` 仅 1 个声明）。  
- **设备内存地址宽度**：i20 **≤ 40 bits**。  
- **Host 可用内存**：i20 机型示例：**64 GB**（运行库说明）。  

### 4) Host 可见内存（pinned）与直访（**新**）
- 通过 `topsMallocHost` 或 `topsHostRegister` 分配/注册 **锁页主存**，可映射到设备地址空间，被设备端直接访问或用于异步拷贝（带宽/延迟逊于设备内存；慎用 write‑combining）。
```cpp
float* hptr = nullptr;
topsMallocHost(&hptr, N * sizeof(float));   // pinned host mem
// … 填充 hptr …
topsMemcpyAsync(devPtr, hptr, N*sizeof(float), topsMemcpyHostToDevice, stream);
// kernel 也可直访 hptr（性能较低，场景化使用）
```

### 5) 设备端全局变量符号访问（**新**）
```cpp
__device__ int gIn[256];
__device__ int gOut[256];

// Host:
int h[256] = {0};
topsMemcpyFromSymbol(h, gIn, sizeof(h));    // 读设备端全局
int* dtmp; topsMalloc(&dtmp, sizeof(h));
topsMemcpy(dtmp, h, sizeof(h), topsMemcpyHostToDevice);
topsMemcpyToSymbol(gOut, dtmp, sizeof(h));  // 写设备端全局
topsFree(dtmp);
```

### 7) Stream / Event / 回调（**新**）
- **默认 stream**：每设备共用一个（当前未区分线程）；不传入显式 stream 参数则落到默认 stream。  
- **显式同步**：`topsDeviceSynchronize()`、`topsStreamSynchronize(s)`；事件依赖：`topsStreamWaitEvent(s, e)`。  
- **回调**：可在 stream 完成前序任务后触发主机回调（不会阻塞后续命令排队）。
```cpp
topsStream_t s[2]; for (int i=0;i<2;++i) topsStreamCreate(&s[i]);
// … H2D → Kernel → D2H 的三段流水，各自在 s[i] 上 …
void MyCb(topsStream_t, topsError_t, void* tag){ printf("cb %ld\n",(long)tag); }
topsStreamAddCallback(s[0], MyCb, (void*)0L);
// 销毁：返回即刻，但资源在设备完成该 stream 后回收
for (int i=0;i<2;++i) topsStreamDestroy(s[i]);
```
### 9) DTE 接口补全（**更丰富**）
- **全量配置**：`config_transpose / config_slice_transpose / config_pad / config_mirror_tb / config_mirror_lr / config_broadcast …`  
- **增量配置**：`set_*_addr / set_*_offset / set_*_dim_size / set_total_size / set_transpose_layout / set_pad_config`  
- **触发**：`trigger()` 返回 `tops::event`，`trigger_and_wait()` 同步；还有 `_async` 族的复合接口：
```cpp
tops_dte_ctx_t ctx; ctx.init();
auto ev = tops::memcpy_async(ctx, dst, src);
tops::wait(ev);
ctx.destroy();
```
- **软件流水模板（双缓冲）**：v3 提供了计算‑搬运并行化的示例骨架，可直接套入 tile‑based 算子（GCU210 适用）。

### 10) 同步原语（**小补**）
- `__syncthreads()`：Block 内同步。  
- `__syncblocks()`：Grid 级同步（Kernel 必须 `__cooperative__`）。


### 12) 编程限制 / FAQ（**新增可操作排错点**）
- **限制汇总（GCU210 相关）**：
  - Block 维度乘积 ≤ **12**；cooperative Grid 维度乘积 ≤ **2**；GridMax：x=65536, y=256, z=256。  
  - 动态 shared 每 kernel 仅 1 个；i20 shared 上限 **24MB**。  
  - **使用 `shared dte` 时，每 Block 只有 1 个 Thread 能执行 L3↔L2 拷贝**。  
- **常见挂起原因**：
  1) `mdspan` 地址空间设置错误（把 L3 标成 `tops::Private` 等）：
```cpp
// 错误示例：arr 是 L3 指针，却被错误地标成 Private
tops::mdspan L3_wrong(tops::Private, arr, size);
tops::mdspan L1_wrong(tops::Global,  buf, size);
tops::memcpy(ctx, L1_wrong, L3_wrong); // 可能 hang
```
  2) **DTE 未初始化** 就调用 `tops::*`：用 `tops::dte_scope` 或 `ctx.init()/destroy()` 包裹：
```cpp
tops_dte_ctx_t ctx;
tops::dte_scope scope(ctx);     // 自动 init/destroy
// ctx.init(); … tops::memcpy(ctx, …); ctx.destroy();
```
避免把运行时函数当作编译期常量，并在使用模板函数时显式指定模板参数。
注意：凡是 L3/L2 的读写，一律 DTE；只有 L1 的私有缓冲可以随便下标读写。
尾块也必须通过 L1→DTE 写回
要点速记（别再踩坑版）：
绝不直访 L3/L2：GCU2.x kernel 只能直接访问 L1（私有）与 __constant__；__device__/__shared__ 都要经 DTE 先搬到 L1。
向量化需要“双重对齐”：不仅数组要 __valigned__，访问起始下标也要向量宽对齐。dilation/stride/padding 下很难保证，因此先用标量路径保正确，再做“对齐块才向量化”的优化。
PyTorch 的 Conv2d 是交叉相关（不翻核）。不确定时留一个“翻核开关”宏能一键切换。
DTE 先 init 再用，推荐 tops::dte_scope；调试期打开 -DTOPS_ENABLE_DTE_CHECK。
printf/assert 会强制同步（同一 stream 串行），性能测试前务必移除。
异步/流水：读行→计算→写回可以做双缓冲/事件同步，但每一步都必须通过 DTE。
SiLU 里用到的异步策略（能学的点）
输入异步 + 双缓冲：memcpy_async(Global→Private) 到 buf0/buf1，用 ev_ld[2] 管理；计算前 wait(ev_ld[cur]) 再读缓冲，避免直访 Global。
输出也用了异步：memcpy_async(Private→Global) 写回，用 ev_st[2] 做逐缓冲的“写回事件”，在重用同一缓冲做下一次加载前，先 wait(ev_st[cur])，防止还没写完就被新一轮读取覆盖。
事件上限明确：每线程最多同时挂 3 个事件（当前 load 完待算 + 下一 tile 的另一个 load + 当前 store）。代码里在复用前必 wait，保证事件不累积。
小任务走同步快路：span <= TILE_TINY 直接 同步搬入 → 向量化计算 → 同步写回，规避小规模下的事件/上下文开销。
线程分片均衡：把总量平均分给所有线程（前 rem 个多 1），避免最后几个线程很忙，其他线程都在等，提升并行度。
向量化细节完善：VLEN = vlength<vfloat>()，8×/4×/1×展开 + 标量尾部；Sigmoid 走 vsigmoid，并保留 __valigned__ 缓冲对齐，保证 vload/vstore 性能。
严格的“先 wait 再用”纪律：任何一次读缓冲或重用缓冲前，一定先 wait 对应事件。
双缓冲“计算-预取”流水线：当前 tile 计算，同时后台把下一 tile 异步搬进另一组缓冲。
全部都用gcu的 __valigned__ 不要用alignas(128)，取消 blocks ≤ 2 的约束（也不再 __cooperative__）
使用函数尽量对这我给的表来做防止函数不存在，记住数组不能赋值,要用指针，有时候需要把标量广播成向量再跟向量去相乘
