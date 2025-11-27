# ImageSteganography-LSB-DCT

本项目旨在提供能写入大量信息的**空域 (LSB) 隐写**以及高鲁棒性的**频域 (FFT/DCT) 隐写**方法。

## 核心功能

### 1. 空域隐写
*   **安全加密**: 使用 ChaCha20-Poly1305 进行数据加密和完整性认证。
*   **物理打散**: 基于密码哈希生成随机像素映射，数据在物理层面上是打散的，抗统计分析。
*   **多文件支持**: 支持在同一张图片中存储多个互不干扰的文件或文本。
*   **性能优化**: 使用 Numba JIT 编译核心读写逻辑。

### 2. 频域隐写
*   **高鲁棒性**: 能够抵抗裁剪、旋转、缩放、涂鸦、压缩、噪声等多种攻击。
*   **盲提取**: 提取水印时不需要原图。
*   **GPU 加速**: 使用 PyTorch 实现 GPU 加速的盲检测算法，极大提高了旋转攻击后的提取效率。
*   **混合算法**: 结合 FFT (同步信标) 和 DCT (数据嵌入)，利用 RS 纠错码保证数据完整性。

## 快速开始

运行主程序：

```bash
python main.py
```

### 菜单选项说明

*   **[A] 空域 - 安全文件系统模式**
    *   **(1) 添加信息**: 将文本或文件隐藏到图片中。支持设置独立的提取密钥。
    *   **(2) 查找信息**: 输入图片路径和密钥，提取隐藏的内容。
*   **[B] 频域 - 盲隐写模式**
    *   **(3) 嵌入水印**: 输入文本或 URL，将其嵌入到图片频域中。
    *   **(4) 提取水印**: 即使图片经过旋转或截图，尝试从中提取水印内容。
*   **[C] 压力测试**
    *   **(5) 压力测试 (LSB)**: 批量写入读取测试，验证隐写的容量和稳定性。
    *   **(6) 压力测试 (FFT)**: 对载体图像进行攻击测试，验证隐写的抗攻击性。

## 核心算法深度解析

### 1. 空域LSB：
本模式在像素层面上实现了一个**离散分布、加密的文件系统**，而非简单的顺序比特写入。

*   **物理层 (散射存储):**
    *   **密钥派生 (KDF):** 使用用户密码通过 PBKDF2 算法生成种子。
    *   **随机映射:** 利用该种子驱动伪随机数生成器 (PRNG)，生成全图所有像素坐标 (x, y) 的乱序索引表。
    *   **数据打散:** 数据被写入这些随机坐标中。如果没有正确的密码，数据在统计上表现为均匀分布的白噪声，无法通过视觉或简单的统计分析定位。
*   **逻辑层 (链表结构):**
    *   **主控块:** 隐写系统的入口点，包含指向第一个数据块的加密指针。
    *   **数据块:** 每个块包含 `[块头 | 加密载荷 | 认证标签 | 下一块指针]`。
    *   这种链表结构允许文件非连续存储，自动处理碎片化空间，支持在同一张图中存入多个互不干扰的文件。

### 2. 频域FFT/DCT：
本模式专为**高鲁棒性**设计，用于抵抗几何攻击和信号处理攻击。提取过程无需原图（盲检测）。

#### 数据包协议结构
水印被封装为**数据包**以确保完整性。嵌入平面被划分为 **36x36 的图块 (Tile)**。每个图块拥有 **1296 bits (162 Bytes)** 的原始物理容量。

**单个图块内部结构表：**

| 字段 | 组件 | 大小 (Bytes) | 说明 |
| :--- | :--- | :--- | :--- |
| **1. 包头** | Magic(4B) + Salt(8B) + Mode(1B) + Len(1B) | 14 Bytes | 基础元数据、解密盐值、数据模式标识。 |
| **2. 包头纠错**| Reed-Solomon Parity | 13 Bytes | 专门保护包头的纠错码，确保元数据可读。 |
| **3. 数据体** | Encrypted Payload | ~100 Bytes | 用户实际数据（经 zlib 压缩 + ChaCha20 加密）。 |
| **4. 数据体纠错** | Reed-Solomon Parity | ~35 Bytes | 填充剩余空间，用于修复受损的用户数据。 |
| **总计** | | **162 Bytes** | 刚好填满一个 36x36 的 DCT 变换块区域。 |

*   **冗余投票机制:** 如果图片分辨率大于图块大小，图块会在全图中平铺重复。提取时，算法会收集所有可见图块的解码结果，进行**多数投票**。
*   **几何同步信标 (FFT Sync):** 算法在**傅里叶频谱 (FFT)** 的特定半径和角度注入高能量的“信标点”。解码器通过搜索这些信标来计算并修正旋转角度。

## 理论数据容量

| 模式 | 容量计算公式 | 1920x1080分辨率 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **LSB** | (宽 x 高 x 3) / 16 字节 | **~380 KB** | 存储机密文件、长文本、源代码。容量随分辨率线性增加。 |
| **FFT** | 固定载荷 (受限于 36x36 图块) | **~100 - 120 字节** | 版权声明、短链接、ID 标识。分辨率增加只提高鲁棒性，不增加容量。图块的大小基于对容量和鲁棒性的取舍。 |

## 鲁棒性测试

*   测试文本: "Google LLC is an American technology corporation."
*   基于每项攻击 50 次测试的结果。

下表展示了算法在不同分辨率下抵抗攻击的能力差异。可以看出，**分辨率越高，冗余度越高，抗破坏性攻击（如剪裁、涂鸦）的能力越强**。

| 攻击类型/分辨率 | 示例图片 | 1920x1080 | 1500x750 | 1000x500 | 320x320 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **涂鸦攻击** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Scribble.png" width="220px"/> | 100.0% | 100.0% | 100.0% | 0.0% |
| **裁剪 (保留40%面积)** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Crop.png" width="220px"/> | 100.0% | 100.0% | 0.0% | 0.0% |
| **旋转 (90°)** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Rotate90.png" width="220px"/> | 100.0% | 100.0% | 100.0% | 100.0% |
| **旋转 (180°)** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Rotate180.png" width="220px"/> | 100.0% | 98.0% | 98.0% | 100.0% |
| **遮挡 (20个块)** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Occlusion.png" width="220px"/> | 100.0% | 100.0% | 100.0% | 0.0% |
| **JPEG压缩 (质量50)** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_JPEG.jpg" width="220px"/> | 100.0% | 100.0% | 100.0% | 100.0% |
| **椒盐噪声** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Pepper.png" width="220px"/> | 100.0% | 100.0% | 88.0% | 0.0% |
| **高斯噪声** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Noise.png" width="220px"/> | 100.0% | 100.0% | 100.0% | 98.0% |
| **模糊 (半径1.5)** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Blur.png" width="220px"/> | 100.0% | 100.0% | 90.0% | 54.0% |
| **亮度 (+50%)** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Brightness.png" width="220px"/> | 82.0% | 80.0% | 76.0% | 64.0% |
| **对比度 (+50%)** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Contrast.png" width="220px"/> | 100.0% | 100.0% | 98.0% | 98.0% |
| **缩放 (50%)** | <img src="https://github.com/Alex-Lee-2190/ImageSteganography-LSB-DCT/raw/main/images/Stego_Scaling.png" width="220px"/> | 100.0% | 100.0% | 100.0% | 80.0% |
| **综合得分** | — | **98.5%** | **98.2%** | **87.5%** | **57.8%** |

## License

MIT License. See [LICENSE](LICENSE) file.
