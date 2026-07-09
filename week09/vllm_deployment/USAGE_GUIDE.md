# USAGE_GUIDE.md — 代码调用与测试指南

本文档覆盖从环境搭建到每个演示脚本的完整执行流程。所有命令假定你已经按照 `ARCHITECTURE.md` 中的环境建议在 WSL2 Ubuntu 22.04 中完成了 vLLM 的安装。

---

## 一、环境准备

### 1.1 WSL2 + Ubuntu 22.04 安装（只需一次）

```powershell
# Windows 管理员 PowerShell
wsl --install -d Ubuntu-22.04
# 重启 → 再跑一次 wsl --install -d Ubuntu-22.04 → 建账号
```

### 1.2 Ubuntu 内的依赖

```bash
# 切清华 apt 源（可选，国内大幅加速）
sudo sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list
sudo apt update
sudo apt install -y python3-pip python3-venv build-essential git curl wget

# 建立虚拟环境（放在 ~/vllm_env/）
python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate

# 配置 pip 清华源
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

# 安装依赖
cd /mnt/d/badou/项目材料准备/vllm_deployment
pip install -r requirements.txt
```

### 1.3 关键兼容性说明（CUDA / 驱动版本）

| 组件 | 版本 | 原因 |
|------|------|------|
| NVIDIA 驱动 | 566.x（CUDA 12.7 兼容） | Windows 侧，WSL2 自动桥接 |
| vLLM | **0.9.2** | 0.20+ 要 CUDA 13（需驱动 580+）不兼容 |
| torch | **2.7.0+cu126** | 与 vLLM 0.9.2 匹配 |
| transformers | **4.52.4** | vLLM 0.9.2 不兼容 transformers 5.x |

如果 `torch.cuda.is_available()` 返回 `False`，99% 是 vLLM/torch 版本选了 CUDA 13（需驱动 580+），降级到上表版本即可。

### 1.4 验证环境就绪

```bash
source ~/vllm_env/bin/activate
python -c "import vllm, torch; print('vLLM:', vllm.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

预期输出：
```
vLLM: 0.9.2
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## 二、启动 vLLM Server

所有 `demo_*.py` 脚本都通过 OpenAI 兼容 API 调用 server，必须先启动它。

### 2.1 启动

```bash
cd /mnt/d/badou/项目材料准备/vllm_deployment/src
bash start_server.sh
```

启动过程（约 15~20 秒）：
1. 加载 Qwen2-0.5B-Instruct 权重（~1GB，从 D 盘读取）
2. 初始化 KV cache（占显存 ~2.5GB）
3. 注册 FastAPI 路由
4. 监听 `0.0.0.0:8000`

看到 `Application startup complete` 即可。

### 2.2 验证可用

新开一个终端：

```bash
# 查询已加载模型
curl http://localhost:8000/v1/models

# 简单对话
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2-0.5b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'
```

### 2.3 停止 server

```bash
# 方式 1：Ctrl+C 终止启动它的终端
# 方式 2：按端口杀进程
fuser -k 8000/tcp
```

---

## 三、各脚本使用方法

所有脚本位于 `src/`，运行前确保已激活 venv 且 server 已启动（`bench_throughput.py` 除外）。

### 3.1 demo_guided_choice.py — 枚举约束

```bash
cd src/
python demo_guided_choice.py
```

**场景**：金融问答意图路由（查股价 / 查财报 / 查新闻 / 对比分析 / 其他）

**内部流程**：
1. 对每个测试问题，调用 server 两次：裸 prompt + `extra_body={"guided_choice": ...}`
2. 对比两种模式的输出合法率和分类准确率

**预期输出**（关键行）：
```
输出合法（在枚举内）   10/12 (83%)    12/12 (100%)
预测正确            3/12 (25%)     3/12 (25%)
```

**教学要点**：guided_choice 100% 保证输出合法，分类正确率决定于模型本身能力。

---

### 3.2 demo_guided_regex.py — 正则约束

```bash
python demo_guided_regex.py
```

**场景**：日期标准化（→ YYYY-MM-DD）、股票代码抽取（→ 6 位数字）

**教学要点**：凡下游有严格解析器的字段，正则约束能把"模型说对但格式错"的问题一次根治。

---

### 3.3 demo_guided_json.py — JSON Schema 基础

```bash
python demo_guided_json.py
```

**场景**：财报问答意图抽取（公司/年度/指标三元组）

**三种模式对比**：
- 裸 prompt：靠指令和 few-shot
- `response_format={"type": "json_object"}`：OpenAI 标准，保证是 JSON
- `guided_json=schema`：vLLM 扩展，保证完全符合 Schema

**关键看点**："22 年" 这类输入下，裸 prompt 和 response_format 可能输出 `year: 22`（违反 `minimum: 2015`），只有 guided_json 能强制修正为 2022。

---

### 3.4 demo_response_format.py — OpenAI 标准方式

```bash
python demo_response_format.py
```

**场景**：新闻情感分类 + 置信度 + 关键词

**教学要点**：`response_format={"type": "json_object"}` 是 OpenAI/Azure/vLLM 都兼容的**可移植方案**。相比 `guided_json` 它跨厂商可用但约束更弱。选型时权衡：
- 跨厂商部署 → response_format
- 单一 vLLM 部署 + 严格解析 → guided_json

---

### 3.5 demo_function_call.py ★ 核心

```bash
# 跑两个工具共 100 个用例
python demo_function_call.py

# 只跑一个
python demo_function_call.py --tool stock
python demo_function_call.py --tool order
```

**两个工具**：
- `get_stock_quote`：金融股价查询，schema 含 string+enum+regex+array+minItems
- `create_order`：电商下单，schema 含 integer 范围+手机号正则+多枚举

**每个工具 50 条测试**，三种模式对比，产出：
- 终端表格：JSON 合法率 / 必选字段率 / schema 完全通过率
- 典型失败案例（前 3 条）
- `outputs/function_call_results.json`：详细数据（可用于后续分析）

**预期结果**：
| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 合法 | ~90% | 100% | 100% |
| 字段齐全 | ~90% | 100% | 100% |
| **完整 schema 通过** | **40-60%** | **40-70%** | **100%** |

**核心教学点**：`response_format` 和 `guided_json` 之间的 30~50 个百分点差距就是约束解码的工程价值——`response_format` 只管语法，不管字段值是否合法。

---

### 3.6 bench_throughput.py — 吞吐对比

```bash
# 先停 vLLM server（需要释放显存）
fuser -k 8000/tcp

python bench_throughput.py
```

**三种路线**：
- [A] transformers 串行（一次一条）
- [B] transformers batch=8（手动 padding）
- [C] vLLM 批处理（内置 continuous batching）

**产出**：
- 终端表格：总耗时 / QPS / tokens/s / 相对 vLLM 的倍率
- `outputs/throughput_comparison.png`：三路对比柱状图
- `outputs/throughput_results.json`：详细数据

**预期倍率**（Qwen2-0.5B / RTX 4060 8GB）：
- 串行 ≈ 基准 1×
- batch=8 ≈ 2~4×
- vLLM ≈ 5~15×（视请求多样性）

======================================================================
  结果汇总
======================================================================
模式                            总耗时         QPS       tokens/s    相对最优
--------------------------------------------------------------------------------
[A] transformers 串行          56.78s      0.35         18       5.40×
[B] transformers batch=4     37.12s      0.54         27       8.27×
[C] llama.cpp 连续批处理(等效vLLM) 306.86s      0.07          0       1.00×

JSON 结果保存：outputs\throughput_results.json

柱状图已保存：outputs\throughput_comparison.png

======================================================================
  核心结论：
    llama.cpp 相对 transformers 串行加速：0.2×
    llama.cpp 相对 transformers batch:    0.1×
    底层优化机制：PagedAttention分页KV缓存 + continuous batching动态批处理
======================================================================
异常问题说明（C 段速度极慢、token 生成 0）
硬件瓶颈：轻薄本纯 CPU 无加速，llama.cpp 单条推理耗时极长，20 条请求堆积队列，总耗时直接拉到 306 秒；
tokens/s=0 代表服务中途频繁断连、部分请求推理失败，没有有效生成 token，指标失效；
理论优势完全被你的 CPU 性能抵消：PagedAttention、连续批处理是 GPU 大并发场景优化，纯轻薄本 CPU 小批量下完全体现不出优势，反而进程通信、调度开销拖慢速度。
二、实测现象分析
原生 transformers 静态 batch 相比串行有明显提升：耗时缩短，QPS、token 生成速度更高；
llama.cpp 在你本机纯 CPU 环境下表现远差于原生 transformers，不具备参考价值，原因：
无 GPU 加速，量化推理解码开销巨大；
客户端与 llama-server 频繁 TCP 连接重置，大量请求推理失败，有效 token 为 0；
调度、进程、网络通信额外开销远大于缓存优化带来的收益。
三、作业报告标准写法
测试环境说明
本机为轻薄本，仅依靠 CPU 运行模型，无独立 GPU 加速；测试模型 Qwen2.5-0.5B-Instruct，共 20 条混合长短请求，单轮最大生成 30token。
实测数据结论
Transformers 串行推理基线：总耗时 56.78s，QPS=0.35，token 生成速度 18 token/s；
Transformers 静态 Batch=4 批处理：相比串行耗时降低 34.6%，QPS 提升至 0.54，token 生成速度提升至 27 token/s；静态批处理通过并行计算小幅提升吞吐；
llama.cpp 连续批处理（等效 vLLM）：受本机纯 CPU 性能限制，服务频繁断连、大量请求推理失效，总耗时 306.86s，无有效生成 token，实测吞吐低于原生 transformers，硬件短板掩盖了底层优化优势。
理论原理补充
PagedAttention：将 KV 缓存分块分页存储，消除传统推理连续内存分配带来的内存碎片，GPU 多并发场景可大幅降低显存占用；
Continuous Batching（连续批处理）：不等待整批全部输入完成，动态将新请求拼入正在推理的批次，解决静态 batch 被最长输入拖慢全局的问题；
适用场景说明：两项优化仅在 GPU 高并发场景下收益显著，纯轻薄本 CPU 单机器小批量测试无法体现性能优势。
四、图表使用建议
柱状图已经自动生成，报告中图表可以保留三段，但文字重点标注 C 段受 CPU 硬件限制，实测数据异常，仅 A、B 两段可用于对比原生 transformers 串行与静态批处理的差异。
======================================================================


跑完后重新启动 server 继续 demo：
```bash
bash start_server.sh
```

---

## 四、作为模块调用

除了命令行，也可以把核心逻辑 import 进自己的应用。

### 4.1 启动 server 后用 OpenAI 客户端（推荐）

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

resp = client.chat.completions.create(
    model="qwen2-0.5b",
    messages=[{"role": "user", "content": "查茅台股价"}],
    extra_body={"guided_json": YOUR_SCHEMA},   # vLLM 扩展字段
    temperature=0,
)
print(resp.choices[0].message.content)
```

### 4.2 离线批处理（无 server）

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/path/to/Qwen2-0.5B-Instruct",
          max_model_len=2048, gpu_memory_utilization=0.6)

outputs = llm.generate(
    ["prompt 1", "prompt 2", ...],
    SamplingParams(temperature=0, max_tokens=100,
                   guided_decoding=GuidedDecodingParams(json=schema))
)
```

---

## 五、常见问题

### Q1：`ModuleNotFoundError: No module named 'vllm'`
先 `source ~/vllm_env/bin/activate` 激活虚拟环境。

### Q2：`torch.cuda.is_available()` 返回 False
多半是装了 CUDA 13 版本的 torch（需要驱动 580+）。降级：
```bash
pip uninstall -y torch vllm
pip install vllm==0.9.2
pip install transformers==4.52.4   # 也要一并降级
```

### Q3：`aimv2 is already used by a Transformers config`
transformers 版本过新（5.x）。`pip install transformers==4.52.4`。

### Q4：server 启动报 `ValueError: No available memory for the cache blocks`
显存不足。降低 `gpu-memory-utilization`（0.6 → 0.4）或降低 `max-model-len`（2048 → 1024）。

### Q5：demo 脚本报 `Connection refused`
vLLM server 没启动。另开终端跑 `bash start_server.sh`，等看到 `Application startup complete`。

### Q6：跑 bench_throughput.py 显存溢出
正常，因为同时有 transformers 模型 + vLLM 模型。请**先停掉 vLLM server**：
```bash
fuser -k 8000/tcp
```

### Q7：矩阵正则或 Schema 约束下解码特别慢
约束解码有 FSM 构建开销（首次约束下 ~1-2 秒），之后缓存命中就快。如果反复构建 schema 导致慢，可以把 schema 改成字符串传入（vLLM 会自动哈希缓存）。

### Q8：Windows 路径在 WSL 里用 `/mnt/d/badou/项目材料准备/` 正常吗
完全正常。WSL2 的文件系统桥接层支持中文路径（UTF-8），跑脚本、加载模型都没问题，只是跨文件系统读写比纯 WSL 原生 ext4 慢约 2-5 倍，模型权重只加载一次，影响可忽略。
