技术栈：
完成内容：
    1. 基于c++17/c++14，cuda构建Qwen模型的完整推理框架，基于RAII思想，依次设计并实现：内存控制器，Buffer层，Tensor层Op算子层，Model模型层。使用mmap映射磁盘的权重文件到内存，并基于预先分配的中间输入输出张量和KVcache，实现Qwen2.5-0.5B-Instruct模型的流式问答推理。并构建整体推理框架的评估流程，在单卡4090，50G的服务器下实现70tokens/s的吞吐量。
性能优化：
（profile, Nsight compute定位性能瓶颈）
在4090，50G的服务器上，Qwen2.5-0.5B-Instruct模型， 输出128tokens的条件下，测得benchmark为：
Qwen2.5-0.5B-Instruct, RTX 4090, FP32, greedy, warmup=3, repeat=10
═══════════════════════════════════════════════════════════════
Prompt tokens:         50
Output tokens:        128
───────────────────────────────────────────────────────────
Model load:          1665.63 ms
TTFT (model-only):    335.38 ms         (prefill 50token)
TTFT (end-to-end):    335.55 ms
Prefill throughput:   149.09 tok/s

Decode:              1370.86 ms
TPOT:                  10.79 ms
Decode throughput:     92.64 tok/s
E2E latency:         1706.63 ms
E2E throughput:        75.00 tok/s
Peak GPU mem:          4183 MB
───────────────────────────────────────────────────────────
ITL P50:              17.35 ms
ITL P90:              22.42 ms
ITL P95:              22.99 ms
ITL P99:              23.53 ms









    1. 实现paged attention, 构建显存池
    2. 优化各个算子性能
    3. 量化