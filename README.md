项目简介：

本项目在 Pythia-70M 模型上用无训练方法实现了 MQA 风格推理优化方法。实验通过 KV Sharing 的方式，让多个 Attention Head 共享 Key 和 Value，从而减少 KV Cache 的冗余。

加速/优化效果：

实验分为未经 MQA 的 baseline 部分和经过 MQA 的部分，分别对 WikiText 进行 PPL 测试和生成时间检测，结果表明，当设置

```
max_new_tokens=256

test_text = wikitext_samples[0][:2000]
```

的条件下，它们的 PPL 都是81.60，而经过 MQA 优化的生成时长为3.53s，未经 MQA 优化的生成时长为3.85s，这表明优化方法

在保持 PPL 基本不变的情况下，实现了一定程度的推理加速，但由于实验使用的是较小规模模型以及CPU推理环境，因此加速效果相对有限。但是当设置

```
max_new_tokens=512

test_text = wikitext_samples[0][:4000]
```

的条件下，经过 MQA 优化的生成时间为7.31s，未经优化的生成时间为7.18s，这是由于在我更长的输入后，Patch 带来的额外 Python 开销超过了 KV Sharing 的收益。实际上，我所实现的优化方法并不是真正工业级别的 MQA ，因为我并没有真正改变 Attention 的主计算，而是修改了 KV Cache，是一种 MQA-Style 的 KV Sharing ，在自回归生成的模式下，后面的 token 会继续读取历史的 KV Cache ，从而造成加速效果，且并不破坏PPL。 
