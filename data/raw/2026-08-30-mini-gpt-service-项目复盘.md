# Mini GPT Service — 项目复盘文档

**Project:** `mini-gpt-service`  
**Document Type:** 项目学习档案与工程复盘  
**Last Updated:** 2026-08-30

## 1. 项目背景与目标

`mini-gpt-service` 是完成 `gpt-from-scratch` 后开展的第一个 AI 工程化项目。

前一阶段主要回答：

> GPT 模型是如何工作的？

本项目进一步回答：

> 一个训练完成的模型，如何从 checkpoint 转变为能够被外部调用、测试和部署的软件服务？

项目重点不是继续提高模型效果，而是围绕模型推理建立一套完整的小型工程体系。

最终核心调用链：

```text
HTTP Request
    ↓
FastAPI Route
    ↓
Dependency
    ↓
GenerationService
    ↓
GPTInferenceRuntime
    ↓
Tokenizer + GPT
    ↓
Model Artifact
```

主要训练能力：

- FastAPI API 开发
- 模型推理服务化
- Application lifecycle management
- Dependency Injection
- API、Service、Runtime 与 Model 分层
- Model artifact 管理
- Domain exception 与 HTTP error mapping
- Configuration 与 logging
- 基础 observability
- Unit tests 与 integration tests
- Docker 容器化
- 从训练工程向推理工程迁移模型的能力

## 2. 从训练工程到 Portable Model Artifact

原始 GPT 训练项目包含 Dataset、DataLoader、Trainer、Optimizer、训练脚本、实验代码和 checkpoint，但推理服务并不需要完整训练工程。

本项目将推理所需内容整理成独立 artifact：

```text
model-artifacts/
└── char-gpt-v1/
    ├── model.pt
    ├── config.json
    └── tokenizer.json
```

| File | Responsibility |
| --- | --- |
| `model.pt` | 保存训练后的模型参数 |
| `config.json` | 保存重建 GPT 所需的结构参数 |
| `tokenizer.json` | 保存训练时使用的字符与 token ID 映射 |

由此形成：

```text
Training Project
      ↓
Portable Model Artifact
      ↓
Serving Project
```

核心结论：

> 模型部署不是把整个训练项目搬到服务器，而是让推理工程消费结构明确、可移植的模型产物。

## 3. Inference Runtime

GPT 模型本身处理的是 token IDs：

```text
Token IDs
    ↓
GPT
    ↓
Token IDs
```

用户输入和期望输出则是文本。因此项目建立 `GPTInferenceRuntime`，封装完整推理过程：

```text
Prompt
  ↓
Tokenizer.encode()
  ↓
Token IDs → Tensor → Device
  ↓
GPT.generate()
  ↓
Generated Token IDs
  ↓
Tokenizer.decode()
  ↓
Text
```

Runtime 负责隐藏：

- Tensor 构造
- Batch dimension
- Tokenizer 调用
- Device placement
- `model.generate()` 调用
- Token IDs 解码

上层只需调用：

```python
runtime.generate(...)
```

由此实现：

```text
Business Logic
      ↓
Inference Runtime
      ↓
Model Implementation Details
```

## 4. Service Layer

Runtime 解决“模型怎样运行”，`GenerationService` 则表达应用提供的“文本生成”业务能力。

```text
Route
  ↓
GenerationService
  ↓
GPTInferenceRuntime
```

Service 负责：

- 调用 Runtime
- 统计生成延迟
- 统计 generated tokens
- 记录 generation 日志
- 返回 `GenerationResult`

项目最终形成四层职责：

| Layer | Responsibility |
| --- | --- |
| API Layer | HTTP request/response |
| Service Layer | Application business logic |
| Runtime Layer | Model inference orchestration |
| Model Layer | Neural network computation |

## 5. FastAPI API

核心接口：

```text
GET  /health
POST /generate
```

`/generate` 接收：

- `prompt`
- `max_tokens`
- `temperature`
- `top_k`

请求参数通过 Pydantic schema 约束，例如：

- `prompt` 不能为空
- `max_tokens > 0`
- `temperature` 位于合法范围
- `top_k > 0`

请求链路：

```text
HTTP Request
      ↓
Pydantic Validation
      ↓
Route
      ↓
Service
```

Route 只负责接收请求、获得依赖、调用 Service，并将业务结果转换为 HTTP response，不直接承担模型推理逻辑。

## 6. Dependency Injection

Route 通过 FastAPI dependency 获取 `GenerationService`：

```python
Depends(get_generation_service)
```

```text
Route declares GenerationService dependency
              ↓
FastAPI calls Dependency
              ↓
Dependency reads app.state.generation_service
              ↓
Route receives ready-to-use Service
```

Route 不需要知道 Service 如何构造、Runtime 位于何处或模型何时加载。

Dependency Injection 的核心价值不是消除一切修改，而是限制变化传播：

```text
Model implementation change  → Runtime
Business rule change          → Service
Dependency acquisition change → Dependency
HTTP protocol change          → Route
```

## 7. Application Lifespan 与模型生命周期

模型是昂贵的 application-level resource，不应在每个 request 中重复加载。

错误生命周期：

```text
Request
  ↓
Load Model
  ↓
Generate
  ↓
Request Ends
```

正确生命周期：

```text
Uvicorn Starts
      ↓
FastAPI Lifespan
      ↓
Read Settings
      ↓
Load Runtime and Model
      ↓
Build GenerationService
      ↓
Save to app.state
      ↓
Application Ready
```

请求阶段复用已经加载的 Service、Runtime 和 GPT：

```text
Request
  ↓
Dependency
  ↓
Existing GenerationService
  ↓
Existing Runtime and Model
```

核心结论：

> 昂贵资源应在 application startup 时加载一次，并由后续 requests 复用。

## 8. Model Loader 与可移植性

`load_runtime()` 负责：

```text
Artifact Path
      ↓
Read config.json
      ↓
Construct GPT
      ↓
torch.load(model.pt, map_location="cpu")
      ↓
load_state_dict
      ↓
Load tokenizer.json
      ↓
Select Device
      ↓
model.to(device) + model.eval()
      ↓
Construct Runtime
```

使用：

```python
map_location = "cpu"
```

先将 checkpoint 恢复到 CPU，再根据服务运行环境选择最终 device，使 artifact 不绑定训练时的特定 GPU，也为 Docker CPU inference 提供基础。

## 9. 异常处理与依赖方向

Character-level tokenizer 可能遇到训练 vocabulary 中不存在的字符。项目建立领域异常转换：

```text
KeyError
    ↓
UnknownCharacterError
    ↓
FastAPI Exception Handler
    ↓
HTTP 400
```

Tokenizer 负责描述领域问题，API 层负责决定 HTTP 表达。

底层 tokenizer 不直接抛 FastAPI `HTTPException`，从而保持正确依赖方向：

```text
API
 ↓
Service
 ↓
Runtime
 ↓
Tokenizer / Model
```

使用：

```python
raise UnknownCharacterError(...) from exc
```

既提供有意义的 domain error，又保留原始异常链。

## 10. HTTP 400、422 与 500

| Situation | HTTP Status |
| --- | ---: |
| Pydantic schema validation error | 422 |
| Request structurally valid but unsupported by model | 400 |
| Unknown internal program error | 500 |

没有使用 catch-all 将所有异常映射为 400，因为这会把真实程序 bug 伪装成用户输入问题。

## 11. Observability

`GenerationService` 返回：

```text
GenerationResult
├── text
├── generated_tokens
└── latency_ms
```

使用：

```python
time.perf_counter()
```

测量完整 generation latency。

当前 tokenizer 为 character-level，因此第一版采用：

```text
generated_tokens ≈ len(output_text) - len(prompt)
```

该计算只适用于当前字符级模型，不能直接迁移到 BPE tokenizer。

日志记录：

- `prompt_length`
- `generated_tokens`
- `latency_ms`

不记录完整 prompt 与 output，以降低真实服务中的隐私和敏感信息泄漏风险。

## 12. 配置管理

项目建立集中式 `Settings`，支持环境变量覆盖：

```text
MINI_GPT_MODEL_ARTIFACT_PATH
MINI_GPT_LOG_LEVEL
```

### Model Config

描述“模型是什么”，例如：

- `vocab_size`
- `block_size`
- `embedding_dim`
- `num_layers`
- `num_heads`

存放于：

```text
model-artifacts/<model-version>/config.json
```

### Application Config

描述“服务如何运行”，例如：

- `model_artifact_path`
- `log_level`

存放于应用 Settings，并可由 environment variables 覆盖。

两类配置保持分离，避免 model architecture 与 serving environment 相互耦合。

## 13. Logging

应用启动阶段统一执行：

```text
configure_logging()
```

业务模块只获取并使用 logger：

```python
logger = logging.getLogger(__name__)
logger.info(...)
```

不在每个模块重复调用 `logging.basicConfig()`。

核心原则：

> 基础设施配置集中管理，业务模块只使用基础设施。

## 14. Testing

项目建立两类测试：

```text
Unit Tests
Integration Tests
```

### Unit Tests

主要验证：

- Tokenizer
- Generation Service
- `GenerationResult`
- Exception translation

### Integration Tests

主要验证：

```text
FastAPI
  ↓
Route
  ↓
Dependency
  ↓
Service / Fake Service
```

通过 dependency override、Fake Runtime 与 Fake Service，测试无需加载真实 GPT checkpoint。

收益：

- 执行更快
- 结果更稳定
- 不依赖 GPU
- 不依赖大型 artifact
- 更容易覆盖异常分支

## 15. Docker 容器化

Docker 将项目从：

```text
Runs in Personal WSL + venv
```

推进到：

```text
Runs in a Defined Clean Environment
```

构建与运行关系：

```text
Dockerfile
   ↓ docker build
Image
   ↓ docker run
Container
```

容器内 Uvicorn 监听：

```text
0.0.0.0:8000
```

Docker 端口映射：

```text
Host :8000
    ↓ docker -p 8000:8000
Container :8000
    ↓
Uvicorn
```

`0.0.0.0` 解决容器内服务是否接受外部网络接口请求，`-p` 解决宿主机流量如何进入容器。

## 16. Serving Image 与 Model Artifact 分离

第一版采用：

```text
Docker Image
      +
External Model Artifact
```

运行时通过 bind mount 或 volume：

```text
Host model-artifacts/
          ↓
Container /app/model-artifacts/
```

并以 read-only 方式挂载：

```text
:ro
```

Serving code 与 model artifact 因此拥有独立生命周期，服务只读取模型，不修改模型。

## 17. Docker 对可复现性的验证价值

Docker 不只是部署工具，也是对依赖声明和运行假设的检查。

如果本地 `.venv` 可以运行，而 clean Docker environment 无法运行，可能暴露：

- 未声明依赖
- 本地隐式安装
- 路径依赖
- 环境变量遗漏
- 平台依赖

因此 Docker build/run 本身也是一次 reproducibility validation。

## 18. 最终形成的工程原则

### 18.1 模型和服务不是同一个对象

```text
Model
  ≠
Inference Runtime
  ≠
Business Service
  ≠
HTTP API
```

### 18.2 每层只解决自己的问题

```text
Route     → HTTP
Service   → Business Logic
Runtime   → Inference
Tokenizer / Model → Model Computation
```

### 18.3 昂贵资源必须管理生命周期

```text
Startup Load Once
        ↓
Requests Reuse
```

### 18.4 底层模块不依赖上层协议

Tokenizer 和 Model 不需要了解 FastAPI、HTTP 或 status code。

### 18.5 Training 与 Serving 通过 Artifact 解耦

```text
Training
   ↓
Artifact
   ↓
Serving
```

### 18.6 测试应能替换昂贵依赖

```text
Real Model / Runtime
        ↓ replace
Fake Runtime / Fake Service
```

## 19. 当前项目边界

当前项目有意保持为小型 AI inference service，尚未引入：

- BPE tokenizer
- Asynchronous inference
- Request batching
- Streaming
- Multi-model runtime
- GPU-oriented Docker serving
- Authentication
- Rate limiting
- Task queue
- Prometheus / OpenTelemetry
- Kubernetes
- 完整 CI/CD

这些属于项目范围控制，而不是当前阶段的核心遗漏。继续堆叠组件会降低现阶段学习投入产出比。

## 20. 项目总结

`mini-gpt-service` 完成了从：

```text
I Can Train a GPT
```

到：

```text
I Can Organize a Trained Model into a Callable,
Testable, Configurable, and Containerized Service
```

的第一次工程跨越。

如果 `gpt-from-scratch` 回答“GPT 如何工作”，那么 `mini-gpt-service` 回答“模型训练完成后如何进入真实软件系统”。

本项目留给后续 AI 工程项目的主要资产是：

- API
- Service
- Runtime
- Dependency Injection
- Lifespan
- Configuration
- Domain Exceptions
- Logging
- Testing
- Model Artifacts
- Docker

> Mini GPT Service 的核心成果不是字符级 GPT 本身，而是围绕 AI 模型建立软件系统的方法。

## 21. 收尾状态备注

本文依据当前项目总结整理。若 Docker 最终验证或 README 收尾后的真实实现与本文存在差异，应回填以下内容：

- 最终 Docker build/run 验证结果
- 最终启动命令与 volume mount 示例
- README 中公开的 API 使用方式
- 最终 project status 与已知限制

