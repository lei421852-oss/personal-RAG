# Mini GPT Service → RAG 项目过渡交接文档

**Source Project:** `mini-gpt-service`  
**Target Project:** RAG System  
**Document Type:** 学习阶段交接与架构迁移指南  
**Last Updated:** 2026-08-30

## 1. 当前学习阶段

已经完成：

```text
GPT From Scratch
        ↓
Mini GPT Service
```

现在准备进入：

```text
RAG System
```

三个阶段分别回答：

| Stage | Core Question |
| --- | --- |
| GPT From Scratch | 模型内部如何工作？ |
| Mini GPT Service | 模型如何成为软件服务？ |
| RAG System | 模型如何与外部知识、检索系统和数据流程协作？ |

RAG 项目不需要从头重复 FastAPI 和 Docker 基础，而应直接复用 Mini GPT Service 已建立的工程能力。

## 2. 可直接迁移的整体架构

Mini GPT Service 已形成：

```text
API
 ↓
Service
 ↓
Runtime
 ↓
Model
```

RAG 可以扩展为：

```text
API
 ↓
RAG Service
 ↓
Retrieval / Generation Pipeline
 ↓
Embedding / Vector Store / LLM
```

RAG 不是推翻原有架构，而是在 Service 内部和 Runtime 以下增加更复杂的 AI pipeline。

## 3. 从单模型推理到多组件 Pipeline

Mini GPT Service：

```text
Prompt
  ↓
Tokenizer
  ↓
GPT
  ↓
Answer
```

RAG：

```text
Question
  ↓
Query Processing
  ↓
Embedding
  ↓
Vector Search
  ↓
Relevant Chunks
  ↓
Prompt Construction
  ↓
LLM
  ↓
Answer
```

核心变化：

```text
Mini GPT → One Model Handles the Main Computation
RAG      → Multiple Components Collaborate per Request
```

下一阶段学习重点将从 Model Serving 转向 AI Pipeline Engineering。

## 4. Service Layer 的迁移

Mini GPT 中：

```text
GenerationService
```

表达“文本生成”业务能力。

RAG 中可演变为：

```text
RAGService
```

表达“基于知识库回答问题”的业务能力：

```text
Route
  ↓
RAGService.answer(question)
  ↓
Retriever + Prompt Builder + LLM
```

核心原则继续成立：

> Service 表达应用能力，不只是某个 Model Wrapper。

## 5. Runtime 与基础组件的扩展

Mini GPT 中只有一个主要 `GPTInferenceRuntime`，封装 Tokenizer、GPT 与 Device。

RAG 中可能出现：

```text
RAGService
 ├── Retriever
 │    ├── Embedding Runtime
 │    └── Vector Store
 ├── Prompt Builder
 └── LLM Runtime
```

Route 不应直接操作：

```text
embedding.encode()
vector_store.search()
prompt concatenation
llm.generate()
```

否则 Route 会迅速膨胀。正确方向仍然是：

```text
Route
  ↓
RAGService
  ↓
RAG Pipeline
```

## 6. Dependency Injection 的进一步价值

RAG 依赖可能包括：

- Embedding model
- Vector store
- Retriever
- LLM client/runtime
- RAG service

这些对象不应在 Route 中逐一创建。

建议沿用：

```text
Application Startup
        ↓
Construct Shared Resources
        ↓
Dependency Providers
        ↓
Route Receives RAGService
```

测试时可替换：

```text
Real Embedding  → Fake Embedding
Real Vector DB  → Fake Vector Store
Real Retriever  → Fake Retriever
Real LLM        → Fake LLM
```

RAG 的多依赖结构会比 Mini GPT 更明显地体现 DI 对解耦和测试的价值。

## 7. Lifespan 与资源生命周期

Mini GPT 中的昂贵资源是 GPT model，因此在 startup 加载一次并复用。

RAG 中的 application-level resources 可能包括：

- Embedding model
- Vector store client/connection
- Retriever
- LLM client/runtime

Request-level data 包括：

- User question
- Query embedding result
- Retrieved chunks
- Constructed prompt
- Generated answer

需要持续回答：

```text
Which resources are application-level?
Which values are request-level?
```

核心原则：

> 可复用的昂贵资源不应在每个 request 中重新创建或加载。

## 8. Configuration Management 的升级

Mini GPT 的主要配置只有 model artifact path 和 log level。RAG 配置会明显增加：

- Embedding model
- Vector store path/connection
- Chunk size
- Chunk overlap
- Retrieval top-k
- LLM model
- Temperature
- Data directory

继续坚持：

```text
Centralized Settings
        ↓
Environment Variable Overrides
        ↓
Injected Configuration
        ↓
Business Components
```

避免每个模块自行调用 `os.getenv()`，否则配置来源会重新散落。

## 9. Artifact 思想迁移到知识库

Mini GPT：

```text
Training
  ↓
Model Artifact
  ↓
Serving
```

RAG：

```text
Raw Documents
      ↓
Parsing and Cleaning
      ↓
Chunking
      ↓
Embedding
      ↓
Vector Index / Processed Document Artifact
      ↓
Serving
```

Vector index、processed chunks 与 metadata 可以视为 RAG artifacts，需要明确生成方式、版本和使用边界。

## 10. Offline / Online Pipeline

RAG 天然分为两个阶段。

### Offline Ingestion

```text
Documents
  ↓
Load
  ↓
Parse and Clean
  ↓
Chunk
  ↓
Embed
  ↓
Index
```

### Online Query

```text
Question
  ↓
Embed Query
  ↓
Retrieve
  ↓
Construct Context
  ↓
LLM Generate
  ↓
Answer
```

Mini GPT 中“模型不要每次 request 重新加载”的原则可推广为：

> 能够预处理并复用的昂贵工作，不应在每次 request 中重新执行。

## 11. Error Handling 的迁移

RAG 可能出现新的领域错误：

```text
DocumentNotFound
IndexNotReady
InvalidQuery
RetrievalError
EmbeddingError
GenerationError
```

继续坚持：

```text
Lower Layer
→ Describes What Failed

API Layer
→ Maps Domain Failure to HTTP Response
```

Retriever、Vector Store 和 Embedding Runtime 不应直接抛 FastAPI `HTTPException`。

## 12. Testing 策略迁移

RAG 可以通过 fake dependencies 分层测试：

```text
Fake Embedding
      +
Fake Vector Store
      +
Fake Retriever
      +
Fake LLM
```

分别验证：

- Document processing
- Chunking behavior
- Retriever input/output
- Prompt construction
- Service orchestration
- API behavior
- Domain exception mapping

大多数工程测试不应依赖真实 embedding model、远程 LLM 或大型 vector database。

## 13. Observability 的升级

Mini GPT 主要观察：

```text
latency_ms
generated_tokens
```

RAG 应进一步拆分：

```text
total_latency
retrieval_latency
generation_latency
retrieved_chunks
retrieval_top_k
```

因为请求已经变成多阶段 pipeline，需要回答：

> 系统慢在哪里？检索慢、生成慢，还是整体编排慢？

同时应继续避免在默认日志中记录完整问题、检索内容和回答，防止隐私与知识库数据泄漏。

## 14. Docker 与 Docker Compose

Mini GPT 已经建立：

```text
Dockerfile → Image → Container
```

以及：

```text
Host Port → Docker Port Mapping → Container Port → Uvicorn
```

RAG 可以直接复用这些知识。

如果第一版使用独立 vector database service，系统可能变为：

```text
Container A: RAG API
Container B: Vector Database
```

此时 Docker Compose 才具有明确价值，用于共同启动、配置网络和管理依赖服务。

如果第一版使用进程内或文件型 vector store，则不必为了技术堆叠而提前引入 Compose。

## 15. 已经具备、无需重新学习的工程基础

以下能力可以直接复用：

- FastAPI basics
- Pydantic schemas
- Route design
- Dependency Injection
- Service Layer
- Application lifespan
- `app.state`
- Centralized Settings
- Environment variables
- Python logging
- Domain exceptions
- Exception handlers
- Unit tests
- Integration tests
- Dependency overrides
- Docker basics
- Port mapping
- Volume/bind mounts

RAG 项目应将时间投入新的检索与知识工程问题，而不是机械重复上述内容。

## 16. RAG 阶段需要新增的知识

### 16.1 Document Ingestion

- Document loading
- Parsing
- Cleaning
- Metadata extraction

### 16.2 Chunking

- Chunk size
- Chunk overlap
- Boundary strategy
- Metadata preservation

### 16.3 Embeddings

- Embedding models
- Embedding dimensions
- Similarity metrics
- Normalization

### 16.4 Vector Retrieval

- Vector stores
- Indexing
- Top-k retrieval
- Similarity search
- Metadata filtering

### 16.5 RAG Pipeline

```text
Question
  ↓
Embedding
  ↓
Retrieval
  ↓
Context Selection
  ↓
Prompt Construction
  ↓
LLM
  ↓
Answer
```

### 16.6 Evaluation

RAG evaluation 需要区分：

```text
Retrieval Quality
        +
Generation Quality
```

不能只观察最终答案，而忽略是否检索到正确、充分和相关的证据。

### 16.7 Later Enhancements

基础 RAG 完成后再评估：

- Reranking
- Hybrid retrieval
- Caching
- Streaming
- Advanced observability
- Docker Compose

## 17. 推荐学习与开发方式

继续采用：

```text
Understand the Problem
        ↓
Discuss the Design
        ↓
Define Module Responsibilities
        ↓
Create Coding Task
        ↓
Implement
        ↓
Human Review
        ↓
Test
        ↓
Git Commit
```

重点 Review 问题：

- 为什么这样切 chunk？
- 为什么选择这个 embedding model？
- 为什么 top-k 设为这个值？
- Retriever 的输入输出 contract 是什么？
- 哪些步骤属于 offline，哪些属于 online？
- 哪些资源应该在 startup 创建？
- Retrieval failure 与 LLM failure 如何区分？
- 如何证明检索到了正确内容？
- 如何分别评价 retrieval 与 answer quality？

## 18. RAG 第一版建议边界

第一版只实现：

```text
Documents
  ↓
Chunking
  ↓
Embedding
  ↓
Vector Store
  ↓
Retriever
  ↓
Prompt Construction
  ↓
LLM
  ↓
FastAPI
```

目标：

> 完成一个真正端到端可运行、能够回答本地知识库问题的 RAG 系统。

第一版暂不加入：

- Agent / Multi-Agent
- GraphRAG
- Complex hybrid search
- Complex reranker
- Long-term memory
- MCP
- Complex frontend
- Kubernetes

这些能力应在 basic RAG 跑通并获得评估结果后，再根据真实需求决定是否引入。

## 19. Mini GPT Service 留给 RAG 的核心资产

```text
                 Mini GPT Service
                        │
         ┌──────────────┼──────────────┐
         ↓              ↓              ↓
    API Engineering   AI Serving    Engineering
         │              │              │
         ↓              ↓              ↓
      FastAPI         Runtime        Testing
      Schema          Lifecycle      Config
      Depends         Artifact       Logging
      Exception       Model Load     Docker
         │              │              │
         └──────────────┼──────────────┘
                        ↓
                    RAG System
                        ↓
              Add New Knowledge Pipeline
                        ↓
         Documents → Chunking → Embedding
                        ↓
                   Vector Store
                        ↓
                    Retrieval
                        ↓
                 Prompt + LLM
                        ↓
                     Answer
```

## 20. 建议的 RAG 启动顺序

```text
1. Define Project Goal and First-Version Boundary
        ↓
2. Select a Small Local Document Set
        ↓
3. Define Document and Chunk Data Models
        ↓
4. Implement Offline Loading and Chunking
        ↓
5. Add Embedding and Vector Store
        ↓
6. Implement Retriever with Explicit Contract
        ↓
7. Build Prompt Construction
        ↓
8. Connect LLM Runtime
        ↓
9. Add RAGService and API
        ↓
10. Add Retrieval and End-to-End Tests
        ↓
11. Measure Retrieval and Generation Separately
```

## 21. 项目过渡结论

Mini GPT Service 阶段解决：

> 怎样把一个 AI 模型放进软件系统？

RAG 阶段将进一步解决：

> 怎样让多个 AI 与数据组件共同组成完整的智能应用 pipeline？

下一阶段不应重新停留在 Route、Depends 或 Docker port mapping 的基础问题，而应重点回答：

```text
How Does Knowledge Enter the System?
              ↓
How Is Knowledge Represented?
              ↓
How Is Knowledge Retrieved?
              ↓
How Does Retrieved Context Enter the Prompt?
              ↓
How Does the LLM Answer from External Knowledge?
              ↓
How Do We Evaluate Retrieval and Generation?
```

这份文档可以直接作为新 RAG 项目的启动与教学交接上下文。

