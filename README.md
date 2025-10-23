# AskBench

AskBench 提供一个端到端的自动化问答（QnA）生成框架，用于从领域文档中构建高质量的多选题数据集。该框架封装了文档导入、LLM 生成、自动验证、语义聚类以及数据导出的完整流程，便于快速搭建 TeleQnA 风格的问答生产线。

## 架构总览

```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────┐
│ Document    │ → │ Question      │ → │ Validation   │ → │ Clustering    │ → │ Exporter  │
│ Ingestion   │   │ Generation    │   │ & Filtering  │   │ & Summaries   │   │           │
└─────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └───────────┘
```

- **Ingestion**：加载 PDF/标准/论文等文档并进行语义切分。
- **Generation**：调用 LLM 针对每个文档块生成多选题候选。
- **Validation**：启发式与 LLM 自一致性打分过滤低质量问题。
- **Clustering**：基于嵌入进行相似题聚类，生成主题摘要。
- **Exporter**：导出 JSONL/CSV 等格式的结构化问答数据。

## 快速上手

```python
from pathlib import Path

from askbench import (
    Pipeline,
    PipelineConfig,
    IngestionConfig,
    GenerationConfig,
    ValidationConfig,
    ClusteringConfig,
    ExportConfig,
)
from askbench.llm.openai_client import OpenAIClient

config = PipelineConfig(
    ingestion=IngestionConfig(input_paths=[Path("docs/standard.txt")]),
    generation=GenerationConfig(llm_model="gpt-4.1"),
    validation=ValidationConfig(min_quality_score=0.6),
    clustering=ClusteringConfig(embedding_model="text-embedding-3-small"),
    export=ExportConfig(output_path=Path("artifacts/qna.jsonl")),
)

client = OpenAIClient(
    model_name="gpt-4.1",
    chat_callable=lambda **kwargs: ...,          # 接入实际的 API 调用
    completion_callable=lambda **kwargs: ...,
    embedding_callable=lambda **kwargs: ...,
)

pipeline = Pipeline(
    config=config,
    llm_for_generation=client,
    llm_for_validation=client,
    llm_for_clustering=client,
)

pipeline.run()
```

## 目录结构

```
askbench/
├── clustering/          # 嵌入聚类与主题摘要
├── domain/              # 核心领域模型 dataclass
├── exporters/           # 数据导出工具
├── generation/          # LLM 生成逻辑
├── ingestion/           # 文档加载与切分
├── llm/                 # LLM 抽象层
├── validation/          # 质量验证模块
└── pipeline.py          # 管线协调入口
```

## 下一步

- 集成实际的 PDF/HTML 文档解析与嵌入模型。
- 扩展验证器，加入答案正确性追问、知识库对齐等。
- 在 `utils/` 下添加监控、指标与可视化模块。

