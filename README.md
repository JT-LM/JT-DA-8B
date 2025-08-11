# JT-DA-8B
<p align="center">
 🤗 <a href="https://huggingface.co/JT-LM/JT-DA-8B" target="_blank">Huggingface</a> 


## Introduction

In this work, we present JT-DA-8B, a specialized large language model designed for complex table reasoning tasks across diverse real-world scenarios. To address the lack of high-quality supervision in tabular reasoning scenarios, we construct a comprehensive and diverse training corpus by aggregating 29 public table QA datasets, over 300 domain-specific non-QA tables, and generating 1 million rule-based table reasoning QA pairs. A structured pipeline is proposed to generate realistic multi-step analytical tasks involving reasoning patterns like correlation, anomaly detection, and hypothesis testing. The model is trained upon open-sourced JT-Coder-8B model, an 8B-parameter decoder-only foundation model trained from scratch. In the training stage, we leverage LLM-based scoring and workflow-aligned filtering to distill high-quality, table-centric data. A four-stage table reasoning workflow is proposed, including table preprocessing, table sensing, tool-integrated reasoning, and prompt engineering, to improve model interpretability and execution accuracy. Experimental results show that JT-DA-8B achieves strong performance across various table tasks, demonstrating the effectiveness of data-centric generation and workflow-driven optimization.

This repository provides a complete open-source solution for both training and inference of our JT-DA-8B model. We hope this open-source release can benefit the community by providing a practical reference for building domain-specific large models, especially in the area of structured data intelligence. 
You can download the full model parameters at [JT-DA-8B](https://huggingface.co/JT-LM/JT-DA-8B).

## SFT

You can fine-tune our model using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a flexible and powerful training framework designed for LLaMA-style large language models. We recommend pulling the latest official code from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training purposes. Follow the steps below to perform full-parameter supervised fine-tuning (SFT) on your custom dataset:

### 1. Environment Setup
First, clone the LLaMA-Factory repository and install its dependencies.
Please see [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for details.


### 2. Register Jiutian Model in Transformers
To integrate Jiutian model with the Hugging Face transformers library, follow the following steps:

#### 2.1 Locate the installation path of ```transformers```

Run the following command to find the package installation directory:
```bash
pip show transformers
```

For example, the path may look like: 
```bash
/opt/conda/envs/jttrain/lib/python3.9/site-packages
```

#### 2.2 Copy the Jiutian configuration and modeling code into transformers source
Copy the ```jiutian``` directory provided in this repo into the transformers path:

```bash
cp -r jiutian /opt/conda/envs/jttrain/lib/python3.9/site-packages/transformers/models/
```

#### 2.3 Register Jiutian in ```configuration_auto.py```

Open the configuration auto-mapping file:
```bash
vim /opt/conda/envs/jttrain/lib/python3.9/site-packages/transformers/models/auto/configuration_auto.py
```

Add the following entries:

```python
CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # ... other configs
        ("jiutian", "JiutianConfig"),
    ]
)

MODEL_NAMES_MAPPING = OrderedDict(
    [
        # ... other model names
        ("jiutian", "Jiutian"),
    ]
)
```

#### 2.4 Register Jiutian in ```modeling_auto.py```

Open the model auto-mapping file:
```bash
vim /opt/conda/envs/jttrain/lib/python3.9/site-packages/transformers/models/auto/modeling_auto.py
```

Add the following entries:

```python
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # ... other models
        ("jiutian", "JiutianForCausalLM"),
    ]
)
```

### 3. Prepare Your Dataset
Organize your training data according to the input format required by LLaMA-Factory (e.g., instruction-style JSONL). Then, register your dataset by modifying data/dataset_info.json, adding a new entry that describes your dataset’s structure, fields, and task type.

### 4. Configure Training Parameters
Modify the training configuration file examples/train_full/jiutian.yaml, especially model path and dataset name.

### 5. Launch Training
Use the following command to launch training with torchrun on 8 GPUs (modify as needed based on your computing environment):

```bash
cd LLaMA-Factory

export NCCL_SOCKET_IFNAME=eth

torchrun --nproc_per_node=8 --nnodes=1 --master_addr=127.0.0.1 --master_port=29500 --node_rank=0 src/train.py examples/train_full/jiutian.yaml
```


## Deployment

To deploy the model as an online service for real-time table reasoning and interactive data analysis, follow the steps below:

### 1. Environment Setup
Navigate to the src directory and install all required Python packages:
```bash
cd src
pip install -r requirements.txt
```

### 2. Launch the Inference Service
Run the following command to start the API service. This will expose the endpoint on port 8006 and make it accessible for external calls:
```bash
python main.py --host 0.0.0.0 --port 8006
```

### 3. Inference
Once the service is running, you can send inference requests using curl. Below is an example to initiate a table reasoning task (e.g., generating a line chart from a CSV file):
```bash 
curl -N -X POST http://0.0.0.0:8006/v1/llm_data_analyze_stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer " \
  -d '{
    "recordId": "000",
    "modelId": "JT-DA-8B",
    "stream": true,
    "userId": "test",
    "filePath": ["dataset/example_1.csv"],
    "params": {},
    "prompt": "Please draw a line chart.",
    "history": []
}'
```



## License
JT-DA-8B is distributed under the terms of the [Apache 2.0](https://gitee.com/link?target=https%3A%2F%2Fspdx.org%2Flicenses%2FApache-2.0.html) license.


## Disclaimer

JT-DA-8B is a large language model. Despite extensive data curation and training efforts, it may still produce content that is inaccurate, biased, or inappropriate. Users are advised to exercise discretion when interpreting the model’s outputs and assume full responsibility for any outcomes arising from its use.

<br>
