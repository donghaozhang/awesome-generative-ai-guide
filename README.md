# :star: :bookmark: awesome-generative-ai-guide

Generative AI is experiencing rapid growth, and this repository serves as a comprehensive hub for updates on generative AI research, interview materials, notebooks, and more!

<a href="https://trendshift.io/repositories/7663" target="_blank"><img src="https://trendshift.io/api/badge/repositories/7663" alt="aishwaryanr%2Fawesome-generative-ai-guide | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

Explore the following resources:

1. [Monthly Best GenAI Papers List](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#star-best-genai-papers-list-january-2024)
2. [GenAI Interview Resources](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#computer-interview-prep)
3. [Applied LLMs Mastery 2024 (created by Aishwarya Naresh Reganti) course material](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#ongoing-applied-llms-mastery-2024)
4. [Generative AI Genius 2024 (created by Aishwarya Naresh Reganti) course material](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/generative_ai_genius/README.md)
5. [List of all GenAI-related free courses (over 90 listed)](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#book-list-of-free-genai-courses)
6. [List of code repositories/notebooks for developing generative AI applications](https://github.com/aishwaryanr/awesome-generative-ai-guide?tab=readme-ov-file#notebook-code-notebooks)

We'll be updating this repository regularly, so keep an eye out for the latest additions!

Happy Learning!

---

## 🌟🌟 Repo of the Month!

<div align="center">
  <img src="https://github.com/user-attachments/assets/c71db8f5-658c-48d6-9125-3a1718c465c3" alt="image" width="200">
</div>

Opik is an open-source framework for evaluating LLM systems. Designed to support RAG chatbots, code assistants, and agentic pipelines, it provides tools for tracing, evaluations, and dashboards to help improve performance and efficiency.
Link: https://github.com/comet-ml/opik

## :speaker: Announcements

- Applied LLMs Mastery full course content has been released!!! ([Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024))
- 5-day roadmap to learn LLM foundations out now! ([Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/genai_roadmap.md))
- 60 Common GenAI Interview Questions out now! ([Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/interview_prep/60_gen_ai_questions.md))
- ICLR 2024 paper summaries ([Click Here](https://areganti.notion.site/06f0d4fe46a94d62bff2ae001cfec22c?v=d501ca62e4b745768385d698f173ae14))
- List of free GenAI courses ([Click Here](https://github.com/aishwaryanr/awesome-generative-ai-guide#book-list-of-free-genai-courses))
- Generative AI resources and roadmaps
  - [3-day RAG roadmap](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/RAG_roadmap.md)
  - [5-day LLM foundations roadmap](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/genai_roadmap.md)
  - [5-day LLM agents roadmap](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/agents_roadmap.md)
  - [Agents 101 guide](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/agents_101_guide.md)
  - [Introduction to MM LLMs](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/mm_llms_guide.md)
  - [LLM Lingo Series: Commonly used LLM terms and their easy-to-understand definitions](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/llm_lingo)

---

## :star: Best Gen AI Papers List (Dec 2024)

\*Updated at the end of every month
| Date | Title | Abstract |
|------|-------|----------|
| 26th December 2024 | [Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](http://arxiv.org/abs/2412.19326v1) | Current multimodal large language models (MLLMs) struggle with fine-grained or precise understanding of visuals though they give comprehensive perception and reasoning in a spectrum of vision applications. Recent studies either develop tool-using or unify specific visual tasks into the autoregressive framework, often at the expense of overall multimodal performance. To address this issue and enhance MLLMs with visual tasks in a scalable fashion, we propose Task Preference Optimization (TPO), a novel method that utilizes differentiable task preferences derived from typical fine-grained visual tasks. TPO introduces learnable task tokens that establish connections between multiple task-specific heads and the MLLM. By leveraging rich visual labels during training, TPO significantly enhances the MLLM's multimodal capabilities and task-specific performance. Through multi-task co-training within TPO, we observe synergistic benefits that elevate individual task performance beyond what is achievable through single-task training methodologies. Our instantiation of this approach with VideoChat and LLaVA demonstrates an overall 14.6% improvement in multimodal performance compared to baseline models. Additionally, MLLM-TPO demonstrates robust zero-shot capabilities across various tasks, performing comparably to state-of-the-art supervised models. The code will be released at https://github.com/OpenGVLab/TPO |
| 24th December 2024 | [Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models](http://arxiv.org/abs/2412.18609v1) | We present an efficient encoder-free approach for video-language understanding that achieves competitive performance while significantly reducing computational overhead. Current video-language models typically rely on heavyweight image encoders (300M-1.1B parameters) or video encoders (1B-1.4B parameters), creating a substantial computational burden when processing multi-frame videos. Our method introduces a novel Spatio-Temporal Alignment Block (STAB) that directly processes video inputs without requiring pre-trained encoders while using only 45M parameters for visual processing - at least a 6.5$\times$ reduction compared to traditional approaches. The STAB architecture combines Local Spatio-Temporal Encoding for fine-grained feature extraction, efficient spatial downsampling through learned attention and separate mechanisms for modeling frame-level and video-level relationships. Our model achieves comparable or superior performance to encoder-based approaches for open-ended video question answering on standard benchmarks. The fine-grained video question-answering evaluation demonstrates our model's effectiveness, outperforming the encoder-based approaches Video-ChatGPT and Video-LLaVA in key aspects like correctness and temporal understanding. Extensive ablation studies validate our architectural choices and demonstrate the effectiveness of our spatio-temporal modeling approach while achieving 3-4$\times$ faster processing speeds than previous methods. Code is available at \url{https://github.com/jh-yi/Video-Panda}. |
| 24th December 2024 | [Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search](http://arxiv.org/abs/2412.18319v1) | In this work, we aim to develop an MLLM that understands and solves questions by learning to create each intermediate step of the reasoning involved till the final answer. To this end, we propose Collective Monte Carlo Tree Search (CoMCTS), a new learning-to-reason method for MLLMs, which introduces the concept of collective learning into ``tree search'' for effective and efficient reasoning-path searching and learning. The core idea of CoMCTS is to leverage collective knowledge from multiple models to collaboratively conjecture, search and identify effective reasoning paths toward correct answers via four iterative operations including Expansion, Simulation and Error Positioning, Backpropagation, and Selection. Using CoMCTS, we construct Mulberry-260k, a multimodal dataset with a tree of rich, explicit and well-defined reasoning nodes for each question. With Mulberry-260k, we perform collective SFT to train our model, Mulberry, a series of MLLMs with o1-like step-by-step Reasoning and Reflection capabilities. Extensive experiments demonstrate the superiority of our proposed methods on various benchmarks. Code will be available at https://github.com/HJYao00/Mulberry |
| 23rd December 2024 | [YuLan-Mini: An Open Data-efficient Language Model](http://arxiv.org/abs/2412.17743v2) | Effective pre-training of large language models (LLMs) has been challenging due to the immense resource demands and the complexity of the technical processes involved. This paper presents a detailed technical report on YuLan-Mini, a highly capable base model with 2.42B parameters that achieves top-tier performance among models of similar parameter scale. Our pre-training approach focuses on enhancing training efficacy through three key technical contributions: an elaborate data pipeline combines data cleaning with data schedule strategies, a robust optimization method to mitigate training instability, and an effective annealing approach that incorporates targeted data selection and long context training. Remarkably, YuLan-Mini, trained on 1.08T tokens, achieves performance comparable to industry-leading models that require significantly more data. To facilitate reproduction, we release the full details of the data composition for each training phase. Project details can be accessed at the following link: https://github.com/RUC-GSAI/YuLan-Mini. |
| 19th December 2024 | [Qwen2.5 Technical Report](http://arxiv.org/abs/2412.15115v1) | In this report, we introduce Qwen2.5, a comprehensive series of large language models (LLMs) designed to meet diverse needs. Compared to previous iterations, Qwen 2.5 has been significantly improved during both the pre-training and post-training stages. In terms of pre-training, we have scaled the high-quality pre-training datasets from the previous 7 trillion tokens to 18 trillion tokens. This provides a strong foundation for common sense, expert knowledge, and reasoning capabilities. In terms of post-training, we implement intricate supervised finetuning with over 1 million samples, as well as multistage reinforcement learning. Post-training techniques enhance human preference, and notably improve long text generation, structural data analysis, and instruction following. To handle diverse and varied use cases effectively, we present Qwen2.5 LLM series in rich sizes. Open-weight offerings include base and instruction-tuned models, with quantized versions available. In addition, for hosted solutions, the proprietary models currently include two mixture-of-experts (MoE) variants: Qwen2.5-Turbo and Qwen2.5-Plus, both available from Alibaba Cloud Model Studio. Qwen2.5 has demonstrated top-tier performance on a wide range of benchmarks evaluating language understanding, reasoning, mathematics, coding, human preference alignment, etc. Specifically, the open-weight flagship Qwen2.5-72B-Instruct outperforms a number of open and proprietary models and demonstrates competitive performance to the state-of-the-art open-weight model, Llama-3-405B-Instruct, which is around 5 times larger. Qwen2.5-Turbo and Qwen2.5-Plus offer superior cost-effectiveness while performing competitively against GPT-4o-mini and GPT-4o respectively. Additionally, as the foundation, Qwen2.5 models have been instrumental in training specialized models such as Qwen2.5-Math, Qwen2.5-Coder, QwQ, and multimodal models. |
| 19th December 2024 | [RobustFT: Robust Supervised Fine-tuning for Large Language Models under Noisy Response](http://arxiv.org/abs/2412.14922v1) | Supervised fine-tuning (SFT) plays a crucial role in adapting large language models (LLMs) to specific domains or tasks. However, as demonstrated by empirical experiments, the collected data inevitably contains noise in practical applications, which poses significant challenges to model performance on downstream tasks. Therefore, there is an urgent need for a noise-robust SFT framework to enhance model capabilities in downstream tasks. To address this challenge, we introduce a robust SFT framework (RobustFT) that performs noise detection and relabeling on downstream task data. For noise identification, our approach employs a multi-expert collaborative system with inference-enhanced models to achieve superior noise detection. In the denoising phase, we utilize a context-enhanced strategy, which incorporates the most relevant and confident knowledge followed by careful assessment to generate reliable annotations. Additionally, we introduce an effective data selection mechanism based on response entropy, ensuring only high-quality samples are retained for fine-tuning. Extensive experiments conducted on multiple LLMs across five datasets demonstrate RobustFT's exceptional performance in noisy scenarios. |
| 19th December 2024 | [Progressive Multimodal Reasoning via Active Retrieval](http://arxiv.org/abs/2412.14835v1) | Multi-step multimodal reasoning tasks pose significant challenges for multimodal large language models (MLLMs), and finding effective ways to enhance their performance in such scenarios remains an unresolved issue. In this paper, we propose AR-MCTS, a universal framework designed to progressively improve the reasoning capabilities of MLLMs through Active Retrieval (AR) and Monte Carlo Tree Search (MCTS). Our approach begins with the development of a unified retrieval module that retrieves key supporting insights for solving complex reasoning problems from a hybrid-modal retrieval corpus. To bridge the gap in automated multimodal reasoning verification, we employ the MCTS algorithm combined with an active retrieval mechanism, which enables the automatic generation of step-wise annotations. This strategy dynamically retrieves key insights for each reasoning step, moving beyond traditional beam search sampling to improve the diversity and reliability of the reasoning space. Additionally, we introduce a process reward model that aligns progressively to support the automatic verification of multimodal reasoning tasks. Experimental results across three complex multimodal reasoning benchmarks confirm the effectiveness of the AR-MCTS framework in enhancing the performance of various multimodal models. Further analysis demonstrates that AR-MCTS can optimize sampling diversity and accuracy, yielding reliable multimodal reasoning. |
| 19th December 2024 | [Agent-SafetyBench: Evaluating the Safety of LLM Agents](http://arxiv.org/abs/2412.14470v1) | As large language models (LLMs) are increasingly deployed as agents, their integration into interactive environments and tool use introduce new safety challenges beyond those associated with the models themselves. However, the absence of comprehensive benchmarks for evaluating agent safety presents a significant barrier to effective assessment and further improvement. In this paper, we introduce Agent-SafetyBench, a comprehensive benchmark designed to evaluate the safety of LLM agents. Agent-SafetyBench encompasses 349 interaction environments and 2,000 test cases, evaluating 8 categories of safety risks and covering 10 common failure modes frequently encountered in unsafe interactions. Our evaluation of 16 popular LLM agents reveals a concerning result: none of the agents achieves a safety score above 60%. This highlights significant safety challenges in LLM agents and underscores the considerable need for improvement. Through quantitative analysis, we identify critical failure modes and summarize two fundamental safety detects in current LLM agents: lack of robustness and lack of risk awareness. Furthermore, our findings suggest that reliance on defense prompts alone is insufficient to address these safety issues, emphasizing the need for more advanced and robust strategies. We release Agent-SafetyBench at \url{https://github.com/thu-coai/Agent-SafetyBench} to facilitate further research and innovation in agent safety evaluation and improvement. |
| 18th December 2024 | [TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks](http://arxiv.org/abs/2412.14161v1) | We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at helping to accelerate or even autonomously perform work-related tasks? The answer to this question has important implications for both industry looking to adopt AI into their workflows, and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper, we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that with the most competitive agent, 24% of the tasks can be completed autonomously. This paints a nuanced picture on task automation with LM agents -- in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. |
| 18th December 2024 | [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](http://arxiv.org/abs/2412.13663v2) | Encoder-only transformer models such as BERT offer a great performance-size tradeoff for retrieval and classification tasks with respect to larger decoder-only models. Despite being the workhorse of numerous production pipelines, there have been limited Pareto improvements to BERT since its release. In this paper, we introduce ModernBERT, bringing modern model optimizations to encoder-only models and representing a major Pareto improvement over older encoders. Trained on 2 trillion tokens with a native 8192 sequence length, ModernBERT models exhibit state-of-the-art results on a large pool of evaluations encompassing diverse classification tasks and both single and multi-vector retrieval on different domains (including code). In addition to strong downstream performance, ModernBERT is also the most speed and memory efficient encoder and is designed for inference on common GPUs. |
| 18th December 2024 | [SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](http://arxiv.org/abs/2412.13649v1) | Key-Value (KV) cache has become a bottleneck of LLMs for long-context generation. Despite the numerous efforts in this area, the optimization for the decoding phase is generally ignored. However, we believe such optimization is crucial, especially for long-output generation tasks based on the following two observations: (i) Excessive compression during the prefill phase, which requires specific full context impairs the comprehension of the reasoning task; (ii) Deviation of heavy hitters occurs in the reasoning tasks with long outputs. Therefore, SCOPE, a simple yet efficient framework that separately performs KV cache optimization during the prefill and decoding phases, is introduced. Specifically, the KV cache during the prefill phase is preserved to maintain the essential information, while a novel strategy based on sliding is proposed to select essential heavy hitters for the decoding phase. Memory usage and memory transfer are further optimized using adaptive and discontinuous strategies. Extensive experiments on LongGenBench show the effectiveness and generalization of SCOPE and its compatibility as a plug-in to other prefill-only KV compression methods. |
| 17th December 2024 | [Are Your LLMs Capable of Stable Reasoning?](http://arxiv.org/abs/2412.13147v2) | The rapid advancement of Large Language Models (LLMs) has demonstrated remarkable progress in complex reasoning tasks. However, a significant discrepancy persists between benchmark performances and real-world applications. We identify this gap as primarily stemming from current evaluation protocols and metrics, which inadequately capture the full spectrum of LLM capabilities, particularly in complex reasoning tasks where both accuracy and consistency are crucial. This work makes two key contributions. First, we introduce G-Pass@k, a novel evaluation metric that provides a continuous assessment of model performance across multiple sampling attempts, quantifying both the model's peak performance potential and its stability. Second, we present LiveMathBench, a dynamic benchmark comprising challenging, contemporary mathematical problems designed to minimize data leakage risks during evaluation. Through extensive experiments using G-Pass@k on state-of-the-art LLMs with LiveMathBench, we provide comprehensive insights into both their maximum capabilities and operational consistency. Our findings reveal substantial room for improvement in LLMs' "realistic" reasoning capabilities, highlighting the need for more robust evaluation methods. The benchmark and detailed results are available at: https://github.com/open-compass/GPassK. |
| 17th December 2024 | [OmniEval: An Omnidirectional and Automatic RAG Evaluation Benchmark in Financial Domain](http://arxiv.org/abs/2412.13018v1) | As a typical and practical application of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) techniques have gained extensive attention, particularly in vertical domains where LLMs may lack domain-specific knowledge. In this paper, we introduce an omnidirectional and automatic RAG benchmark, OmniEval, in the financial domain. Our benchmark is characterized by its multi-dimensional evaluation framework, including (1) a matrix-based RAG scenario evaluation system that categorizes queries into five task classes and 16 financial topics, leading to a structured assessment of diverse query scenarios; (2) a multi-dimensional evaluation data generation approach, which combines GPT-4-based automatic generation and human annotation, achieving an 87.47\% acceptance ratio in human evaluations on generated instances; (3) a multi-stage evaluation system that evaluates both retrieval and generation performance, result in a comprehensive evaluation on the RAG pipeline; and (4) robust evaluation metrics derived from rule-based and LLM-based ones, enhancing the reliability of assessments through manual annotations and supervised fine-tuning of an LLM evaluator. Our experiments demonstrate the comprehensiveness of OmniEval, which includes extensive test datasets and highlights the performance variations of RAG systems across diverse topics and tasks, revealing significant opportunities for RAG models to improve their capabilities in vertical domains. We open source the code of our benchmark in \href{https://github.com/RUC-NLPIR/OmniEval}{https://github.com/RUC-NLPIR/OmniEval}. |
| 16th December 2024 | [RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation](http://arxiv.org/abs/2412.11919v1) | Large language models (LLMs) exhibit remarkable generative capabilities but often suffer from hallucinations. Retrieval-augmented generation (RAG) offers an effective solution by incorporating external knowledge, but existing methods still face several limitations: additional deployment costs of separate retrievers, redundant input tokens from retrieved text chunks, and the lack of joint optimization of retrieval and generation. To address these issues, we propose \textbf{RetroLLM}, a unified framework that integrates retrieval and generation into a single, cohesive process, enabling LLMs to directly generate fine-grained evidence from the corpus with constrained decoding. Moreover, to mitigate false pruning in the process of constrained evidence generation, we introduce (1) hierarchical FM-Index constraints, which generate corpus-constrained clues to identify a subset of relevant documents before evidence generation, reducing irrelevant decoding space; and (2) a forward-looking constrained decoding strategy, which considers the relevance of future sequences to improve evidence accuracy. Extensive experiments on five open-domain QA datasets demonstrate RetroLLM's superior performance across both in-domain and out-of-domain tasks. The code is available at \url{https://github.com/sunnynexus/RetroLLM}. |
| 16th December 2024 | [Next Token Prediction Towards Multimodal Intelligence: A Comprehensive Survey](http://arxiv.org/abs/2412.18619v2) | Building on the foundations of language modeling in natural language processing, Next Token Prediction (NTP) has evolved into a versatile training objective for machine learning tasks across various modalities, achieving considerable success. As Large Language Models (LLMs) have advanced to unify understanding and generation tasks within the textual modality, recent research has shown that tasks from different modalities can also be effectively encapsulated within the NTP framework, transforming the multimodal information into tokens and predict the next one given the context. This survey introduces a comprehensive taxonomy that unifies both understanding and generation within multimodal learning through the lens of NTP. The proposed taxonomy covers five key aspects: Multimodal tokenization, MMNTP model architectures, unified task representation, datasets \& evaluation, and open challenges. This new taxonomy aims to aid researchers in their exploration of multimodal intelligence. An associated GitHub repository collecting the latest papers and repos is available at https://github.com/LMM101/Awesome-Multimodal-Next-Token-Prediction |
| 13th December 2024 | [Apollo: An Exploration of Video Understanding in Large Multimodal Models](http://arxiv.org/abs/2412.10360v1) | Despite the rapid integration of video perception capabilities into Large Multimodal Models (LMMs), the underlying mechanisms driving their video understanding remain poorly understood. Consequently, many design decisions in this domain are made without proper justification or analysis. The high computational cost of training and evaluating such models, coupled with limited open research, hinders the development of video-LMMs. To address this, we present a comprehensive study that helps uncover what effectively drives video understanding in LMMs.   We begin by critically examining the primary contributors to the high computational requirements associated with video-LMM research and discover Scaling Consistency, wherein design and training decisions made on smaller models and datasets (up to a critical size) effectively transfer to larger models. Leveraging these insights, we explored many video-specific aspects of video-LMMs, including video sampling, architectures, data composition, training schedules, and more. For example, we demonstrated that fps sampling during training is vastly preferable to uniform frame sampling and which vision encoders are the best for video representation.   Guided by these findings, we introduce Apollo, a state-of-the-art family of LMMs that achieve superior performance across different model sizes. Our models can perceive hour-long videos efficiently, with Apollo-3B outperforming most existing $7$B models with an impressive 55.1 on LongVideoBench. Apollo-7B is state-of-the-art compared to 7B LMMs with a 70.9 on MLVU, and 63.3 on Video-MME. |
| 13th December 2024 | [Large Action Models: From Inception to Implementation](http://arxiv.org/abs/2412.10047v1) | As AI continues to advance, there is a growing demand for systems that go beyond language-based assistance and move toward intelligent agents capable of performing real-world actions. This evolution requires the transition from traditional Large Language Models (LLMs), which excel at generating textual responses, to Large Action Models (LAMs), designed for action generation and execution within dynamic environments. Enabled by agent systems, LAMs hold the potential to transform AI from passive language understanding to active task completion, marking a significant milestone in the progression toward artificial general intelligence.   In this paper, we present a comprehensive framework for developing LAMs, offering a systematic approach to their creation, from inception to deployment. We begin with an overview of LAMs, highlighting their unique characteristics and delineating their differences from LLMs. Using a Windows OS-based agent as a case study, we provide a detailed, step-by-step guide on the key stages of LAM development, including data collection, model training, environment integration, grounding, and evaluation. This generalizable workflow can serve as a blueprint for creating functional LAMs in various application domains. We conclude by identifying the current limitations of LAMs and discussing directions for future research and industrial deployment, emphasizing the challenges and opportunities that lie ahead in realizing the full potential of LAMs in real-world applications.   The code for the data collection process utilized in this paper is publicly available at: https://github.com/microsoft/UFO/tree/main/dataflow, and comprehensive documentation can be found at https://microsoft.github.io/UFO/dataflow/overview/. |
| 12th December 2024 | [InternLM-XComposer2.5-OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions](http://arxiv.org/abs/2412.09596v1) | Creating AI systems that can interact with environments over long periods, similar to human cognition, has been a longstanding research goal. Recent advancements in multimodal large language models (MLLMs) have made significant strides in open-world understanding. However, the challenge of continuous and simultaneous streaming perception, memory, and reasoning remains largely unexplored. Current MLLMs are constrained by their sequence-to-sequence architecture, which limits their ability to process inputs and generate responses simultaneously, akin to being unable to think while perceiving. Furthermore, relying on long contexts to store historical data is impractical for long-term interactions, as retaining all information becomes costly and inefficient. Therefore, rather than relying on a single foundation model to perform all functions, this project draws inspiration from the concept of the Specialized Generalist AI and introduces disentangled streaming perception, reasoning, and memory mechanisms, enabling real-time interaction with streaming video and audio input. The proposed framework InternLM-XComposer2.5-OmniLive (IXC2.5-OL) consists of three key modules: (1) Streaming Perception Module: Processes multimodal information in real-time, storing key details in memory and triggering reasoning in response to user queries. (2) Multi-modal Long Memory Module: Integrates short-term and long-term memory, compressing short-term memories into long-term ones for efficient retrieval and improved accuracy. (3) Reasoning Module: Responds to queries and executes reasoning tasks, coordinating with the perception and memory modules. This project simulates human-like cognition, enabling multimodal large language models to provide continuous and adaptive service over time. |
| 12th December 2024 | [Phi-4 Technical Report](http://arxiv.org/abs/2412.08905v1) | We present phi-4, a 14-billion parameter language model developed with a training recipe that is centrally focused on data quality. Unlike most language models, where pre-training is based primarily on organic data sources such as web content or code, phi-4 strategically incorporates synthetic data throughout the training process. While previous models in the Phi family largely distill the capabilities of a teacher model (specifically GPT-4), phi-4 substantially surpasses its teacher model on STEM-focused QA capabilities, giving evidence that our data-generation and post-training techniques go beyond distillation. Despite minimal changes to the phi-3 architecture, phi-4 achieves strong performance relative to its size -- especially on reasoning-focused benchmarks -- due to improved data, training curriculum, and innovations in the post-training scheme. |
| 10th December 2024 | [Evaluation Agent: Efficient and Promptable Evaluation Framework for Visual Generative Models](http://arxiv.org/abs/2412.09645v2) | Recent advancements in visual generative models have enabled high-quality image and video generation, opening diverse applications. However, evaluating these models often demands sampling hundreds or thousands of images or videos, making the process computationally expensive, especially for diffusion-based models with inherently slow sampling. Moreover, existing evaluation methods rely on rigid pipelines that overlook specific user needs and provide numerical results without clear explanations. In contrast, humans can quickly form impressions of a model's capabilities by observing only a few samples. To mimic this, we propose the Evaluation Agent framework, which employs human-like strategies for efficient, dynamic, multi-round evaluations using only a few samples per round, while offering detailed, user-tailored analyses. It offers four key advantages: 1) efficiency, 2) promptable evaluation tailored to diverse user needs, 3) explainability beyond single numerical scores, and 4) scalability across various models and tools. Experiments show that Evaluation Agent reduces evaluation time to 10% of traditional methods while delivering comparable results. The Evaluation Agent framework is fully open-sourced to advance research in visual generative models and their efficient evaluation. |
| 10th December 2024 | [Maya: An Instruction Finetuned Multilingual Multimodal Model](http://arxiv.org/abs/2412.07112v1) | The rapid development of large Vision-Language Models (VLMs) has led to impressive results on academic benchmarks, primarily in widely spoken languages. However, significant gaps remain in the ability of current VLMs to handle low-resource languages and varied cultural contexts, largely due to a lack of high-quality, diverse, and safety-vetted data. Consequently, these models often struggle to understand low-resource languages and cultural nuances in a manner free from toxicity. To address these limitations, we introduce Maya, an open-source Multimodal Multilingual model. Our contributions are threefold: 1) a multilingual image-text pretraining dataset in eight languages, based on the LLaVA pretraining dataset; 2) a thorough analysis of toxicity within the LLaVA dataset, followed by the creation of a novel toxicity-free version across eight languages; and 3) a multilingual image-text model supporting these languages, enhancing cultural and linguistic comprehension in vision-language tasks. Code available at https://github.com/nahidalam/maya. |
| 6th December 2024 | [Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling](http://arxiv.org/abs/2412.05271v3) | We introduce InternVL 2.5, an advanced multimodal large language model (MLLM) series that builds upon InternVL 2.0, maintaining its core model architecture while introducing significant enhancements in training and testing strategies as well as data quality. In this work, we delve into the relationship between model scaling and performance, systematically exploring the performance trends in vision encoders, language models, dataset sizes, and test-time configurations. Through extensive evaluations on a wide range of benchmarks, including multi-discipline reasoning, document understanding, multi-image / video understanding, real-world comprehension, multimodal hallucination detection, visual grounding, multilingual capabilities, and pure language processing, InternVL 2.5 exhibits competitive performance, rivaling leading commercial models such as GPT-4o and Claude-3.5-Sonnet. Notably, our model is the first open-source MLLMs to surpass 70% on the MMMU benchmark, achieving a 3.7-point improvement through Chain-of-Thought (CoT) reasoning and showcasing strong potential for test-time scaling. We hope this model contributes to the open-source community by setting new standards for developing and applying multimodal AI systems. HuggingFace demo see https://huggingface.co/spaces/OpenGVLab/InternVL |
| 6th December 2024 | [Evaluating and Aligning CodeLLMs on Human Preference](http://arxiv.org/abs/2412.05210v1) | Code large language models (codeLLMs) have made significant strides in code generation. Most previous code-related benchmarks, which consist of various programming exercises along with the corresponding test cases, are used as a common measure to evaluate the performance and capabilities of code LLMs. However, the current code LLMs focus on synthesizing the correct code snippet, ignoring the alignment with human preferences, where the query should be sampled from the practical application scenarios and the model-generated responses should satisfy the human preference. To bridge the gap between the model-generated response and human preference, we present a rigorous human-curated benchmark CodeArena to emulate the complexity and diversity of real-world coding tasks, where 397 high-quality samples spanning 40 categories and 44 programming languages, carefully curated from user queries. Further, we propose a diverse synthetic instruction corpus SynCode-Instruct (nearly 20B tokens) by scaling instructions from the website to verify the effectiveness of the large-scale synthetic instruction fine-tuning, where Qwen2.5-SynCoder totally trained on synthetic instruction data can achieve top-tier performance of open-source code LLMs. The results find performance differences between execution-based benchmarks and CodeArena. Our systematic experiments of CodeArena on 40+ LLMs reveal a notable performance gap between open SOTA code LLMs (e.g. Qwen2.5-Coder) and proprietary LLMs (e.g., OpenAI o1), underscoring the importance of the human preference alignment.\footnote{\url{https://codearenaeval.github.io/ }} |
| 5th December 2024 | [BigDocs: An Open and Permissively-Licensed Dataset for Training Multimodal Models on Document and Code Tasks](http://arxiv.org/abs/2412.04626v1) | Multimodal AI has the potential to significantly enhance document-understanding tasks, such as processing receipts, understanding workflows, extracting data from documents, and summarizing reports. Code generation tasks that require long-structured outputs can also be enhanced by multimodality. Despite this, their use in commercial applications is often limited due to limited access to training data and restrictive licensing, which hinders open access. To address these limitations, we introduce BigDocs-7.5M, a high-quality, open-access dataset comprising 7.5 million multimodal documents across 30 tasks. We use an efficient data curation process to ensure our data is high-quality and license-permissive. Our process emphasizes accountability, responsibility, and transparency through filtering rules, traceable metadata, and careful content analysis. Additionally, we introduce BigDocs-Bench, a benchmark suite with 10 novel tasks where we create datasets that reflect real-world use cases involving reasoning over Graphical User Interfaces (GUI) and code generation from images. Our experiments show that training with BigDocs-Bench improves average performance up to 25.8% over closed-source GPT-4o in document reasoning and structured output tasks such as Screenshot2HTML or Image2Latex generation. Finally, human evaluations showed a preference for outputs from models trained on BigDocs over GPT-4o. This suggests that BigDocs can help both academics and the open-source community utilize and improve AI tools to enhance multimodal capabilities and document reasoning. The project is hosted at https://bigdocs.github.io . |
| 5th December 2024 | [NVILA: Efficient Frontier Visual Language Models](http://arxiv.org/abs/2412.04468v1) | Visual language models (VLMs) have made significant advances in accuracy in recent years. However, their efficiency has received much less attention. This paper introduces NVILA, a family of open VLMs designed to optimize both efficiency and accuracy. Building on top of VILA, we improve its model architecture by first scaling up the spatial and temporal resolutions, and then compressing visual tokens. This "scale-then-compress" approach enables NVILA to efficiently process high-resolution images and long videos. We also conduct a systematic investigation to enhance the efficiency of NVILA throughout its entire lifecycle, from training and fine-tuning to deployment. NVILA matches or surpasses the accuracy of many leading open and proprietary VLMs across a wide range of image and video benchmarks. At the same time, it reduces training costs by 4.5X, fine-tuning memory usage by 3.4X, pre-filling latency by 1.6-2.2X, and decoding latency by 1.2-2.8X. We will soon make our code and models available to facilitate reproducibility. |
| 5th December 2024 | [VisionZip: Longer is Better but Not Necessary in Vision Language Models](http://arxiv.org/abs/2412.04467v1) | Recent advancements in vision-language models have enhanced performance by increasing the length of visual tokens, making them much longer than text tokens and significantly raising computational costs. However, we observe that the visual tokens generated by popular vision encoders, such as CLIP and SigLIP, contain significant redundancy. To address this, we introduce VisionZip, a simple yet effective method that selects a set of informative tokens for input to the language model, reducing visual token redundancy and improving efficiency while maintaining model performance. The proposed VisionZip can be widely applied to image and video understanding tasks and is well-suited for multi-turn dialogues in real-world scenarios, where previous methods tend to underperform. Experimental results show that VisionZip outperforms the previous state-of-the-art method by at least 5% performance gains across nearly all settings. Moreover, our method significantly enhances model inference speed, improving the prefilling time by 8x and enabling the LLaVA-Next 13B model to infer faster than the LLaVA-Next 7B model while achieving better results. Furthermore, we analyze the causes of this redundancy and encourage the community to focus on extracting better visual features rather than merely increasing token length. Our code is available at https://github.com/dvlab-research/VisionZip . |
| 4th December 2024 | [Evaluating Language Models as Synthetic Data Generators](http://arxiv.org/abs/2412.03679v1) | Given the increasing use of synthetic data in language model (LM) post-training, an LM's ability to generate high-quality data has become nearly as crucial as its ability to solve problems directly. While prior works have focused on developing effective data generation methods, they lack systematic comparison of different LMs as data generators in a unified setting. To address this gap, we propose AgoraBench, a benchmark that provides standardized settings and metrics to evaluate LMs' data generation abilities. Through synthesizing 1.26 million training instances using 6 LMs and training 99 student models, we uncover key insights about LMs' data generation capabilities. First, we observe that LMs exhibit distinct strengths. For instance, GPT-4o excels at generating new problems, while Claude-3.5-Sonnet performs better at enhancing existing ones. Furthermore, our analysis reveals that an LM's data generation ability doesn't necessarily correlate with its problem-solving ability. Instead, multiple intrinsic features of data quality-including response quality, perplexity, and instruction difficulty-collectively serve as better indicators. Finally, we demonstrate that strategic choices in output format and cost-conscious model selection significantly impact data generation effectiveness. |
| 4th December 2024 | [PaliGemma 2: A Family of Versatile VLMs for Transfer](http://arxiv.org/abs/2412.03555v1) | PaliGemma 2 is an upgrade of the PaliGemma open Vision-Language Model (VLM) based on the Gemma 2 family of language models. We combine the SigLIP-So400m vision encoder that was also used by PaliGemma with the whole range of Gemma 2 models, from the 2B one all the way up to the 27B model. We train these models at three resolutions (224px, 448px, and 896px) in multiple stages to equip them with broad knowledge for transfer via fine-tuning. The resulting family of base models covering different model sizes and resolutions allows us to investigate factors impacting transfer performance (such as learning rate) and to analyze the interplay between the type of task, model size, and resolution. We further increase the number and breadth of transfer tasks beyond the scope of PaliGemma including different OCR-related tasks such as table structure recognition, molecular structure recognition, music score recognition, as well as long fine-grained captioning and radiography report generation, on which PaliGemma 2 obtains state-of-the-art results. |
| 4th December 2024 | [Surveying the Effects of Quality, Diversity, and Complexity in Synthetic Data From Large Language Models](http://arxiv.org/abs/2412.02980v2) | Synthetic data generation with Large Language Models is a promising paradigm for augmenting natural data over a nearly infinite range of tasks. Given this variety, direct comparisons among synthetic data generation algorithms are scarce, making it difficult to understand where improvement comes from and what bottlenecks exist. We propose to evaluate algorithms via the makeup of synthetic data generated by each algorithm in terms of data quality, diversity, and complexity. We choose these three characteristics for their significance in open-ended processes and the impact each has on the capabilities of downstream models. We find quality to be essential for in-distribution model generalization, diversity to be essential for out-of-distribution generalization, and complexity to be beneficial for both. Further, we emphasize the existence of Quality-Diversity trade-offs in training data and the downstream effects on model performance. We then examine the effect of various components in the synthetic data pipeline on each data characteristic. This examination allows us to taxonomize and compare synthetic data generation algorithms through the components they utilize and the resulting effects on data QDC composition. This analysis extends into a discussion on the importance of balancing QDC in synthetic data for efficient reinforcement learning and self-improvement algorithms. Analogous to the QD trade-offs in training data, often there exist trade-offs between model output quality and output diversity which impact the composition of synthetic data. We observe that many models are currently evaluated and optimized only for output quality, thereby limiting output diversity and the potential for self-improvement. We argue that balancing these trade-offs is essential to the development of future self-improvement algorithms and highlight a number of works making progress in this direction. |
| 3rd December 2024 | [VideoGen-of-Thought: A Collaborative Framework for Multi-Shot Video Generation](http://arxiv.org/abs/2412.02259v1) | Current video generation models excel at generating short clips but still struggle with creating multi-shot, movie-like videos. Existing models trained on large-scale data on the back of rich computational resources are unsurprisingly inadequate for maintaining a logical storyline and visual consistency across multiple shots of a cohesive script since they are often trained with a single-shot objective. To this end, we propose VideoGen-of-Thought (VGoT), a collaborative and training-free architecture designed specifically for multi-shot video generation. VGoT is designed with three goals in mind as follows. Multi-Shot Video Generation: We divide the video generation process into a structured, modular sequence, including (1) Script Generation, which translates a curt story into detailed prompts for each shot; (2) Keyframe Generation, responsible for creating visually consistent keyframes faithful to character portrayals; and (3) Shot-Level Video Generation, which transforms information from scripts and keyframes into shots; (4) Smoothing Mechanism that ensures a consistent multi-shot output. Reasonable Narrative Design: Inspired by cinematic scriptwriting, our prompt generation approach spans five key domains, ensuring logical consistency, character development, and narrative flow across the entire video. Cross-Shot Consistency: We ensure temporal and identity consistency by leveraging identity-preserving (IP) embeddings across shots, which are automatically created from the narrative. Additionally, we incorporate a cross-shot smoothing mechanism, which integrates a reset boundary that effectively combines latent features from adjacent shots, resulting in smooth transitions and maintaining visual coherence throughout the video. Our experiments demonstrate that VGoT surpasses existing video generation methods in producing high-quality, coherent, multi-shot videos. |
| 3rd December 2024 | [Personalized Multimodal Large Language Models: A Survey](http://arxiv.org/abs/2412.02142v1) | Multimodal Large Language Models (MLLMs) have become increasingly important due to their state-of-the-art performance and ability to integrate multiple data modalities, such as text, images, and audio, to perform complex tasks with high accuracy. This paper presents a comprehensive survey on personalized multimodal large language models, focusing on their architecture, training methods, and applications. We propose an intuitive taxonomy for categorizing the techniques used to personalize MLLMs to individual users, and discuss the techniques accordingly. Furthermore, we discuss how such techniques can be combined or adapted when appropriate, highlighting their advantages and underlying rationale. We also provide a succinct summary of personalization tasks investigated in existing research, along with the evaluation metrics commonly used. Additionally, we summarize the datasets that are useful for benchmarking personalized MLLMs. Finally, we outline critical open challenges. This survey aims to serve as a valuable resource for researchers and practitioners seeking to understand and advance the development of personalized multimodal large language models. |
| 2nd December 2024 | [MALT: Improving Reasoning with Multi-Agent LLM Training](http://arxiv.org/abs/2412.01928v1) | Enabling effective collaboration among LLMs is a crucial step toward developing autonomous systems capable of solving complex problems. While LLMs are typically used as single-model generators, where humans critique and refine their outputs, the potential for jointly-trained collaborative models remains largely unexplored. Despite promising results in multi-agent communication and debate settings, little progress has been made in training models to work together on tasks. In this paper, we present a first step toward "Multi-agent LLM training" (MALT) on reasoning problems. Our approach employs a sequential multi-agent setup with heterogeneous LLMs assigned specialized roles: a generator, verifier, and refinement model iteratively solving problems. We propose a trajectory-expansion-based synthetic data generation process and a credit assignment strategy driven by joint outcome based rewards. This enables our post-training setup to utilize both positive and negative trajectories to autonomously improve each model's specialized capabilities as part of a joint sequential system. We evaluate our approach across MATH, GSM8k, and CQA, where MALT on Llama 3.1 8B models achieves relative improvements of 14.14%, 7.12%, and 9.40% respectively over the same baseline model. This demonstrates an early advance in multi-agent cooperative capabilities for performance on mathematical and common sense reasoning questions. More generally, our work provides a concrete direction for research around multi-agent LLM training approaches. |
| 2nd December 2024 | [Yi-Lightning Technical Report](http://arxiv.org/abs/2412.01253v4) | This technical report presents Yi-Lightning, our latest flagship large language model (LLM). It achieves exceptional performance, ranking 6th overall on Chatbot Arena, with particularly strong results (2nd to 4th place) in specialized categories including Chinese, Math, Coding, and Hard Prompts. Yi-Lightning leverages an enhanced Mixture-of-Experts (MoE) architecture, featuring advanced expert segmentation and routing mechanisms coupled with optimized KV-caching techniques. Our development process encompasses comprehensive pre-training, supervised fine-tuning (SFT), and reinforcement learning from human feedback (RLHF), where we devise deliberate strategies for multi-stage training, synthetic data construction, and reward modeling. Furthermore, we implement RAISE (Responsible AI Safety Engine), a four-component framework to address safety issues across pre-training, post-training, and serving phases. Empowered by our scalable super-computing infrastructure, all these innovations substantially reduce training, deployment and inference costs while maintaining high-performance standards. With further evaluations on public academic benchmarks, Yi-Lightning demonstrates competitive performance against top-tier LLMs, while we observe a notable disparity between traditional, static benchmark results and real-world, dynamic human preferences. This observation prompts a critical reassessment of conventional benchmarks' utility in guiding the development of more intelligent and powerful AI systems for practical applications. Yi-Lightning is now available through our developer platform at https://platform.lingyiwanwu.com. |
| 29th November 2024 | [Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability](http://arxiv.org/abs/2411.19943v2) | Large Language Models (LLMs) have exhibited remarkable performance on reasoning tasks. They utilize autoregressive token generation to construct reasoning trajectories, enabling the development of a coherent chain of thought. In this work, we explore the impact of individual tokens on the final outcomes of reasoning tasks. We identify the existence of ``critical tokens'' that lead to incorrect reasoning trajectories in LLMs. Specifically, we find that LLMs tend to produce positive outcomes when forced to decode other tokens instead of critical tokens. Motivated by this observation, we propose a novel approach - cDPO - designed to automatically recognize and conduct token-level rewards for the critical tokens during the alignment process. Specifically, we develop a contrastive estimation approach to automatically identify critical tokens. It is achieved by comparing the generation likelihood of positive and negative models. To achieve this, we separately fine-tune the positive and negative models on various reasoning trajectories, consequently, they are capable of identifying identify critical tokens within incorrect trajectories that contribute to erroneous outcomes. Moreover, to further align the model with the critical token information during the alignment process, we extend the conventional DPO algorithms to token-level DPO and utilize the differential likelihood from the aforementioned positive and negative model as important weight for token-level DPO learning.Experimental results on GSM8K and MATH500 benchmarks with two-widely used models Llama-3 (8B and 70B) and deepseek-math (7B) demonstrate the effectiveness of the propsoed approach cDPO. |
| 29th November 2024 | [o1-Coder: an o1 Replication for Coding](http://arxiv.org/abs/2412.00154v2) | The technical report introduces O1-CODER, an attempt to replicate OpenAI's o1 model with a focus on coding tasks. It integrates reinforcement learning (RL) and Monte Carlo Tree Search (MCTS) to enhance the model's System-2 thinking capabilities. The framework includes training a Test Case Generator (TCG) for standardized code testing, using MCTS to generate code data with reasoning processes, and iteratively fine-tuning the policy model to initially produce pseudocode and then generate the full code. The report also addresses the opportunities and challenges in deploying o1-like models in real-world applications, suggesting transitioning to the System-2 paradigm and highlighting the imperative for world model construction. Updated model progress and experimental results will be reported in subsequent versions. All source code, curated datasets, as well as the derived models are disclosed at https://github.com/ADaM-BJTU/O1-CODER . |
| 28th November 2024 | [Open-Sora Plan: Open-Source Large Video Generation Model](http://arxiv.org/abs/2412.00131v1) | We introduce Open-Sora Plan, an open-source project that aims to contribute a large generation model for generating desired high-resolution videos with long durations based on various user inputs. Our project comprises multiple components for the entire video generation process, including a Wavelet-Flow Variational Autoencoder, a Joint Image-Video Skiparse Denoiser, and various condition controllers. Moreover, many assistant strategies for efficient training and inference are designed, and a multi-dimensional data curation pipeline is proposed for obtaining desired high-quality data. Benefiting from efficient thoughts, our Open-Sora Plan achieves impressive video generation results in both qualitative and quantitative evaluations. We hope our careful design and practical experience can inspire the video generation research community. All our codes and model weights are publicly available at \url{https://github.com/PKU-YuanGroup/Open-Sora-Plan}. |




---

## :mortar_board: Courses

#### [Ongoing] Applied LLMs Mastery 2024

Join 1000+ students on this 10-week adventure as we delve into the application of LLMs across a variety of use cases

#### [Link](https://areganti.notion.site/Applied-LLMs-Mastery-2024-562ddaa27791463e9a1286199325045c) to the course website

##### [Feb 2024] Registrations are still open [click here](https://forms.gle/353sQMRvS951jDYu7) to register

🗓️\*Week 1 [Jan 15 2024]**\*: [Practical Introduction to LLMs](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week1_part1_foundations.md)**

- Applied LLM Foundations
- Real World LLM Use Cases
- Domain and Task Adaptation Methods

🗓️\*Week 2 [Jan 22 2024]**\*: [Prompting and Prompt
Engineering](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week2_prompting.md)**

- Basic Prompting Principles
- Types of Prompting
- Applications, Risks and Advanced Prompting

🗓️\*Week 3 [Jan 29 2024]**\*: [LLM Fine-tuning](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week3_finetuning_llms.md)**

- Basics of Fine-Tuning
- Types of Fine-Tuning
- Fine-Tuning Challenges

🗓️\*Week 4 [Feb 5 2024]**\*: [RAG (Retrieval-Augmented Generation)](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week4_RAG.md)**

- Understanding the concept of RAG in LLMs
- Key components of RAG
- Advanced RAG Methods

🗓️\*Week 5 [ Feb 12 2024]**\*: [Tools for building LLM Apps](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week5_tools_for_LLM_apps.md)**

- Fine-tuning Tools
- RAG Tools
- Tools for observability, prompting, serving, vector search etc.

🗓️\*Week 6 [Feb 19 2024]**\*: [Evaluation Techniques](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week6_llm_evaluation.md)**

- Types of Evaluation
- Common Evaluation Benchmarks
- Common Metrics

🗓️\*Week 7 [Feb 26 2024]**\*: [Building Your Own LLM Application](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week7_build_llm_app.md)**

- Components of LLM application
- Build your own LLM App end to end

🗓️\*Week 8 [March 4 2024]**\*: [Advanced Features and Deployment](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week8_advanced_features.md)**

- LLM lifecycle and LLMOps
- LLM Monitoring and Observability
- Deployment strategies

🗓️\*Week 9 [March 11 2024]**\*: [Challenges with LLMs](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week9_challenges_with_llms.md)**

- Scaling Challenges
- Behavioral Challenges
- Future directions

🗓️\*Week 10 [March 18 2024]**\*: [Emerging Research Trends](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week10_research_trends.md)**

- Smaller and more performant models
- Multimodal models
- LLM Alignment

🗓️*Week 11 *Bonus\* [March 25 2024]**\*: [Foundations](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/Applied_LLMs_Mastery_2024/week11_foundations.md)**

- Generative Models Foundations
- Self-Attention and Transformers
- Neural Networks for Language

---

#### :book: List of Free GenAI Courses

##### LLM Basics and Foundations

1. [Large Language Models](https://rycolab.io/classes/llm-s23/) by ETH Zurich

2. [Understanding Large Language Models](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/) by Princeton

3. [Transformers course](https://huggingface.co/learn/nlp-course/chapter1/1) by Huggingface

4. [NLP course](https://huggingface.co/learn/nlp-course/chapter1/1) by Huggingface

5. [CS324 - Large Language Models](https://stanford-cs324.github.io/winter2022/) by Stanford

6. [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms) by Coursera

7. [Introduction to Generative AI](https://www.coursera.org/learn/introduction-to-generative-ai) by Coursera

8. [Generative AI Fundamentals](https://www.cloudskillsboost.google/paths/118/course_templates/556) by Google Cloud
9. [5-Day Gen AI Intensive Course](https://www.youtube.com/watch?v=kpRyiJUUFxY&list=PLqFaTIg4myu-b1PlxitQdY0UYIbys-2es) by Google & Kaggle

10. [Introduction to Large Language Models](https://www.cloudskillsboost.google/paths/118/course_templates/539) by Google Cloud
11. [Introduction to Generative AI](https://www.cloudskillsboost.google/paths/118/course_templates/536) by Google Cloud
12. [Generative AI Concepts](https://www.datacamp.com/courses/generative-ai-concepts) by DataCamp (Daniel Tedesco Data Lead @ Google)
13. [1 Hour Introduction to LLM (Large Language Models)](https://www.youtube.com/watch?v=xu5_kka-suc) by WeCloudData
14. [LLM Foundation Models from the Ground Up | Primer](https://www.youtube.com/watch?v=W0c7jQezTDw&list=PLTPXxbhUt-YWjMCDahwdVye8HW69p5NYS) by Databricks
15. [Generative AI Explained](https://courses.nvidia.com/courses/course-v1:DLI+S-FX-07+V1/) by Nvidia
16. [Transformer Models and BERT Model](https://www.cloudskillsboost.google/course_templates/538) by Google Cloud
17. [Generative AI Learning Plan for Decision Makers](https://explore.skillbuilder.aws/learn/public/learning_plan/view/1909/generative-ai-learning-plan-for-decision-makers) by AWS
18. [Introduction to Responsible AI](https://www.cloudskillsboost.google/course_templates/554) by Google Cloud
19. [Fundamentals of Generative AI](https://learn.microsoft.com/en-us/training/modules/fundamentals-generative-ai/) by Microsoft Azure
20. [Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-122979-leestott) by Microsoft
21. [ChatGPT for Beginners: The Ultimate Use Cases for Everyone](https://www.udemy.com/course/chatgpt-for-beginners-the-ultimate-use-cases-for-everyone/) by Udemy
22. [[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) by Andrej Karpathy
23. [ChatGPT for Everyone](https://learnprompting.org/courses/chatgpt-for-everyone) by Learn Prompting
24. [Large Language Models (LLMs) (In English)](https://www.youtube.com/playlist?list=PLxlkzujLkmQ9vMaqfvqyfvZV_o8EqjAk7) by Kshitiz Verma (JK Lakshmipat University, Jaipur, India)
25. [Generative AI for Beginners](https://codekidz.ai/lesson-intro/generative-a-362093) By CodeKidz, based on Microsoft's open sourced course.

##### Building LLM Applications

1. [LLMOps: Building Real-World Applications With Large Language Models](https://www.udacity.com/course/building-real-world-applications-with-large-language-models--cd13455) by Udacity

2. [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) by FSDL

3. [Generative AI for beginners](https://github.com/microsoft/generative-ai-for-beginners/tree/main) by Microsoft

4. [Large Language Models: Application through Production](https://www.edx.org/learn/computer-science/databricks-large-language-models-application-through-production) by Databricks

5. [Generative AI Foundations](https://www.youtube.com/watch?v=oYm66fHqHUM&list=PLhr1KZpdzukf-xb0lmiU3G89GJXaDbAIF) by AWS

6. [Introduction to Generative AI Community Course](https://www.youtube.com/watch?v=ajWheP8ZD70&list=PLmQAMKHKeLZ-iTT-E2kK9uePrJ1Xua9VL) by ineuron

7. [LLM University](https://docs.cohere.com/docs/llmu) by Cohere
8. [LLM Learning Lab](https://lightning.ai/pages/llm-learning-lab/) by Lightning AI
9. [LangChain for LLM Application Development](https://learn.deeplearning.ai/login?redirect_course=langchain&callbackUrl=https%3A%2F%2Flearn.deeplearning.ai%2Fcourses%2Flangchain) by Deeplearning.AI
10. [LLMOps](https://learn.deeplearning.ai/llmops) by DeepLearning.AI
11. [Automated Testing for LLMOps](https://learn.deeplearning.ai/automated-testing-llmops) by DeepLearning.AI
12. [Building Generative AI Applications Using Amazon Bedrock](https://explore.skillbuilder.aws/learn/course/external/view/elearning/17904/building-generative-ai-applications-using-amazon-bedrock-aws-digital-training) by AWS
13. [Efficiently Serving LLMs](https://learn.deeplearning.ai/courses/efficiently-serving-llms/lesson/1/introduction) by DeepLearning.AI
14. [Building Systems with the ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) by DeepLearning.AI
15. [Serverless LLM apps with Amazon Bedrock](https://www.deeplearning.ai/short-courses/serverless-llm-apps-amazon-bedrock/) by DeepLearning.AI
16. [Building Applications with Vector Databases](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/) by DeepLearning.AI
17. [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/) by DeepLearning.AI
18. [Build LLM Apps with LangChain.js](https://www.deeplearning.ai/short-courses/build-llm-apps-with-langchain-js/) by DeepLearning.AI
19. [Advanced Retrieval for AI with Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) by DeepLearning.AI
20. [Operationalizing LLMs on Azure](https://www.coursera.org/learn/llmops-azure) by Coursera
21. [Generative AI Full Course – Gemini Pro, OpenAI, Llama, Langchain, Pinecone, Vector Databases & More](https://www.youtube.com/watch?v=mEsleV16qdo) by freeCodeCamp.org
22. [Training & Fine-Tuning LLMs for Production](https://learn.activeloop.ai/courses/llms) by Activeloop
    

##### Prompt Engineering, RAG and Fine-Tuning

1. [LangChain & Vector Databases in Production](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVhnQW8xNDdhSU9IUDVLXzFhV2N0UkNRMkZrQXxBQ3Jtc0traUxHMzZJcGJQYjlyckYxaGxYVWlsOFNGUFlFVEdhNzdjTWpPUlQ2TF9XczRqNkxMVGpJTnd5YmYzV0prQ0IwZURNcHhIZ3h1Z051VTl5MXBBLUN0dkM0NHRkQTFua1Jpc0VCRFJUb0ZQZG95b0JqMA&q=https%3A%2F%2Flearn.activeloop.ai%2Fcourses%2Flangchain&v=gKUTDC13jys) by Activeloop

2. [Reinforcement Learning from Human Feedback](https://learn.deeplearning.ai/reinforcement-learning-from-human-feedback) by DeepLearning.AI

3. [Building Applications with Vector Databases](https://learn.deeplearning.ai/building-applications-vector-databases) by DeepLearning.AI

4. [Finetuning Large Language Models](https://learn.deeplearning.ai/finetuning-large-language-models) by Deeplearning.AI
5. [LangChain: Chat with Your Data](http://learn.deeplearning.ai/langchain-chat-with-your-data/) by Deeplearning.AI

6. [Building Systems with the ChatGPT API](https://learn.deeplearning.ai/chatgpt-building-system) by Deeplearning.AI
7. [Prompt Engineering with Llama 2](https://www.deeplearning.ai/short-courses/prompt-engineering-with-llama-2/) by Deeplearning.AI
8. [Building Applications with Vector Databases](https://learn.deeplearning.ai/building-applications-vector-databases) by Deeplearning.AI
9. [ChatGPT Prompt Engineering for Developers](https://learn.deeplearning.ai/chatgpt-prompt-eng/lesson/1/introduction) by Deeplearning.AI
10. [Advanced RAG Orchestration series](https://www.youtube.com/watch?v=CeDS1yvw9E4) by LlamaIndex
11. [Prompt Engineering Specialization](https://www.coursera.org/specializations/prompt-engineering) by Coursera
12. [Augment your LLM Using Retrieval Augmented Generation](https://courses.nvidia.com/courses/course-v1:NVIDIA+S-FX-16+v1/) by Nvidia
13. [Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/) by Deeplearning.AI
14. [Open Source Models with Hugging Face](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/) by Deeplearning.AI
15. [Vector Databases: from Embeddings to Applications](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/) by Deeplearning.AI
16. [Understanding and Applying Text Embeddings](https://www.deeplearning.ai/short-courses/google-cloud-vertex-ai/) by Deeplearning.AI
17. [JavaScript RAG Web Apps with LlamaIndex](https://www.deeplearning.ai/short-courses/javascript-rag-web-apps-with-llamaindex/) by Deeplearning.AI
18. [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/) by Deeplearning.AI
19. [Preprocessing Unstructured Data for LLM Applications](https://www.deeplearning.ai/short-courses/preprocessing-unstructured-data-for-llm-applications/) by Deeplearning.AI
20. [Retrieval Augmented Generation for Production with LangChain & LlamaIndex](https://learn.activeloop.ai/courses/rag) by Activeloop
21. [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/) by Deeplearning.AI

##### Evaluation

1. [Building and Evaluating Advanced RAG Applications](https://learn.deeplearning.ai/building-evaluating-advanced-rag) by DeepLearning.AI
2. [Evaluating and Debugging Generative AI Models Using Weights and Biases](https://learn.deeplearning.ai/evaluating-debugging-generative-ai) by Deeplearning.AI
3. [Quality and Safety for LLM Applications](https://www.deeplearning.ai/short-courses/quality-safety-llm-applications/) by Deeplearning.AI
4. [Red Teaming LLM Applications](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/?utm_campaign=giskard-launch&utm_medium=headband&utm_source=dlai-homepage) by Deeplearning.AI

##### Multimodal

1. [How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) by DeepLearning.AI
2. [How to Use Midjourney, AI Art and ChatGPT to Create an Amazing Website](https://www.youtube.com/watch?v=5wdCev86RYE) by Brad Hussey
3. [Build AI Apps with ChatGPT, DALL-E and GPT-4](https://scrimba.com/learn/buildaiapps) by Scrimba
4. [11-777: Multimodal Machine Learning](https://www.youtube.com/playlist?list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA) by Carnegie Mellon University
5. [Prompt Engineering for Vision Models](https://www.deeplearning.ai/short-courses/prompt-engineering-for-vision-models/) by Deeplearning.AI

##### Agents
1. [Building RAG Agents with LLMs](https://courses.nvidia.com/courses/course-v1:DLI+S-FX-15+V1/) by Nvidia
2. [Functions, Tools and Agents with LangChain](https://learn.deeplearning.ai/functions-tools-agents-langchain) by Deeplearning.AI
3. [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) by Deeplearning.AI
4. [AI Agentic Design Patterns with AutoGen](https://www.deeplearning.ai/short-courses/ai-agentic-design-patterns-with-autogen/) by Deeplearning.AI
5. [Multi AI Agent Systems with crewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) by Deeplearning.AI
6. [Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/) by Deeplearning.AI
7. [LLM Observability: Agents, Tools, and Chains](https://courses.arize.com/p/agents-tools-and-chains) by Arize AI
8. [Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/) by Deeplearning.AI
9. [Agents Tools & Function Calling with Amazon Bedrock (How-to)](https://www.youtube.com/watch?app=desktop&v=2L_XE6g3atI) by AWS Developers
10. [ChatGPT & Zapier: Agentic AI for Everyone](https://www.coursera.org/learn/agentic-ai-chatgpt-zapier) by Coursera
11. [Multi-Agent Systems with AutoGen](https://www.manning.com/books/multi-agent-systems-with-autogen) by Victor Dibia [Book]
12. [Large Language Model Agents MOOC, Fall 2024](https://llmagents-learning.org/f24) by Dawn Song & Xinyun Chen – A comprehensive course covering foundational and advanced topics on LLM agents.
13. [CS294/194-196 Large Language Model Agents](https://rdi.berkeley.edu/llm-agents/f24) by UC Berkeley





#### Miscellaneous

1. [Avoiding AI Harm](https://www.coursera.org/learn/avoiding-ai-harm) by Coursera
2. [Developing AI Policy](https://www.coursera.org/learn/developing-ai-policy) by Coursera

---

## :paperclip: Resources

- [ICLR 2024 Paper Summaries](https://areganti.notion.site/06f0d4fe46a94d62bff2ae001cfec22c?v=d501ca62e4b745768385d698f173ae14)

---

## :computer: Interview Prep

#### Topic wise Questions:

1. [Common GenAI Interview Questions](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/interview_prep/60_gen_ai_questions.md)
2. Prompting and Prompt Engineering
3. Model Fine-Tuning
4. Model Evaluation
5. MLOps for GenAI
6. Generative Models Foundations
7. Latest Research Trends

#### GenAI System Design (Coming Soon):

1. Designing an LLM-Powered Search Engine
2. Building a Customer Support Chatbot
3. Building a system for natural language interaction with your data.
4. Building an AI Co-pilot
5. Designing a Custom Chatbot for Q/A on Multimodal Data (Text, Images, Tables, CSV Files)
6. Building an Automated Product Description and Image Generation System for E-commerce

---

## :notebook: Code Notebooks

#### RAG Tutorials

- [AWS Bedrock Workshop Tutorials](https://github.com/aws-samples/amazon-bedrock-workshop) by Amazon Web Services
- [Langchain Tutorials](https://github.com/gkamradt/langchain-tutorials) by gkamradt
- [LLM Applications for production](https://github.com/ray-project/llm-applications/tree/main) by ray-project
- [LLM tutorials](https://github.com/ollama/ollama/tree/main/examples) by Ollama
- [LLM Hub](https://github.com/mallahyari/llm-hub) by mallahyari
- [RAG cookbook](https://docs.camel-ai.org/cookbooks/agents_with_rag.html) by CAMEL-AI

#### Fine-Tuning Tutorials

- [LLM Fine-tuning tutorials](https://github.com/ashishpatel26/LLM-Finetuning) by ashishpatel26
- [PEFT](https://github.com/huggingface/peft/tree/main/examples) example notebooks by Huggingface
- [Free LLM Fine-Tuning Notebooks](https://levelup.gitconnected.com/14-free-large-language-models-fine-tuning-notebooks-532055717cb7) by Youssef Hosni


#### Comprehensive LLM Code Repositories 
- [LLM-PlayLab](https://github.com/Sakil786/LLM-PlayLab) This playlab encompasses a multitude of projects crafted through the utilization of Transformer Models


---

## :black_nib: Contributing

If you want to add to the repository or find any issues, please feel free to raise a PR and ensure correct placement within the relevant section or category.

---

## :pushpin: Cite Us

To cite this guide, use the below format:

```
@article{areganti_generative_ai_guide,
author = {Reganti, Aishwarya Naresh},
journal = {https://github.com/aishwaryanr/awesome-generative-ai-resources},
month = {01},
title = {{Generative AI Guide}},
year = {2024}
}
```

## License

[MIT License]
