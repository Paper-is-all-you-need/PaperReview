# Rog paper summary

작성자 : 정지원

## REASONING ON GRAPHS: FAITHFUL AND INTERPRETABLE LARGE LANGUAGE MODEL REASONING

- [https://arxiv.org/pdf/2310.01061.pdf](https://arxiv.org/pdf/2310.01061.pdf)

### Abstract

대형 언어 모델(LLM)은 복잡한 작업에서 인상적인 추론 능력을 보여주었습니다. 그러나 그들은 **최신 지식이 부족하고 추론 중 경험 환영을 보여주어 잘못된 추론 과정으로 이어질 수 있으며, 이는 그들의 성능과 신뢰성을 저하시킬 수 있습니다**. 지식 그래프(KG)는 구조화된 형식으로 방대한 양의 사실을 포착하여 추론에 신뢰할 수 있는 지식 원천을 제공합니다. 그러나 기존의 KG 기반 LLM 추론 방법은 KG를 사실적인 지식 기반으로만 취급하고 추론에 대한 **구조적 정보의 중요성을 간과**합니다. 본 논문에서는 LLM과 KG를 협력하여 충실하고 해석 가능한 추론을 가능하게 하는 새로운 방법인 그래프 추론(RoG)을 제안합니다. 구체적으로, 우리는 계획-검색-추론 프레임워크를 제시하여 RoG가 먼저 KG에 근거한 관계 경로를 충실한 계획으로 생성합니다. 이러한 계획은 그런 다음 LLM이 충실한 추론을 수행하기 위해 KG에서 유효한 추론 경로를 검색하는 데 사용됩니다. 더 나아가, RoG는 LLM의 추론 능력을 향상시키기 위해 KG에서 지식을 정제하는 것뿐만 아니라 추론 중에 임의의 LLM과의 원활한 통합도 허용합니다. 두 개의 벤치마크 KGQA 데이터셋에서의 광범위한 실험 결과 RoG가 KG 추론 작업에서 최신 기술성을 달성하고 충실하고 해석 가능한 추론 결과를 생성한다는 것을 보여줍니다.

### Introduction

LLM은 많은 NLP task에서 좋은 성능을 보여주고 있다. 특히 놀랍다고 할 만한 점은 그들의 추론을 통해 복잡한 작업을 처리할 수 있는 능력입니다.

하지만 이러한 능력에도 불구하고, LLMs는 추론하는데 있어서 여전히 지식의 부족과 hallucination에 빠지기 쉬운것을 보여준다.

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/ae164a9c-e329-4946-bf14-ae5830b0cdac)


위 그림처럼 지식이 부족하거나(lack of knowledge) hallucination이 LLM에서 추론할 때 종종 발생한다.

이 문제를 다루기 위해, LLM의 추론 능력을 높이고자 KG를 합치는 연구가 나오고 있다.

Previous works that jointly use KGs and LLMs for KGQA reasoning can be broadly divided into two categories : 

1. Semantic parsing method
- which use LLMs to convert questions into logical queries that are executed on KGs to obtain answers
- 단점 : can often be non-executable and yield no answers, due to syntax and semantic limitations
1. Retrieval-augmented method
- which retrieve triples from KGs as knowledge context and uses LLMs to obtain the final answers
- 단점 : overlook the importance of their structural information for reasoning.

To **address the issues of hallucinations and lack of knowledge**, we present a **planning-retrieval-reasoning framework**, where RoG first generates relation paths grounded by KGs as faithful plans via the planning module,.

These plans are then used to retrieve valid reasoning paths from KGs to conduct faithful reasoning by the retrieval-reasoning module.

### Preliminary

![Untitled](Rog%20paper%20summary%2025016068e1834773922ab19458995bc5/Untitled%201.png)

### Method

In this section, we introduce our method: reasoning on graphs (RoG). We present a novel planning-retrieval-reasoning framework that synergizes LLMs and KGs to conduct faithful and interpretable
reasoning for KGQA.

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/d9be4eff-1045-4d73-9252-de15742db9c0)


### Reasoning on Graphs : Planning-Retrieval-Reasoning

we present a novel planning-retrieval-reasoning framework, which makes the reasoning plans grounded
by KGs and then retrieves faithful reasoning paths for LLM reasoning.

entity대신 relation path를 선택한 이유 : 엔티티는 동적으로 업데이트가 되지만 KG에서 relation은 더 안정적이기 때문이다.

By using relation paths, we can always retrieve the latest knowledge from KGs for reasoning.

Therefore, relation paths can serve as faithful plans for reasoning the answer to KGQA task.

**By treating relation paths as plans**, we can make sure the plans are grounded by KGs, which enables LLMs to conduct faithful and interpretable reasoning on graphs.

간단히 말해서, we formulate our *RoG* as an optimization problem that aims to maximize the probability of reasoning the answer from a knowledge graph $G$ w.r.t the question $q$ by generating relation paths $z$ as the plan:

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/f2866a4e-28e5-4017-b51c-efeecf622409)


![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/07fcbef1-bc5f-4ab5-b571-441f46b44de7)


### Optimization framework

Despite the advantage of generating relation paths as plans, the LLMs have zero knowledge of the
relations contained in KGs. Therefore, **LLMs cannot directly generate relation paths grounded by
KGs as faithful plans.** Moreover, **LLMs might not understand the reasoning paths correctly and
conduct reasoning based on them**. To address these issues, we design **two instruction tuning tasks:**

1. **planning optimization**, which distills the knowledge from KGs into LLMs to generate faithful
relation paths as plans; 
2. **retrieval-reasoning optimization**, which enables LLMs to reason based on
the retrieved reasoning paths.

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/b37d95f2-ba11-4f51-ab66-511f1b18ff5d)


위 식을 ELBO로 최적화를 진행하면 다음과 같다 : 

<img width="579" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/692c5bee-7c04-415c-9962-365207a59107">


where $Q(z)$ denotes the **posterior distribution of faithful relation paths grounded by KGs**. The
latter term minimizes the KL divergence between the posterior and the prior, which encourages
LLMs to generate faithful relation paths (planning optimization). The former term maximizes the
expectation that retrieval-reasoning module generates correct answers based on the relation paths
and KGs (retrieval-reasoning optimization).

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/068c438b-c36b-4446-bd22-1e394c10de77)


위 내용을 실질적으로 어떻게 implementation하는지 알아보자

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/adfb5e5e-13e8-4140-becb-76309e9786be)


![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/76f7f001-e469-4f16-89fc-58ba42a5ff7c)
