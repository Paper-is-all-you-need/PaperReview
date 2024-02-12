# Paper Summary

## 1. TALK LIKE A GRAPH: ENCODING GRAPHS FOR LARGE LANGUAGE MODELS

[https://arxiv.org/pdf/2310.04560.pdf](https://arxiv.org/pdf/2310.04560.pdf)

### Abstract

그래프는 소셜 네트워크, 추천 시스템 및 계산 금융과 같은 현실 세계 응용 프로그램에서 복잡한 관계를 표현하고 분석하는 강력한 도구입니다. 그래프 추론은 복잡한 시스템 내의 엔티티 간의 관계를 추론하고 숨겨진 패턴과 동향을 식별하기 위해 필수적입니다. 자연어 텍스트를 사용한 자동 추론에 있어서 놀라운 진전이 있었지만, 대규모 언어 모델(Large Language Models, LLMs)을 사용한 그래프 추론은 아직 연구가 충분히 이루어지지 않은 문제입니다. 이 연구에서는 그래프 구조로 된 데이터를 LLMs가 소비할 수 있는 텍스트로 인코딩하는 첫 종합적인 연구를 수행합니다. 우리는 LLM의 그래프 추론 작업에서 성능이 다음 세 가지 기본 수준에서 다양하게 변하는 것을 보여줍니다: (1) 그래프 인코딩 방법, (2) 그래프 작업의 특성 및 (3) 흥미로운 점은 그래프의 자체 구조입니다. 이러한 새로운 결과들은 그래프를 텍스트로 인코딩하는 전략에 대한 중요한 통찰력을 제공합니다. 이러한 통찰력을 사용하여 인코더의 올바른 선택이 그래프 추론 작업의 성능을 개선할 수 있음을 보여주며, 작업에 따라 4.8%에서 61.8%까지 성능을 향상시킬 수 있음을 설명합니다.

### Introduction

introduction에서 핵심 text, 문구를 찾아보자.

However, despite all their success, there are a number of limitations with the current methodology of design and implementation of LLMs. 

이렇게 한계가 있는데 그 중 하나는 unstructured text에 의존해서 학습했기 때문에 논리적인 문제와 hallucination 문제가 발생한다는 것이다. 이를 해결하기 위해 graph를 LLM과 함께 사용하면 되는데 아직 이분야는 연구가 되지 않았다고 한다.

The intersection of graphs and LLMs has been relatively understudied. 굳이 있다면 지식 그래프를 사용한다는 정도 뿐이지, graph-structured data의 일반적인 목적으로 사용하는 분야는 연구 되지 않고 있다.

In this work, we perform the first comprehensive study about reasoning over graph-structured data as text for consumption by LLMs.

To analyze graph reasoning more closely, we decompose the problem into **graph encoding** and **graph prompt engineering.**

아래 그림은 graph encoder의 방법이다. 즉, graph encoder == graph to text

<img width="778" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/0a751370-1a4b-4e87-8275-01e6f73041c5">


### Contributions

1. An extensive study of graph-structure prompting techniques for use in LLMs.
2. Insights and best practices for encoding graphs as text for use in LLMs.
3. A new graph benchmark(GraphQA) to aid the community in studying the effects of graph structure on LLM prompting further.

<img width="741" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/16b4cbfd-22a4-4a6a-8b33-bbf9d2869b1b">


### **What to get**

이 논문에서 내가 얻어갈 수 있는 것은 graph encoder function g인것같다. 이것은 graph 정보를 text로 바꾸는 함수로써, 이것은 우리가 원하는 텍스트에서 구조화된 데이터를 도출하기 위해 지식 그래프를 활용할 수 있는 핵심 정보로 작용할 것으로 생각된다.

---

## 2. Unifying Large Language Models and Knowledge Graphs : A Roadmap




### Abstract

대형 언어 모델 (LLMs)인 ChatGPT와 GPT4와 같은 모델은 새로운 물결을 자연어 처리 및 인공 지능 분야에 일으키고 있으며, 이는 그들의 신생 능력과 일반성 때문입니다. 그러나 LLMs는 종종 사실적인 지식을 포착하고 접근하는 데 한계가 있는 블랙박스 모델입니다. 반면, 지식 그래프 (Knowledge Graphs, KGs)인 Wikipedia와 Huapu와 같은 구조화된 지식 모델은 풍부한 사실적인 지식을 명시적으로 저장하는 모델입니다. KGs는 추론과 해석 가능성을 위해 외부 지식을 제공함으로써 LLMs를 향상시킬 수 있습니다. 한편, KGs는 본질적으로 구축하고 발전하기 어려워서, 기존 KGs의 방법들이 새로운 사실을 생성하고 보이지 않는 지식을 나타내는 것에 도전하고 있습니다. 따라서 LLMs와 KGs를 통합하고 동시에 그들의 이점을 활용하는 것이 보완적입니다. 이 글에서는 LLMs와 KGs의 통합을 위한 전망적인 로드맵을 제시합니다. 우리의 로드맵은 세 가지 일반적인 프레임워크로 구성되어 있으며, 1) KG를 강화하는 LLMs, 이 프레임워크는 LLMs의 사전 훈련 및 추론 단계에서 KG를 통합하거나 LLMs가 학습한 지식을 이해하기 위한 목적으로 KG를 활용합니다; 2) LLMs를 보완하는 KGs, 이 프레임워크는 임베딩, 완성, 구축, 그래프에서 텍스트 생성 및 질문 응답과 같은 다양한 KG 작업에 LLMs를 활용합니다; 3) Synergized LLMs + KGs, 이 프레임워크에서 LLMs와 KGs는 동등한 역할을 하며 데이터와 지식 모두에 의해 주도되는 양방향 추론을 위해 상호적으로 작동하여 LLMs와 KGs를 모두 향상시킵니다. 우리는 로드맵 내에서 이 세 가지 프레임워크에서의 기존 노력을 검토하고 요약하며, 그들의 미래 연구 방향을 지적합니다.

### Introduction

Despite their success in many applications, LLMs have been criticized for their **lack of factual knowledge.**

LLMs의 한계점은 다음과 같다.

1. LLMs represent knowledge **implicitly** in their parammeters.
2. LLMs perform reasoning by a probability model, which is an indecisive process

위와 같은 문제를 다루기 위해, 잠재적인 방법으로 incorporate knowledge graphs into LLMs.

KGs are crucial for various applications as they offer accurate explicit knowledge.

하지만, KG도 문제가 있다.

1. KGs are difficult to construct, and current approaches in KGs are inadequate in handling the incomplete and dynamically changing nature of real-world KGs.
2. They often ignore the abundant textual information in KGs.
3. Existing methods in KGs are often customized for specific KGs or tasks, which are not generalizable enough.

따라서, KGs와 LLM은 서로 상호 보완적으로 도와야 한다. 

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/437a1a61-734e-42ea-81f3-a903f9d1dbd0)


이 논문은 LLMs와 KGs를 통합하여 각 접근 방식의 장점을 활용하고 각 접근 방식의 한계를 극복하기 위한 전망적인 로드맵을 제시한다. 다양한 하류 작업을 위해 세부적인 분류를 제안하고 포괄적인 리뷰를 실시하며, 이 빠르게 성장하는 분야에서 떠오르는 방향을 강조한다. 우리의 주요 기여는 다음과 같이 요약된다:

1. Roadmap. 

We present a forward-looking roadmap for integrating LLMs and KGs. Our roadmap,
consisting of three general frameworks to unify LLMs and KGs, namely, KG-enhanced LLMs, LLM-augmented KGs, and Synergized LLMs + KGs, provides guidelines for the unification of these two distinct but complementary technologies.

1. Categorization and review. 

For each integration framework of our roadmap, we present a detailed
categorization and novel taxonomies of research on unifying LLMs and KGs. In each category, we
review the research from the perspectives of different integration strategies and tasks, which provides more insights into each framework.

1. Coverage of emerging advances. 

We cover the advanced techniques in both LLMs and KGs. We include the discussion of state-of-the-art LLMs like ChatGPT and GPT-4 as well as the novel KGs e.g., multi-modal knowledge graphs.

1. Summary of challenges and future directions. 

We highlight the challenges in existing research and present several promising future research directions.

여기서 저자는 LLMs과 KGs를 통합하는데 있어 크게 3가지 방법으로 나뉜다고 한다.

1. KG-enhanced LLMs
2. LLM-augmented KGs
3. Synergized LLMs + KGs

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/2703c274-295b-4d91-a54e-425b4954613e)


내 연구에 적합한 방향성은 KG-enhanced LLMs 와 Synergized LLMs + KGs이다.

Synergized LLMs + KG의 일반적인 framework를 보면 다음과 같다.

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/b19f6105-ac32-46de-9027-18011dc239c6)


그렇다면 좀 더 디테일하게 unifying LLMs with KGs에 대한 research를 카테고리화한 그림을 살펴보자.

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/94236ee8-3cf4-4863-8527-c91660524276)


### KG-enhanced LLM methods

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/13a84c01-8dc7-47aa-a20d-533f44c4036b)


여기서 내가 볼만한 논문은 다음과 같다.

1. RoG : 이것은 저번에 봤음
2. Mindmap
3. Cok

### LLM-augmented KG methods

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/9e0b2843-97d2-404a-b926-45548c20cf45)


여기서 LLM-augmented KGQA는 내가 예전부터 연구해왔던 분야로서, prompt기반이 아닌 LMs과 KG를 함께 fusion해서 QA-task를 하는 분야이다.

### Synergize KGs and LLMs

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/ea327a79-3420-4c50-a670-f77326d8fb4d)


여기서 확인할 것은 Think-on-graph 논문, KSL이다.

### **What to get**

해당 논문은 LLMs와 KGs를 통합하는 방법에 관한 조사 및 로드맵을 다루는 논문으로, LLMs과 지식 그래프를 결합하여 어떻게 활용할 수 있는지를 이해하고자 하는 분야를 구체화하고 관심 있는 논문을 찾는 데 유용한 자료라고 볼 수 있다.

나는 KG-enhanced LLMs라는 키워드에서 특히 KG-enhanced LLM inference 쪽에서 찾으면 적합하다.

이에 맞는 논문들은 다음과 같다.

1. **Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning → 봤음**
2. **“Mindmap: Knowledge graph prompting sparks graph of thoughts in large language models”**
3. **Boosting language models reasoning with chain-of-knowledge prompting**
4. **“Think-on-graph: Deep and responsible reasoning of large language model with knowledge graph”**

---

## 3. Boosting Language Models Reasoning with Chain-of-Knowledge Prompting

[https://arxiv.org/pdf/2306.06427.pdf](https://arxiv.org/pdf/2306.06427.pdf)

### Abstract

최근에는 Chain-of-Thought (CoT) 프롬프팅이 복잡한 추론 작업에서 성공을 거두었으며, 이 프롬프팅은 "단계별로 생각해보자"와 같이 간단한 프롬프트를 설계하거나 잘 설계된 근거를 가진 다중 인-컨텍스트 예제를 사용하여 Large Language Models (LLMs)를 중간 추론 단계를 생성하도록 유도하는 것을 목표로 합니다. 그러나 생성된 근거는 종종 오류가 있어 사실적이지 않고 충실하지 않은 추론 체인을 생성하게 됩니다. 이러한 취약점을 완화하기 위해 우리는 Chain-of-Knowledge (CoK) 프롬프팅을 제안합니다. 이 프롬프팅에서 우리는 구조 트리플 형식의 명시적인 지식 증거를 생성하도록 LLMs를 유도하려고 합니다. 이 아이디어는 우리 인간의 행동에서 영감을 받았으며, 복잡한 질문에 답하기 전에 뇌에서 추론 증거로 마인드 맵이나 지식 맵을 그릴 수 있습니다. CoK의 이점을 살려, 우리는 사실성과 충실성 측면에서 추론 체인의 신뢰성을 추정하기 위한 F2-Verification 방법을 추가로 소개합니다. 신뢰성이 낮은 응답의 경우, 잘못된 증거를 나타내어 LLM에 재고하도록 유도할 수 있습니다. 포괄적인 실험 결과는 우리의 방법이 상식, 사실, 기호 및 산술 추론 작업의 성능을 더욱 향상시킬 수 있음을 보여줍니다.

### Introduction

A series of recent works have explored that LLMs can spontaneously decompose the complex multi-step problem into intermediate reasoning chains, elicited by a simple prompt like “Let’s think step by step” or well-designed demonstrations with human-annotated rationales, which are called **chain-of-thought(CoT)** prompting.

This finding is intriguing and has been sensational because CoT may mainly specify an output space/format that regularizes the model generation to look step-by-step while being in order and relevant to the query.

그러나, 이러한 퍼포먼스에도 불구하고, 현재 LLMs은 신뢰할 수 없는 답변을 생성하기에 쉽다. 그리고 사실적이지 않거나 충실하지 않은 추론 체인을 제공하여 불가피하게 잘못된 결론으로 이어질 수 있다.

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/1d5022fd-efec-4928-8f1b-960c31d163ef)


To address these concerns, we propose a novel Chain-of-Knowledge (CoK) prompting method to boost the LLM’s reasoning capability by a series of exemplars that **combine explicit structure knowledge evidence with textual explanations.**

- Evidence triples(CoK-ET) : a list of structure triples can reflect the overall reasoning evidence from the query towards the answer
- Explanation hints(CoK-EH) : the explanation of this evidence.

표준 ICL과 CoT와 마찬가지로, CoK 프롬프트는 출력 공간/형식을 규제하는 규칙으로 간주될 수 있으며 LLMs에게 모호한 텍스트 추론 체인을 생성하는 대신 명시적인 증거를 생성하도록 한다.

Furthermore, we propose an $F^2$ - Verificiation strategy to estimate the reliability of the reasoning chains in terms of factuality and faithfulness.

- Factuality : the quantification of the **matching degree** between reasoning evidence and ground-truth knowledge
- Faithfulness : **consistency degree** between reasoning evidence and the textual explanation with the final answer

이전의 모델에 외부 지식을 주입하는 노력은 주로 모델 추론 단계 이전에 지식을 프롬프트에 통합하는 것에 의존해왔다. 우리는 이 절차를 통해 얻은 지식이 LLMs의 필요에 반드시 부합하지 않을 수 있다는 견해를 가지고 있다. 따라서 우리가 제안한 CoK는 모델이 명시적으로 증거 트리플을 생성하도록 유도한 다음, 포스트-검증을 통해 잘못된 트리플을 식별합니다. 그에 따라 올바른 지식이 재고되며, 이로써 세심한 지식 주입을 실현합니다.

### Methodology

CoT 프롬프팅을 통해 생성되는 reasoning chain은 가끔씩 실수가 나오고, 결국 잘못된 답변으로 이끈다.

저자들은 이러한 문제점을 textuaal reasoning chain에 문제가 있다고 지적한다.

LLMs may forcibly generate a textual rationale that conforms to the prompt format of CoT but is logically ambiguous and reaches the wrong answer.

이러한 문제점을 해결하기 위해 저자들은 두 가지 핵심을 고려한다.

1. Elicitation format of the prompt
- Text-only reasoning chains are not enough to unleash LLMs to generate reliable and concrete reasoning processes.
- Inspired by the triple structure in the KB, we need to enhance the prompt with structured features
1. Post-verification
- LLMs are usually not capable of inspecting what answers they had responded to, indicating us to leverage external
knowledge to make verification.
- Based on these considerations, we provide our specific solution on how to boost LLM’s reasoning ability.

본 논문의 프레임워크는 다음과 같다.

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/e6e8158d-521b-4379-b18b-78b84b2c358e)


### Exemplars Construction

다른 논문들에 따르면 ICL(In-Context Learning)의 성능은 annotated rationale에 따라 정해진다고 한다.

이것이 의미하는 것은 CoK(Chain-of-Knowledge)의 key challenge는 textual rationale와 함께 structure evidence triple에 있다고 한다.

그렇다면 어떻게 prompting을 하는지 살펴보자. 우선, 데이터셋 별로 Test Example에 대해서 $K$개의 Eexemplar를 만든다.

1. We randomly select $K$ questions as the basic demonstrations.
2. To automatically obtain CoK-EH, we generate a textual rationale for each question via zero-shot CoT with a simple prompt “***Let’s think step by step***”.([https://arxiv.org/abs/2205.11916](https://arxiv.org/abs/2205.11916))

2번 방법을 통해 Evidence Hint를 얻을 수 있다. CoK-ET를 얻기 위해 어떻게 해야하는지 살펴보자.

1. To figure it out, we first follow [https://arxiv.org/abs/2210.16433](https://arxiv.org/abs/2210.16433) to construct a KB $\kappa$ from six domains, involving dictionary, commonsense, entity, evnt, script, and causality, which are in the form of triple.
2. We then directly use the retrieving tool proposed by [https://arxiv.org/abs/2210.16433](https://arxiv.org/abs/2210.16433) **to retrieve some candidate triples.**
    - ScaNN 라이브러리에서 Maximim Inner Product Search 알고리즘을 사용한다.
3. Finally, we invite 5 teachers or Ph.D. students as professional annotators to manually design the corresponding CoK-ET based on the retrieved triples.

이렇게 해서 Exemplars가 완성이 된다.

완성이 된 K개의 annotated data $\epsilon =$ $\{(Q_i, T_i, H_i, A_i)\}^K_{i=1}$

Q는 input query(question),H는 explanation hint, T는 evidence triple, A는 final answer 이다. 

$\epsilon$와 test query $\hat Q_i$를 concat해서  프롬프팅을 진행해서 최종 결과를 낸다.

여기서 만약 evidence triple과 explanation hint에 대해서 성능을 측정한 reliable score 가 일정 threshold를 넘지 못하면 다시 생성하도록해서 Verification을 진행한다.

### What to get

이 논문은 Commonsense & factual reasoning, arithmetic reasoning, symbolic reasoning 네가지 QA 데이터셋에 대해서 진행했다.

그리고 각 데이터셋별로 맞는 Knowledge graph + Wiktionary를 사용했다.

각 문제 별로 K개의 exemplar를 만드는데, 이 exemplar들은 evidence triple과 explanation hint가 들어있다. 아쉬운 점은 evidence triple을 사람들이 직접 검수를 했다는점이다. 이것을 사람이 아닌 모델이 직접 하도록(prompting)하고 수식으로 당위성을 부여하면 어떨까 생각했다.(꼭 수식이 아니더라도) 또한 나는 텍스트에서 Logical fallacy가 발생하는 것을 감지하고 분류하는 것을 해결하려고 하기 때문에 그 텍스트에 대해서 외부 정보, 텍스트에서 단어들을 설명하는 정보도 중요하지만 텍스트 내에 연결관계도 중요하다. 따라서 텍스트 내에 키워드들간의 relation path(by dependency graph), 이 relation path를 지식 그래프로부터 얻는 relation path(grounded by KG) 두 가지 정보가 필요하다.

예를 들어 문장 “A student argues that because they got an 'A' on a test without studying, studying is not necessary for success”. 이 있을 때 이 문장의 dependency graph의 정보 + 지식 그래프로부터 이 문장의 논리 오류를 해결하기 위해 키워드들 간의 연결관계를 보충해줄 만한 정보(relation path, text description)를 사용하는 것이다.

이 논문에서 또 하나 얻어갈 것은 Verification 방법이라고 생각한다. 위와 같이 생성된 추가 정보들은 LLM이 직접 검증, 확인을 하는것이 아니기 때문에 Verification 방법은 얻어갈 만한 정보라 생각한다.

---

## 4. Knowledge solver : Teaching LLMs to search for domain knowledge from knowledge graphs

[https://arxiv.org/pdf/2309.03118.pdf](https://arxiv.org/pdf/2309.03118.pdf)

### Abstract

대규모 언어 모델(LLM)인 ChatGPT와 GPT-4와 같은 모델은 급부상한 능력과 일반성 때문에 다양한 작업을 수행할 수 있는 다재다능한 모델입니다. 그러나 LLM은 때로는 작업 수행을 위해 도메인 특정 지식이 부족하여 추론 과정에서 환상적인 결과를 만들어내기도 합니다. 이전 연구에서는 외부 지식 베이스에서 검색한 지식을 훈련시키기 위해 그래프 신경망(GNN)과 같은 추가 모듈이 사용되었습니다. 이는 도메인 특정 지식 부족 문제를 완화하기 위한 것이었습니다. 그러나 추가 모듈을 통합하는 것은 다음과 같은 문제점이 있습니다: 1) 새로운 도메인을 만날 때 추가 모듈을 다시 훈련해야 합니다; 2) LLM의 강력한 능력이 검색에 완전히 활용되지 못할 수 있으므로 병목 현상을 초래할 수 있습니다. 본 논문에서는 LLMs에게 강력한 일반화 능력을 활용하여 외부 지식 베이스에서 필수 지식을 검색하도록 가르치는 패러다임인 "Knowledge Solver (KSL)"을 제안합니다. 구체적으로, 검색 작업을 다중 점프 결정 순서로 변환하는 간단하면서 효과적인 프롬프트를 설계하여 LLMs에게 제로샷 방식으로 지식 검색 능력을 부여합니다. 또한, KSL은 완전한 검색 경로를 제공하여 LLMs의 추론 과정의 설명 가능성을 높일 수 있습니다. 우리는 CommonsenseQA (Talmor et al., 2018), OpenbookQA (Mihaylov et al., 2018) 및 MedQA-USMLE (Jin et al., 2021) 세 가지 데이터셋에서 실험을 수행하였으며, 우리의 접근 방식이 LLM의 기준 성능을 상당히 향상시킨다는 결과를 얻었습니다.

### Introduction

일부 시나리오에서 LLMs는 도메인 특정 지식이 부족하거나 사실과 지식을 정확하게 기억하지 못하므로 hallucination을 일으킨다.

Knowldge Graphs(KGs) are clear, logical, and superior mediums of knowledge. Thus, effectively leveraging KGs for LLMs should benefit LLMs’ performance on knowledge-required tasks.

그래서, there is a line of work using KGs to help LLMs make predictions. However, they all require **training additional knowledge-aware modules like graph neural networks (GNNs) on retrieved knowledge.**

위 내용은 두 가지 단점이 있다.

1. It would suffer from pains of retraining when encountering novel domains.
2. It would become a bottleneck since LLMs’ strong abilities are not fully utilized for retrieval.

그래서, 본 논문은 Knowledge Solver(KSL)을 소개한다 .이것은 외부 Knowledge base로 부터 지식을 LLMs 스스로 찾도록 가르치는 것이다.

### Method

Method overview는 아래 그림과 같다.

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/5e141757-4ba2-4754-9306-ffbce10dfb67)


이것이 이 논문의 전부라 볼 수 있다.

1. Question을 주고 answer entity들을 준다. 그리고 question entity로부터 랜덤하게 head entity를 준다.
2. head entity로 부터 1-Hop 연결된 entity들도 함께 prompt를 준다.

2번까지는 첫번째 prompt의 내용이며 여기서 entity는 Question-answer pair의 subgraph로 부터 나온다.

1. LLMs는 head entity로 부터 1-hop으로 연결된 next entity에서 하나를 답한다. 이것을 마지막 answer entity중 하나가 나올때 까지 반복한다. (물론 최대 횟수는 정한다.)
2. 최종 answer entity가 나오면 answer candidate중 가장 유사한 것을 고른다.

이렇게 일종의 dialogue 형태이므로 “**LLM이 스스로 찾는다**”라고 한 것이다. 

### What to get

이 논문은 내가 연구했던 상식기반QA(CommonsenseQA, OpenBookQA, MedQA)데이터셋을 다룬다.

심지어, 내가 주로 확인했던 subgraph 기반으로 모델이 구성되었다. 다만 이 논문의 발전된 점은 medium-LM(BERT,Robert)가 아니라 LLMs(GPT-3.5,LLaMA)을 사용하기 때문에 prompting을 사용했다 할 수 있다.

Subgraph를 text로 변환해서 Prompt에 넣었다. 즉, graph to text ; graph encoder?라고 볼 수 있지 않을까? 이것은 TALK LIKE A GRAPH: ENCODING GRAPHS FOR LARGE LANGUAGE MODELS 논문의 graph encoder과 유사하다고 볼 수 있다.

현재, 나는지식 그래프를 사용해서 얻은 정보를 LLM의 prompt에 넣기 위한 방법을 고민 중에 있다.

본 논문은 QA pair의 subgraph의 question entity → answer entity의 relation path를 찾아가는 방향을 제시한 방법론이다. 

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/7bce5c4d-4777-483e-b2c8-e935bcdbbb82)


이것은 subgraph정보를 prompting 했다고 볼 수 있지 않을까??

---

### 5. MindMap : Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models

[https://arxiv.org/pdf/2308.09729.pdf](https://arxiv.org/pdf/2308.09729.pdf)

### Abstract

일반적으로 LLMs는 새로운 지식 통합 능력의 제약, 환상 생성 및 의사 결정 과정의 투명성과 관련된 한계가 있습니다. 본 논문에서는 지식 그래프 (KG)를 사용하여 LLMs를 최신 지식과 연결하고 LLMs의 추론 경로를 유도하기 위한 방법을 탐구합니다. 구체적으로, LLMs에게 KG 입력을 이해하고 내재적 지식과 검색된 외부 지식을 결합하여 추론하는 능력을 부여하는 프롬프팅 파이프라인을 구축합니다. 또한, LLMs가 추론을 수행하고 답변을 생성하는 기반이 되는 마인드 맵을 유도하는 연구를 진행합니다. 생성된 마인드 맵은 지식 온톨로지에 근거한 LLMs의 추론 경로를 나타내며, 이로써 LLM 추론을 조사하고 측정하는 가능성을 제시합니다. 또한, 세 개의 질문 및 답변 데이터셋에서 진행한 실험 결과, MindMap 프롬프팅이 현저한 경험적 이득을 가져오는 것을 보여줍니다. 예를 들어, GPT-3.5에 MindMap을 사용하면 GPT-4에 비해 일관된 압도적인 성능을 나타냅니다. 또한 KG에서 검색한 구조화된 사실을 활용하는 경우, MindMap은 문서 검색을 통한 프롬프팅 방법들을 능가할 수 있으며, 더 정확하고 간결하며 포괄적인 KG 지식을 활용합니다.

### Introduction

Pre-trained LLMs can be adapted to domain tasks with further **fine-tuning** or be aligned with human preferences with **instruction-tuning.**

그럼에도 불구하고, LLMs을 제품화 하는 과정에는 여러가지 장애물이 존재한다.

1. Inflexibility
- The pre-trained LLMs posses outdated knowledge and are inflexible to parameter updating
1. Hallucination
2. Transparency
- LLMs are also criticized for their lack of transparency due to the black-box nature.

지식 그래프는 명시적인 knowledge representation과 해석가능한 reasoning path를 제공할 수 있다.

게다가, 지식 그래프는 기존 지식의 디버깅이나 새로운 지식의 추가와 같은 지속적인 수정에 용이하다.

Due to their flexibility, preciseness, and interpretability, KGs emerged as a promising complement to the drawbacks of LLMs.

우리의 작업은 KGs와 고정된 LLMs의 시너지적 추론에 중점을 두며, 이는 상업용 LLM-as-service API와 같은 강력한 사전 훈련된 LLMs에 적용 가능하다.

The goal of this work is to build a plug-and-play prompting approach to elicit the **graph-of-thoughts** reasoning capability in LLMs.

Graph-of-thoughts : 개념, 사실, 아이디어, 또는 의견들이 노드로 표현되고, 이러한 노드 간의 관계나 연결성이 엣지로 표현되는 그래프 구조를 사용하여 인간의 사고 패턴이나 지식 구조를 시각화하거나 구조화하는 방법.

We call our method MindMap because it enables LLMs to comprehend graphical inputs to build their own mind map that supports evidence-grounded generation.

구체적으로, MindMap은 LLM의 사고 그래프를 활성화시켜 (1) KG로부터 검색된 사실들을 통합하고 LLM으로부터의 암묵적 지식을 결합하며, (2) 입력 KG에서 새로운 패턴을 발견하고, (3) 최종 출력을 도출하기 위해 마인드 맵을 통해 추론한다.

### Method

1. **Evidence graph mining** : We begin by identifying the set of entities $V_q$ from the raw input and query the source KG $G$ to build multiple evidence sub-graphs $G_q$
2. **Evidence graph aggregation** : Next, LLMs are prompted to comprehend and aggregate the retrieved evidence sub-graphs to build the reasoning graphs $G_m.$
3. **LLM reasoning on the mind map** : Last, we prompt LLMs to consolidate the built reasoning graph and their implicit knowledge to generate the answer and build a mind map explaining the reasoning process.

여기서 내게 필요한 부분은 1번

### Step 1 : Evidence Graph Mining

외부 지식 그래프에서 relevant evidence sub-graphs $G_q$를 발견하는 과정은 주로 두 단계로 나뉜다.

1. **Entity Recoginition**
    1. Prompting LLMs to extract the key entities from the question query Q via in-context learning
    2. a로부터 생성되는 엔티티들을 M set이라하자
    3. M이 실제 지식 그래프에 존재하지 않을 수 도 있으니 entity linking을 수행함
    4. entity linking은 G(KG)의 모든 엔티티들과 M의 모든 엔티티들을 BERT encoder를 사용해서 임베딩 $H_G$와 $H_M$을 만든다.
    5. 코사인 유사도를 비교하여 M에  있는 각 엔티티들을 G의 가장 가까운 이웃 엔티티에 링크한다. 이는 다음 단계인 evidence sub-graph를 구축하기 위한 초기 entity set $V_q$를 생성한다.
2. **Evidence sub-graphs Exploration**

![image](https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/d952edde-32f5-4647-ae69-33e270c9fcc5)


### Step 2 : Evidence Graph Aggregation

Step1에서 생성된 path-based evidence Graph exploration과 Neighbor-based evidence graph exploration 두 가지를 자연어로 변환해서 함께 prompting을 진행해서 최종 결과를 도출한다.

This design offers two advantages: 

1. LLM adopts a holistic view of small sub-graphs, condensing them into a unified and succinct format.
2. LLM aids in the disambiguation of similar entities in natural language.

### What to Get

본 논문은 Text로부터 keyword를 추출하고, entity linking을 통해 최종 subgraph에 들어갈 entity를 선정한다. 그 다음 두 가지 방법으로 subgraph를 생성하는데 이 방법들은 KG의 특징인 structured knowledge를 적극적으로 잘 활용했다 볼 수 있다. 그렇게 해서 생성된 두 가지 subgraph의 relation pathway를 자연어 형태로 변환(e.g., “(Fatigue, Nausea) - IsSymptomOf - LiverProblem)해서 prompt에 넣어 최종 결과를 도출한다.

본 논문을 통해 얻을 수 있는 정보는 sub graph를 다양한 방법의 형태로 만들고 모두 활용한 점이다. 물론 내 연구와는 다르게 본 논문의 task는 Medical Question answering이므로 subgraph를 생성하는 두 가지 방법이 나와 다를 수 있다. 하지만 이 논문을 통해 얻은 아이디어는 다음과 같다.

- 텍스트로부터 키워드를 추출하는 점은 내 생각과 동일하다.
- sub-graph 생성방법을 텍스트에 대해서 두 가지 방법을 실행했는데, 이를 통해 나는 텍스트 자체의 구조 관계에 대한 그래프(dependency graph)와 텍스트로 부터 external knowledge KG를 사용해서 생성되는 subgraph를 사용하면 어떨까라는 아이디어가 떠올랐다.
