# Paper summary

작성자 : 정지원

## 1. Selection-Inference : Exploiting Large Language Models for Interpretable Logical Reasoning

- [https://arxiv.org/pdf/2205.09712.pdf](https://arxiv.org/pdf/2205.09712.pdf)

### Abstract

대형 언어 모델(LLM)은 새로운 작업에 대한 훌륭한 적은 양의 데이터를 활용한 일반화 능력을 보여주었습니다. 그러나 여전히 복수 단계의 논리적 추론 문제에서 성능이 저조한 경향이 있습니다. 여기서 우리는 논리적 추론의 다른 측면을 조사하는 50가지 작업에 대한 LLM의 철저한 평가를 실시합니다. 우리는 언어 모델이 단일 단계 추론이나 함의 작업에서 꽤 잘 수행되지만, 보다 복잡한 문제를 해결하기 위해 여러 추론 단계를 연결하는 데 어려움을 겪는다는 것을 보여줍니다. 이에 따라 우리는 **사전 훈련된 LLM을 일반 처리 모듈로 활용하고 선택과 추론을 번갈아가며 최종 답변으로 이어지는 해석 가능한 원인 추론 단계 시리즈를 생성하는 Selection-Inference(SI) 프레임워크를 제안합니다**. 우리는 동일한 세트의 10가지 논리 추론 작업에 대한 일련의 7B 파라미터 LLM이 SI 프레임워크 내에서 5회의 샷 일반화 설정에서, 세밀한 조정 없이, 동등한 바닐라 베이스라인 대비 100% 이상의 성능 향상을 보여줍니다. 동일한 모델이 동일한 환경에서 동일한 작업 세트에서 크게 더 큰 280B 파라미터 베이스라인을 심지어 능가합니다. 또한, SI 프레임워크에 의해 생성된 답변은 중요한 시스템의 안전성과 신뢰성에 대한 함의를 갖는 원인 기반의 자연어 추론 추적으로 수반됩니다.

### Introduction

신경 기호 접근법은 인공 지능 (AI)의 두 가지 주요 패러다임인 심볼릭 AI와 연결주의 AI를 통합하는 방법론을 말한다. 심볼릭 AI는 기호 및 규칙을 기반으로 추론하고 논리를 다루는 반면, 연결주의 AI는 데이터에서 패턴을 학습하고 추론하는 방법을 중심으로 한다. 신경 기호 접근법은 이러한 두 가지 접근법을 통합하여 기호적 표현과 신경망의 강점을 결합하여 보다 강력하고 유연한 인공 지능 시스템을 구축하려는 목적으로 개발되었다. 이러한 방법론은 기호적 추론과 신경망 기반 학습을 통합하여 문제 해결과 추론 과정을 보다 효과적으로 수행할 수 있도록 돕는다.

**selection-inference framework**

그림은 다음과 같다.

<img width="902" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/d2d539c4-4ec1-4815-864d-01c2a215022a">


(a)는 일반적인 baseline, (b)는 COT, (c)는 sI이다.

(c)인 Selection-Inference (SI) 프레임워크는 두 단계로 구성된다.

1. 선택 단계 (Selection step): 이 단계에서는 [문맥, 질문, 선택]×𝑘를 입력으로 받아 k-샷 프롬프팅을 수행한다. 여기서 문맥은 주어진 정보나 상황을 나타내며, 질문은 사용자가 제공한 질문을 의미한다. 선택 단계는 문맥에서 단일 추론 단계를 수행하기에 충분한 관련 정보의 하위 집합을 선택하는 데 중점을 둔다. 따라서 선택 모듈은 주어진 문맥과 질문에 대해 적절한 정보를 선택하여 나중에 이어지는 추론 단계를 지원할 수 있는 사실의 부분 집합을 생성한다.
2. 추론 단계 (Inference step): 이 단계에서는 [선택, 추론]×𝑘를 입력으로 받는다. 선택 단계에서 생성된 선택 정보와 추론 정보를 입력으로 받아, 새로운 사실(즉, 추론)을 생성한다. 이렇게 생성된 새로운 사실은 문맥에 추가된다. 각 [선택 + 추론 + 문맥에 사실 추가]의 조합은 하나의 추론 단계를 형성하며, 이러한 단계들은 연쇄적으로 연결되어 어려운 문제에 대한 답변을 생성한다. 최종적으로 추론된 사실은 최종 답변으로 취급된다.

이런 방식으로 SI 프레임워크는 선택과 추론 단계를 통해 여러 단계의 추론을 연결하여 복잡한 문제를 해결하고, 최종적으로 해석 가능한 추론 과정을 제공한다.

질문에 직접적으로 접근하거나 이전 추론 단계에 대한 직접적인 접근은 없습니다. 이는 Post-hoc rationalisation를 통해 얻는 더 일반적인 접근법과 대조된다.

COT, 이 접근법에서는 LLMs가 답변 이전에 추론 트레이스를 생성하도록 권장된다. 그러나 답변이 추론에 의존하는 정도가 명시적으로 인과적이라는 것은 장려되지 않는다. 실제로 저자들은 COT 설명이 최종 답변 정확도를 향상시키는 데 도움이 되지만, 모델이 생성한 추론 트레이스는 종종 최종 답변이 정확한 경우에도 잘못되었음을 보여줍니다.

### Method

뉴로심볼릭 방법에 영감을 받아 저자들은 reasoning과정을 두 구성요소로 나누었다.

1. Selection : which selects a subset of the information present in the context
2. Inference : which produces the new fact, based on the information passed to it by the selection step.

이렇게 reasoning 과정을 selection과 inference로 나누는 과정은 몇가지 중요한 점이 있다고 한다.

1. It makes the resulting reasoning causal, since both steps have limited capabilities by design, and are interdependent.
2. Each step of reasoning is broken down into even smaller sub-tasks, which are easier for LLMs to adapt to, and which helps make the reasoning more generalisable to harder problems.

알고리즘은 다음과 같다.

<img width="984" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/bae7786a-fba6-4b91-bcc9-a39cbcbd0a9a">


### What to get

본 논문은 교수님께서 “Logical Fallacy detection/classification 를 수행하는데 있어서 LLM에 지식 그래프를 사용하는데 있어서 지식 그래프의 정보를 “step-by-step” 방법으로 프롬프팅을 사용해서 적용하면 어떨까” 에서 소개한 논문이다. 

저자들은 reasoning과정을 Selection과정과 Inference과정으로 나누어서 reasoning을 진행했다.

Selection 과정을 통해 문맥에서의 정보를 수집하고 선택한다. 선택된 정보는 Inference과정에서 문맥과 함께 사용되어 새로운 사실을 도출한다. 새로운 사실은 기존 문맥에 들어가 다시 selection과정을 거친다. 이러한 과정이 최종 정답을 도출할때 까지 진행한다(물론 종료 조건은 하이퍼 파라미터로 있다).

이 과정을 내 연구와 접목 시킨다면, 논리 오류를 발생 시키는 텍스트로 부터 키워드를 추출하고 키워드 간 relation path or subgraph를 생성한다…. 그 이후 방식은 지금 당장 떠오르지 않네,,

하여튼 이런 느낌을 접근하게 해준 첫 논문이다. 물론 비슷한 느낌의 방법은 있었다. [Knowledge solver paper](https://arxiv.org/abs/2309.03118)

---

## 2. **Post Hoc Explanations of Language Models Can Improve Language Models**

- [https://arxiv.org/pdf/2305.11426.pdf](https://arxiv.org/pdf/2305.11426.pdf)

### Abstract

대규모 언어 모델(LLMs)은 복잡한 작업을 수행하는 놀라운 능력을 보여주었습니다. 더욱이 최근의 연구는 인간이 주석을 단 근거들(예: 사고 과정 프롬프트)을 적용하는 것이 문맥 내 학습 중에 이 모델의 성능을 크게 향상시킬 수 있다는 것을 보여주었습니다, 특히 추론 능력이 필요한 작업에 대해 말이죠. 그러나 이러한 근거들을 포함하는 것은 인간의 개입이 많이 필요하기 때문에 확장 가능성 면에서 도전을 겪고 있습니다. 본 연구에서는 자동화된 근거 생성 프로세스를 통해 이러한 도전에 대응하는 새로운 프레임워크인 Amplifying Model Performance by Leveraging In-Context Learning with Post Hoc Explanations (AMPLIFY)를 제안합니다. 이를 위해 후후 설명 방법을 활용하여 각 입력 특성이 모델 예측에 미치는 영향을 잡아내는 속성 점수(설명)를 출력합니다. 더 구체적으로, **우리는 후후 설명에서 얻은 통찰을 포함하여 자동으로 자연어 근거를 구성하여 LLMs에 수정 신호를 제공합니다.** 실제 데이터셋을 사용한 포괄적인 실험 결과, AMPLIFY 프레임워크가 사고 과정 프롬프트와 같이 인간이 주석을 단 근거에 의존하는 기존 접근 방법이 부족한 작업을 포함하여 다양한 작업에서 약 10-25%의 예측 정확도 향상을 이끌어냅니다. 우리의 연구는 Post hoc explanations이 LLMs의 효과성을 향상시키는 데 중요한 도구로서의 잠재력을 강조하는 최초의 시도 중 하나입니다. 더 나아가, AMPLIFY의 각 구성 요소가 미치는 영향을 보여주기 위해 추가적인 경험적 분석 및 제거 실험을 실시하여 문맥 내 학습을 개선하기 위한 중요한 통찰을 제공합니다.

### Introduction

While incorporating such human-annotated rationales has contributed to enhancing model performance, it is not a scalable approach as it involves a lot of human intervention, thus, limiting its applicability to the ever-expanding range of tasks that are being handled by LLMs.

이러한 문제를 다루기 위해, 저자들은 Amplifying Model Performance by Leverage In-Context Learning with Post Hoc Explanations(AMPLIFY) framework를 제안한다. which can automatically generate rationales to improve the performance of LLMs on tasks involving sophisticated reasoning and language understanding.

To this end, we leverage post hoc explanation methods which output attribution scores that capture the influence of each of the input features on model predictions.

AMPLIFY framework는 4단계의 구성으로 이뤄져있다.

1. Select a proxy model for which post hoc explanation generation is computationally viable
- LLM보다 작은 BERT나 GPT-2를 proxy 모델로 사용한다. 왜냐하면 그래디언트 계산에 LLM은 부적합하기 때문에
1. Identify samples that are most likely to provide corrective signals to LLMs
2. Compute post hoc explanations for the samples identified in the previous step
3. Construct a few-shot prompt using the samples chosen in step2, their true labels, and the post hoc explanations obtained in step 3 as rationales.

### Method

<img width="583" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/09db6855-683d-45d7-b092-fdf107604228">


1. Proxy Model Selection
- 작업 가능한 계산 비용 내에서 post hoc 설명을 생성할 수 있는 프록시 모델을 선택한다. 이를 위해 미리 학습된 모델(예: GPT-2, BERT 등)을 사용하거나 대상 작업에 대해 더 작은 언어 모델을 직접 fine-tune 또는 pre-train하는 두 가지 전략을 고려한다.
1. Few-shot Sample Selection
- LLM이 수정 신호를 받을 가능성이 있는 샘플(입력, 라벨) 쌍을 식별한다. 이를 위해 검증 세트에서 LLM에 의해 잘못 분류된 인스턴스를 식별하고, Misclassification Confidence Score(MCS)라는 지표를 사용하여 랭크한다.
1. Rationale Generation
- 이전 단계에서 선택된 각 샘플에 대해 post hoc 설명을 계산한다. 각 샘플에 대해 (입력, 라벨) 쌍과 프록시 모델을 사용하여 post hoc 설명 방법을 적용하고, 입력 문장의 각 토큰에 대한 기여도 점수를 계산한다. 그래디언트를 사용해서 기여도를 계산한다.
1. Prompt Design for LLMs
- 마지막으로, 선택된 각 샘플에 대해 수정된 근거를 구성한다. 이를 위해 이전 단계에서 얻은 최상위 k개 중요 단어들을 사용하여 근거를 구성하고, 이를 입력, 근거, 라벨로 구성된 few-shot prompt로 결합하여 LLMs에 제공한다. 이로써 LLMs는 테스트 세트의 샘플들에 대한 예측을 수행할 수 있다.

### Conclusion

이 연구에서는 인간이 주석을 단 근거들을 자동으로 생성된 근거들로 대체하여 LLM의 성능을 향상시키기 위한 새로운 프레임워크인 AMPLIFY를 소개합니다. 우리의 독특한 네 단계 접근법은 작은 오픈 소스 모델을 활용하여 후속 설명을 효율적으로 계산합니다. 우리의 프레임워크는 다양한 작업에서 10-25%의 성능 향상을 가져오며, 인간 주석을 사용하는 CoT 프롬프팅과 같은 전통적인 기술을 능가합니다. 우리의 연구 결과는 후속 설명 방법이 LLM의 효과성을 향상시키는 데 유용한 도구로서의 잠재력을 강조합니다.

### What to Get

본 논문은 Post Hoc Explanation 방법을 사용해서 prompting을 진행한 논문이다. 문장내 각 토큰을 계산하기 전에 MCS를 계산해서 잘못된 정답을 높이는데 가장 큰 역할을 하는 문장들을 선택하고, 그 문장들 내에서 토큰 별 gradient를 계산한다. gradient가 높은 top-k개의 토큰이 선택되면, 그 토큰들이 해당 문장 내의 keyword가 되는 것이다.

이렇게 few-sample을 만들고, 최종적으로 few-shot prompting을 진행하는것이다.

내 연구 분야인 Logical Fallacy text에서 논리 오류가 발생하는 조건은 다양하며, 지식 그래프가 사용되기에 적합한 logical fallacy class들은 주로 문장내 전제와 결론간의 연결관계의 오류 및 부정으로 인해 발생하는 오류들이다. 이러한 문장들에서 논리 오류가 발생하는 원인의 단어를 찾는데는 도움이 될 것같지만, 현재 ChatGPT로 평가해도 매우 낮은 점수가 나오는 상황에서  chatgpt 보다 성능이 훨씬 떨어지는 Proxy model을 사용해서 그래디언트를 찾는다면, 큰 의미가 없을 것으로 보인다. 

얻어갈 수 있는 것은 keyword를 선택하는 방법?(post hoc explanation) 정도라 생각한다.

---

## 3. Verify-and-Edit : A Knowledge-Enhanced Chain-of-Thought Framework

- [https://arxiv.org/pdf/2305.03268.pdf](https://arxiv.org/pdf/2305.03268.pdf)

### Abstract

대형 언어 모델(LLM)이 자연어 처리(NLP)에서 표준이 되면서 생성 및 추론 작업에서 우수한 성능을 나타내고 있지만, 그 중 가장 치명적인 단점 중 하나는 사실적인 정확성의 부족입니다. 사실에 반하는 텍스트를 생성하는 것은 성능 하락뿐만 아니라 응용 프로그램의 신뢰도와 타당성을 저하시킵니다. Chain-of-Thought (CoT) 프롬프팅은 해석 가능한 추론 체인을 생성함으로써 복잡한 추론 작업에서 신뢰와 모델 성능을 향상시키지만, 지식 중심 작업에서는 여전히 사실성에 대한 우려가 있습니다. 본 논문에서는 CoT 프롬프팅을 위한 **Verify-and-Edit 프레임워크를 제안하여 외부 지식에 따라 추론 체인을 사후 편집하여 예측 사실성을 증가시키고 있습니다.** GPT-3를 기반으로 한 우리의 프레임워크는 다양한 개방형 도메인 질문 응답 작업에서 정확도를 향상시켰습니다.

### Introduction

논문에서는 대규모 언어 모델(LLMs)이 다양한 NLP 작업에서 새로운 표준이 되었다고 언급하면서, Chain-of-Thought (CoT) 프롬프팅이 복잡한 추론이 필요한 작업에서 성능을 향상시키는 데 도움이 되는 것으로 밝혀졌다고 소개하고 있다. 그러나 이러한 방법의 주요 초점은 생성된 CoT를 그대로 사용하여 최종 작업 성능을 향상시키는 데에 있었다. 그리고 이런 방식은 종종 너무 많은 정보를 제공하거나 관련 없는 세부 사항을 포함하기 때문에 바람직하지 않다. 따라서 좀 더 관련성이 높고 논리적인 근거를 얻기 위해 자연스럽고 생성적인 접근 방식을 사용하고자 한다. 이에 영감을 받아 논문에서는 Verify-and-Edit (VE) 프레임워크를 제안하여 이러한 추론 체인을 사실적으로 조정하여 예측 성능을 향상시키는 것을 목표로 한다. 이 프레임워크는 특히 불확실한 사례를 선택하여 편집하고 외부 지식을 도입하여 추론 체인을 수정하는 것으로 구성되어 있습니다. 이와 같은 수정된 근거를 사용하여 새로운 예측을 생성하면서 보다 사실적으로 정렬된 추론 트레이스를 고려한다. 이러한 프로세스는 새롭게 수정된 CoT 스타일의 추론 체인을 향상시키기 위한 최초의 작업으로 알려져 있다. 이를 통해 논문은 추론이 필요한 두 개의 오픈 도메인 질문 응답(QA) 작업에서 실험을 수행하고, 더 많은 사실적인 추론 체인에서 혜택을 얻어 더 정확한 예측을 생성한다는 것을 확인했다. 결과적으로, VE 프레임워크는 고도로 사실적인 예측을 위해 LLMs의 추론 과정을 보완하고 성능을 향상시키는 데 유용한 도구로 나타난다.

### Method

Our goal is to make LLMs generate more factual reasoning chains with CoT prompting assisted with external knowledge, thereby also improving prediction accuracy of the final answer.

<img width="517" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/9c2f77a9-6588-4b1b-b3ed-46fd74839bf4">


논문의 저자들은 인간의 추론과정을 따르기를 희망한다. 사람이 질문에 답변할 때, 그또는 그녀가 확신이 없다면, 지지하는 사실을 찾아 고려한 후에 최종 답변을 내놓을 것이다.

그래서, 저자들은 Verify-and-Edit(VE) framework를 3단계로 구분한다.

1. finding uncertain predictions
2. editing their rationales by searching for supporting facts.
3. using the edited rationales to generate final answers.

**Deciding when to edit**

모델이 예측에 자신이 없는 경우를 어떻게 파악할 수 있을까? 자기 일관성 방법(Wang et al., 2022)이 해결책을 제공한다. 다양한 추론 경로와 답변을 샘플링할 때, 자기 일관성은 정확성과 높은 상관 관계가 있으며, 이는 모델이 "모르는 것을 알 때"에 대한 불확실성 추정을 제공하고 모델이 그것을 알 수 있는 능력을 부여할 수 있다는 것을 시사한다. 따라서 예측 작업을 위해 자기 일관성 방법을 사용하여 다양한 추론 경로를 n개 샘플링한다. 높은 일관성을 보이는 예측은 그대로 두고, 일관성이 [n/2]보다 낮으면, 즉, 대다수가 동일한 답변에 동의할 수 없는 경우, 이를 "불확실"로 라벨링한다.

**How to edit a specific rationale**

특정한 사고과정, 즉 이유(Chain of Thought, CoT)는 사실과 추론 두 부분으로 나눌 수 있다. 이에 따라 우리는 CoT를 양쪽에서 개선하는 것을 고려한다.

- 사실(Fact): 사고과정을 보다 사실적으로 만들기 위해, 외부 지식 소스(예: 위키피디아, 구글)에서 지지하는 사실을 찾는다. 이를 위해, 사실을 검증하기 위해 인간의 질문을 모방하여 자연스러운 질문이 생성된다. 이를 위해, 동일한 LLM의 컨텍스트 학습 기능을 사용한다. 원본 질문과 이유가 검증 질문 생성을 위해 프롬프트로 제공되어 원본 질문을 대답하기 위해 필요한 가장 관련된 정보를 요구함을 보장한다. 예를 들어, 이유(잘못된)가 "1961년 8월 4일에 태어난 미국 대통령은 존 F. 케네디입니다."이고, 원본 질문이 "1961년 8월 4일에 태어난 미국 대통령의 배우자는 누구입니까?"라면 생성된 검증 질문이 "1961년 8월 4일에 태어난 미국 대통령은 누구입니까?"가 되는 것을 기대한다. 이렇게 생성된 관련 질문은 생성된 이유를 직접 쿼리하는 대신 관련 질문을 생성함으로써 잘못된 사실 생성으로 인한 잠재적인 잡음을 제거한다.
- 추론(Reasoning): Creswell et al. (2022)의 Selection-Inference와 같은 방법은 검색된 사실을 이유로 직접 사용하지만, 이러한 방법은 보통 원하는 것보다 너무 많은 설명이나, 원하지 않는 길이의 이유, 또는 관련 없는 세부 정보를 포함할 수 있다. Ye and Durrett (2022)은 지원 문장을 직접 사용하는 것이 보통 너무 장황하고 충분하지 않다고 지적했다. 보다 관련성 있고 논리적인 이유를 얻기 위해 우리는 자연스럽고 생성적인 접근을 다시 활용한다. 추론 능력이 이미 LLM에 내재되어 있다고 믿기 때문이다. 구체적으로, "질문, 이유, 답변" 형식의 프롬프트를 제공함으로써 LLM은 답변 생성 전 몇 단계를 추론한다. 원래의 이유를 조사한 결과, 잘못된 사실을 포함하더라도 논리적 추론 요소가 일반적으로 유지되는 것을 관찰한다. 따라서 우리는 검증 질문(논리)과 검색된 사실(정보)을 사용하여 유의미한 답변을 생성한다. 유의미한 답변은 새로운 이유로 구성되어 잠재적으로 더 사실적인 CoT를 제공한다.

알고리즘은 다음과 같다.

<img width="740" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/6abb4d33-5273-4d18-8d98-7b5cfb508a64">


Retrieve external knowledge : DrQA, Wikipedia, Google search

### Conclusion

본 논문에서는 오픈 도메인 질문 응답을 위한 Verify-and-Edit 프레임워크를 소개합니다. 이는 더 나은 최종 작업 성능을 위해 CoT 스타일의 추론 체인을 후속 편집하는 최초의 시도입니다. 지식 검색과 추론을 결합하여 프레임워크는 CoT를 자연스럽고 대화식으로 편집하여 예측의 사실성을 향상시킵니다. Google 검색과 결합된 이 프레임워크는 최신 LLM의 오픈 생성 능력과 검색 엔진이 제공하는 최신 정보를 결합한 유망한 방향을 보여줍니다.

### What to Get

본 논문은 Selection-Inference논문에서 selection부분에서 선택되는 모든 정보를 사용하는 것이 아니라, 그 정보에서 top-k개를 선택한다. 이정도 차이점이 있다. COT에서 reasoning부분을 기반으로 검증을 재진행하는 것이다. 검증을 하기 위해 external knowledge를 사용하고, 검증이 된 reasoning을 가지고 정답을 검증한다.

내 연구에 적용하자면 fallacy text로부터 키워드를 추출후 지식 그래프를 사용해서, sub-graph or relation path를 사용할때 이 정보들이 제대로 되었는지, 아닌지를 검증할때 사용할 법한 아이디어라는 생각이 들었다.

sub-graph or relation path 뿐만 아니라, 지식 그래프를 통해 얻은 정보를 검증하는데 유용하지 않을까?

검증에 대해서 생각해보자

---

## 4. Knowledge-Driven CoT : Exploring Faithful Reasoning in LLMs for Knowledge-intensive Question Answering

- [https://arxiv.org/pdf/2308.13259.pdf](https://arxiv.org/pdf/2308.13259.pdf)

### Abstract

Chain-of-Thought (CoT)을 장착한 대형 언어 모델(LLM)은 다양한 하류 작업에서 인상적인 추론 능력을 보여주었습니다. 그럼에도 불구하고 환각과 외부 지식에 대한 접근 불능으로 인해, LLM은 종종 부정확하거나 충실하지 않은 중간 추론 단계를 가지고 오며, 특히 KBQA와 같은 지식 집약적 작업에 대한 응답 문맥에서 그러한 문제가 두드러집니다. 이 문제를 완화하기 위해, 우리는 Knowledge-Driven Chain-of-Thought (KD-CoT)라는 프레임워크를 제안합니다. 이는 외부 지식과의 상호 작용을 통해 CoT의 추론 트레이스를 검증하고 수정하여 환각과 오류 전파를 극복합니다. 구체적으로, LLM의 CoT 이유화 프로세스를 구조화된 다중 라운드 QA 형식으로 제시합니다. 각 라운드에서 LLM은 외부 지식을 검색하고 정확한 답변을 검색한 후 신뢰할 수 있는 추론 트레이스를 생성합니다. LLM의 구조화된 CoT 추론은 저희가 개발한 KBQA CoT 컬렉션에 의해 용이하게 됩니다. 이 컬렉션은 문맥에서의 학습 데모를 제공하며 강력한 리트리버를 훈련시키기 위한 피드백 증강으로도 활용될 수 있습니다. WebQSP와 ComplexWebQuestion 데이터셋에서의 광범위한 실험은 제안된 KD-CoT의 작업 해결 추론 생성에서의 효과를 입증합니다. 이는 순수 CoT ICL에 비해 절대적인 성공률이 8.0%와 5.1% 향상됩니다. 더 나아가, 우리가 제안한 피드백 증강 리트리버는 지식 검색에서 최첨단 베이스라인을 능가하여 히트와 리콜 성능에서 상당한 향상을 이룹니다.

### Introduction

The ability of LLMs can be further unleashed through in-context learning conditioning on a few concatenated demonstrations without task-specific training or fine-tuning.

이러한 발전, 진보에도 불구하고, LLMs still encounter hallucinations or lack of knowledge while solving knowledge-intensive tasks.

이전 연구에서 COT과정에서 관련 정보를 찾을 때 Web을 사용한다든지, 중간 추론과정을 검증하기 위해 추가적인 verification system을 사용해서 rationale을 다시 생성하는 연구들이 있었다.

하지만, the problem of hallucinations in complex multi-hop problem scenarios is still understudied.

현재 최첨단 기법들이 외부 지식을 검색하고 지식 관련 답변을 반환하는 검색 후 독해 파이프라인을 활용한다고 고려할 때, 우리는 이러한 패러다임을 적용하여 문제를 해결할 수 있다.

이러한 문제를 다루기 위해, 저자들은 Knowledge-Driven Chain-of-Thought(KD-COT)를 제안한다.

본 논문의 contribution은 다음과 같다 :

- We present a KBQA CoT collection by prompting LLMs, which could be used for fine-tuning smaller LMs to acquire CoT reasoning ability and be applied to perform ICL.
- We propose a retriever-reader-verifier QA system to access external knowledge and interact with LLM. We leverage the constructed CoT collection as feedback augmentation to train a robust retriever, achieving significant improvement in Hit scores on WebQSP and CWQ.
- We introduce KD-CoT, a Knowledge-Driven Chain-ofThought framework to improve the reasoning performance of large language models. Experimental results demonstrate the effectiveness of our proposed framework, achieving 8.0 and 5.1 Hit@1 improvement on WebQSP and CWQ compared to the vanilla ICL method.

### Methodology

In this section, we first present the procedure for constructing the CoT Collection. Then we introduce the Knowledge-Driven CoT framework, which encompasses the implementation of the interaction and the training of the external QA system.

**CoT Collection**

CoT fine-tuning을 위한 rationale data는 큰 가치를 보여주었지만, 이러한 고품질 rationales를 구축하는 것은 인간이 직접 작성하는 rationale을 수집하는데 어려움과 대형 모델의 hallucination으로 인해 매우 어려워지고 있다.

rationale은 모델이 작업을 더 잘 이해하고 더 정확한 응답을 생성하는데 도움이 되는 예시나 근거를 말한다.

**Here we provide a detailed description of how we construct knowledge-intensive rationales with the help of LLMs.**

ICL에서 demonstraions이 있을 때,LLM은 더 좋은 성능을 보인다고 한다.

Inspired by this, we assign the demonstrations every time request LLMs.

To obtain the demand demonstrations, we first manually write several accurate CoT demonstrations as the anchor set, and then we employ an iterative algorithm to construct our full collection.

In each iteration, we choose the candidate in the current collection that holds the highest cosine similarity with the question in the training set to serve as the demonstration.

코사인 유사도를 측정할 때, Roberta를 사용해서 임베딩을 측정한다고 한다.

Next, we request ChatGPT to generate the structured CoT, and append generated results ”Finish” with the correct answer to the collection.

알고리즘은 다음과 같다;

<img width="355" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/8ee700aa-291b-4056-96cf-9975ce7862bf">


**Knowledge-Driven CoT**

Due to hallucinations and inability to access external knowledge, LLMs struggle to generate faithful reasoning steps for knowledge-intensive QA tasks.

To address this issue, we propose Knowledge-Driven Chain-of-Thought Reasoning(KD-CoT), which incorporates a QA system to interact with LLMs.

KD-CoT는 test 데이터에서 진행한다.

1. we select the instance with the highest cosine similarity from the collection, and utilize its rationale as the demonstration to perform one-shot ICL.
2. Then the extracted intermediate sub-question is taken as the **input of the QA system** to perform interaction, which is comprised of a retrieve-then-read pipeline and an answer verifier.
    - retrieve-then-read module retrieves external knowledge and proposes a candidate answer based on the retrieved information.
    - An answer verifier chooses between the original sub-answers generated by LLM and the proposed candidate answer.
    
    이 interaction을 CoT가 끝날때까지 진행한다.
    

Note that our motivation is to supervise the intermediate reasoning of LLM and not to alter the ultimate answer. Therefore, we restrict our interaction with the external QA system to sub-questions leading up to the final Action.

intermediate reasoning의 정확성을 보장하기 위해 external knowledge에 효과적으로 접근하고 고도로 정확한 답변을 생성할 수 있는 강력한 QA system이 필수적이다.

We then introduce **how to train our retrieve-then-read pipeline and the verifier.**

**KB Linearization**

We aim to interact with both structural(KBs) and unstructured (Wikipedia) external knowledge.

하지만, KB에서 정보를 직접 검색하는 것은 큰 규모와 의미론적 및 구조적 복잡성 때문에 쉽지 않다.

이 문제를 다루기 위해, 저자들은 Freebase KB data를 unstructured text로 변환하기 위해 linearization method를 사용한다. linearization method는 이 논문의 방법을 따랐다고 한다.([https://arxiv.org/pdf/2210.00063.pdf](https://arxiv.org/pdf/2210.00063.pdf))

Given a head entity, we extract its 1-hop subgraph and concatenate the KB triplets with spaces, then group the entire subgraph into a single passage.

즉 트리플을 텍스트로 변환해서 retrieve model에 넣을 input이 되는것이다.

그리고 head entity를 wikipedia에 검색한 텍스트도 있는데, wikipedia passage와 KB passage를 concat해서 최종 knowledge retrieval 하기 위한 준비가 되었다.

**Feedback-Augmented Retriever**

BM25를 통해 positive, negative passage를 정한다. 정해진 passage들로 DPR알고리즘을 사용해서 Retriever모델이 완성이 된다.

**Fuse-in-Decoder Reader**

<img width="239" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/3ec9e6bd-891c-4e64-9761-6a1c585ba832">


**Verifier**

<img width="232" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/29262ef6-a40e-45cc-9edb-8ac581e0dc8a">


### What to get

본 논문은 Freebase KB기반 QA-dataset WebQSP와 CWQ 데이터셋을 대상으로 KBQA tasK를 수행하는 과정에서 LLM에 지식 그래프 정보를 어떻게 활용할까에 대한 논문이다.

external knowledge로 Freebase와 wikipedia를 사용했으며, Freebase KB의 구조적 정보를 텍스트로 변환하여 사용했기 때문에 **“구조적”** 특징을 활용했다 보기는 어려웠다.

또한, Retriever-reader 부분을 흔히 알고 있는 BM25, DPR알고리즘을 사용했다.

CoT collection에 CoT의 내용(demonstration, rationale)들을 retriever-reader 모델의 입력으로 사용했기 때문에 LLM과 KG를 적절하게 사용했다고 볼 수 있다.

나는 논리 오류(logical fallacy)가 발생하는 텍스트에서 지식 그래프를 활용해서 텍스트의 논리 오류를 인식하고 분류하는 것을 목적으로 두고 있다. 여러가지 논리 오류가 있지만 지식 그래프가 활용되기 위해서는 텍스트내에 연결관계에서의 정보 오류로 인한 문제가 있는 논리 오류에 관심을 가지고 있다. 예를 들어, hasty generalization, false causality, cherry picking 같은 것이 있다.

앞서 말한 논리 오류 텍스트들은 보통 두 가지 문장, 전제와 결론으로 나눌 수 있다.

전제 문장에서의 키워드, 결론 문장에서의 키워드들을 기반으로 지식 그래프를 사용하는 과정의 정보들을 CoT collection으로 만들고 test data에 prompting을 진행한다면 어떨까?

다만, 지식 그래프의 정보를 어떻게 사용하냐이다…

계속 LLM with KGs paper들은 KBQA dataset이다. 

이 논문의 방법을 잘 사용하면 될 것 같은데,,?

1. without sufficient evidence (hasty generalization)
2. cause and effect are incorrectly linked (false causality)
3. selective evidence is presented (cherry-picking).

---

## 5. Re-Reading Improves Reasoning in Language Models

- Abstract : [https://arxiv.org/pdf/2309.06275.pdf](https://arxiv.org/pdf/2309.06275.pdf)

### Abstract

추론은 대형 언어 모델(LLM)에 대한 중요하고 도전적인 문제를 제기합니다. 연구의 주요 초점은 LLM의 추론 과정을 안내하고 구조화하기 위한 다양한 프롬프팅 전략을 개발하는 데 둘러싸여 왔습니다. 그러나 디코더 전용 인과 언어 모델에 기반한 이러한 접근 방식은 종종 입력 질문을 단일 순방향 패스에서 작동시키며, 이로 인해 인간 추론에 내재된 풍부한 순방향 및 역방향 상호 작용을 놓칠 수 있습니다. 입력 질문 자체가 프롬프트 내에 포함되는 중요한 차원에는 거의 주의가 기울어지지 않았습니다. 이에 대응하여 우리는 **"질문 재독"이라는 간단하지만 매우 효과적인 프롬프팅** 전략을 소개합니다. 인간 학습과 문제 해결에서 영감을 받은 재독은 입력 프롬프트 내에 포함된 질문 정보를 다시 방문하는 것을 의미합니다. 이 접근 방식은 강화의 인지 원칙과 완벽하게 일치하여 LLM이 더 깊은 통찰력을 추출하고 복잡한 패턴을 식별하며 더 세밀한 연결을 수립하고 궁극적으로 다양한 작업에서 추론 능력을 향상시킬 수 있습니다. 추론 벤치마크에 대한 실험은 우리 방법의 효과성과 일반성을 강조하기 위해 진행되었습니다. 더 나아가 우리의 결과는 우리의 접근 방식이 다양한 언어 모델, 유도 프롬프팅 방법 및 앙상블 기술과 완벽하게 통합되며, 이를 통해 LLM의 영역에서의 다양성과 호환성을 더욱 강조합니다.

### Method

이 논문은 매우 간단하다.

<img width="861" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/972ffd39-1ffe-4017-b648-9cd7c67d14b7">

<img width="790" alt="image" src="https://github.com/Paper-is-all-you-need/PaperReview/assets/70795645/1c5130da-b1dc-4dcd-a301-f4f22a6dbc61">

논문의 제목처럼 input query를 한 번 더 언급하는 것이다. 위 식에서 $z$는 rationale이다.

### What to get

본 논문은 트랜스포머의 디코더 기반인 LLM모델들은 bi-directional이 아닌 forward pass이기 때문에, 인간의 추론처럼 다시 생각을 곱씹어보는, 방법이 없다고 지적하였고, 이를 위해 RE-2 방법을 제안했으며, 이것은 매우간단하게 Question을 한 번 더 입력하는 방식을 사용했다.

내 연구에 적용한다면 prompting과정에서 계속해서 reasoning을 진행할 때, 중간중간 original 질문을 다시 한 번 써주는 방식으로 사용하면 좋지 않을까 라는 생각이 떠오른다.
