---
layout: post
title:  "딥러닝을 활용한 중국어 문법 연구 - 1"
subtitle: "데이터 로더(Data Loader) 만들기"
type: "#NLP #Chinese #Grammar"
blog: true
text: true
author: "boychaboy"
post-header: true
header-img: "img/header.jpg"
order: 3
---
### 들어가며...

우리는 **중국어의 방향보어**를 [BERT](https://arxiv.org/abs/1810.04805)를 통해 학습 시켰을 때 얻을 수 있는 문법적인 인사이트를 알아보기 위해서 본 프로젝트를 시작했다. 서강대학교 중국문화 학과 강병규 교수님, 중국문화학 석사 강수민, 중국문화학 박사 이명월, 중국문화학 학사 엄윤경 그리고 컴퓨터공학 석사 정영훈 총 5명의 구성원으로 이루어진 본 스터디는 2020년 7월부터 본격적으로 시작하여 방학 때는 일주일에 한 번, 학기 중에는 이주일에 한번씩 모임이 이루어졌다.

중국어가 모국어가 아닌 사람들이 중국어를 배우는 과정에서 가장 헷갈려 하는 부분 중 하나가 바로 방향보어이다. 우리는 방향보어를 인공지능 언어모델인 [BERT](https://arxiv.org/abs/1810.04805) 로 학습시켰을 때 모델이 주목하는 문장 성분을 분석하여 그 동안 밝혀지지 못했던 언어적인 인사이트를 얻어보고자 하였다. 본 프로젝트는 아래와 같은 과정으로 이루어졌고, 이에 대한 역할 분담을 적어두었다. 

1. 방향보어가 포함된 문장을 모아서 전처리하기 : 수민, 명월
2. 방향보어 예측하는 중국어 BERT 모델 사후학습(Fine-tuning)하기 : 영훈, 윤경
   - 분류기(Classifier) 학습하기 : 영훈
   - 언어모델(Masked Language Model) 학습하기 : 윤경
3. 학습된 모델을 해석(Interpret)하기 : 영훈, 윤경

개발 과정에서 작성된 코드는 모두 [Github link](https://github.com/boychaboy/deep-chinese.git)에 공개해두었으니 참고 바란다. 



## 데이터 수집(Data Collect)

### 1. 방향보어가 포함된 문장 수집하기

- 수집 사이트 출처, 해당 출처에 있는 문장들의 특성, 사용 가능 여부 밝히기

### 2. 방향보어가 올바르게 사용된 문장 검증하기

- 검증 과정에서 선정한 기준, 해당 작업을 하는데 사용한 코드, 패키지 등등

### 3. 테스트 데이터 선정하기

- 테스트 셋의 출처, 이 문장을 선정한 기준



## 데이터 로더(Data Loader)

데이터 로딩 과정은 필자가 맡았다. 데이터 수집 과정이 끝난 후에는 아래 같은 파일 리스트가 생성되었는데, 이를 **합치고, 랜덤하게 섞어서, 학습 데이터(train data)와 검증 데이터(validation data)로 나누는 일**이었다. 

`verb_{label}_{literature|media|textbook}.txt`

`label`은 총 8가지 `上来 上去 下来 下去 过来 过去 起来 出来` 의 방향보어가 있었고, 각각 문학작품, 미디어, 그리고 교과서에서 수집한 문장이었기 때문에 위와 같은 형태로 이루어져있었다. 필자는 interpret 언어인 python의 강점을 최대한 살린 [Jupyter Notebook](https://jupyter.org)을 활용하여 데이터 로더 코드를 작성하였다. 

### 1. [MASK] 토큰으로 대치

먼저 위의 데이터를 종합하여  `data` 라는 이름의 [문장, 레이블] 리스트로 저장하는 코드는 아래와 같다. 이 과정에서 **(1) 여러 문장으로 이루어진 데이터의 경우 label이 있는 문장 단위로 분리하였고, (2) 한 문장 안에 label이 여러개 있는 경우 맨 첫번째 label만 [MASK]로 대치시켰다. 

```python
def get_data(data_dir):
    mask_len = 100
    label_dict = {'shanglai':'上来',
                  'shangqu':'上去',
                  'xialai':'下来',
                  'xiaqu':'下去',
                  'guolai':'过来',
                  'guoqu':'过去',
                  'qilai':'起来',
                  'chulai':'出来'
                 }
    label_list = ['shanglai', 'shangqu', 'xialai', 'xiaqu', 'guolai', 'guoqu', 'qilai', 'chulai']
    data = []
    for label in label_list:
        file_list = [data_dir+'/verb_'+label+'_literature.txt',
                     data_dir+'/verb_'+label+'_media.txt',
                     data_dir+'/verb_'+label+'_textbook.txt']
        for file in file_list:
            with open(file, 'r') as f:
                for line in f.readlines():
                    line = line.replace('？','。')
                    line = line.replace('!','。')
                    sent_list = line.split('。')
                    for sent in sent_list:
                        sent = sent.strip()
                        if label_dict[label] in sent:
                            sent = sent.replace(label_dict[label],'[MASK]', 1)
                            data.append([sent, label_dict[label]])
                            
    return data
```

### 2. 학습 데이터, 검증 데이터로 분리(Train, Val split)

두 번째로, 모델을 학습시킬 학습 데이터와 학습 과정에서 정확도를 검증할 검증 데이터로 분리시키는 함수를 작성하였다. 이 과정에서는 (1) 먼저 전체 데이터를 섞고(shuffle), (2) 원하는 비율(ratio)로 데이터를 분리하여 두 개의 리스트로 저장한다. 

```python
import random

def split_data(data, seed, rate):
    # shuffle data
    random.seed(seed)
    random.shuffle(data)
    
    # split data
    data_len = len(data)
    val_len = int(rate * data_len)
    train_len = data_len - val_len
    val_data = data[:val_len]
    train_data = data[val_len:]
    
    return [train_data, val_data]
```

### 3. 데이터 저장(Save .txt, .json files)

마지막으로는 생성된 리스트를 사용하기 쉽게 파일로 저장한다. 이 때 크게 두 가지 형식을 사용하는데, 어느 것을 사용해도 상관 없다. 개인적으로 느낀 두 형식의 각각의 장점, 단점은 다음과 같다. 

**텍스트 파일(.txt)**

- 장점 : 읽기, 쓰기가 편리하다. 
- 단점 : 모델에서 데이터를 불러올 때 다시 자료형(list, dict, ...)으로 전처리를 통해 바꿔주어야 한다. 

```python
# dump txt files
def dump_txt(file, file_name):
    with open(file_name+'.txt', 'w') as f:
        for line in file:
            sent = line[0].replace('[MASK]', line[1])
            f.write(sent+'\n')
```

**JSON 파일(.json)**

- 장점 : 데이터를 바로 자료형으로 불러올(load) 수 있다. 
- 단점 : 읽고 쓰는 방식이 약간(?) 복잡하다. 

```python
# dump json files
import json
def dump_json(file, file_name):
    with open(file_name+'.json', 'w') as f:
        json.dump(file, f)
```



### ...마치며

본 포스팅에서는 **딥러닝을 활용한 중국어 문법 분석**의 **데이터 로딩** 과정을 위주로 살펴보았다. 이는 이미 수집되고 가공된 데이터를 <u>리스트 형태로 만들고, 학습 및 검증 데이터로 분리</u>하는 과정에 해당한다. 

다음 포스팅에서는 이를 통해 **모델을 사후학습(Fine-tuning)**하는 방법을 다룰 예정이다. 
