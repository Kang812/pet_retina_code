# pet_retina_disease_model_develop

반려 동물의 안구 질병을 예측하기 위한 모델 개발

## Data preprocessing
```
python ./pet_retrain_code/code/make_dataframe.py
```
- 이미지 경로와 라벨로 구성이 되어져 있는 데이터 프레임 생성

```
python ./pet_retrain_code/code/disease_selection.py
```
- 스크립트에서 특정 질병을 selection한 데이터 프레임 만들고, Train/Valid/Test Split을 수행

## Model Training
```
python train.py
```
./pet_retrain_code/train.py에 있는 sys.path.append에 있는 model과 utils의 경로 수정, train이랑 valid 데이터 프레임 경로 수정 선택한 질병의 label_seq 수정 모델 저장 경로 수정

## 

고양이 결막염 모델 : https://drive.google.com/file/d/1BuAhoyiQBAvF3oBSAYbNKlLwoe-TOh3K/view?usp=share_link

데이터 출처 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=562
