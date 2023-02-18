# pet_retina_disease_model_develop

반려 동물의 안구 질병을 예측하기 위한 모델 개발을 위한 장소

1. 해당 질병의 DataFrame을 만들 것 ./pet_retrain_code/code/make_dataframe.py 경로를 수정해서 만들면된다. 전체 고양의 데이터 이미지 경로와 라벨이 있는 데이터 프레임을 만듬. -> 경로 수정
2. 1번에서 만든 전체 데이터 프레임을 선택해서 ./pet_retrain_code/code/disease_selection.py 스크립트에서 특정 질병을 selection한 데이터 프레임 만들기 -> 경로 수정하면됨
   여기서 train, valid, test 데이터 셋으로 나눔
3. ./pet_retrain_code/train.py에 있는 sys.path.append에 있는 model과 utils의 경로 수정, train이랑 valid 데이터 프레임 경로 수정 선택한 질병의 label_seq 수정 모델 저장 경로 수정

