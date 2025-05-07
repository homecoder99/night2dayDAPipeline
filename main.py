# # 작업 루트
# export ROOT=/Users/$USER/Downloads
# export NIGHT_DS=$ROOT/bdd100k_yolo_night        # 제공된 야간 데이터
# export CYC=$ROOT/cyclegan_data                  # CycleGAN 형식 폴더
# mkdir -p $CYC/trainA $CYC/trainB                # night = A, day = B

# # (간단 전략) ① trainA = 모든 night 이미지
# rsync -av --include='*.jpg' --exclude='*' \
#   "$NIGHT_DS/images/train/" \
#   "$CYC/trainA/"

# # (간단 전략) ② trainB = day 라벨이 있는 이미지 일부
# # BDD100K yaml에 day/night 구분 속성이 없으므로, 우선 'val' 세트를 day 로 가정
# cp $NIGHT_DS/images/val/*.jpg  $CYC/trainB/
