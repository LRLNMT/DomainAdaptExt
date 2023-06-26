#!/bin/bash

# Note: directory structure requirements:
#   run in parent directory: experiment6
#   require: ./fairseq
#   require: ../mbart50.pretrained
#   require: ../Datasets

declare -A LANG_TAG=( ["English"]="en_XX" ["Hindi"]="hi_IN" ["Gujarati"]="gu_IN" ["Kannada"]="te_IN" ["Sinhala"]="si_LK" ["Tamil"]="ta_IN")

declare -A BIBLE_LIST=(["Hindi"]="new_bible_g1" ["Gujarati"]="new_bible_g1" ["Kannada"]="new_bible_g1" ["Sinhala"]="new_bible_g2" ["Tamil"]="new_bible_g1")
declare -A PMO_LIST=(["Hindi"]="PrimeMinisterCorpus" ["Gujarati"]="PrimeMinisterCorpus" ["Kannada"]="PrimeMinisterCorpus" ["Sinhala"]="government" ["Tamil"]="government")

preprocess () {

    dev_set=$domain2

    # train data paths
    path_to_data1="../Datasets/${lang}/train/mixed/${domain1}/${size1}"
    # path_to_data2="../Datasets/${lang}/train/${domain2}/${size2}"

    perl -E "print '=' x 80 "
    printf "\n"
    echo "******************** Start PREPROCESSING for ${lang_pair} ********************\n"

    mkdir -p ${stage1_model_save_path}/encode/${domain1}
    # mkdir -p ${save_path}/encode/${domain2}
    mkdir -p ${stage1_model_save_path}/preprocess/${domain1}
    # mkdir -p ${save_path}/preprocess/${domain2}

    DICT_s="${PRETRAINED_PATH}/dict.${lang_tag_from}.txt" 
    DICT_t="${PRETRAINED_PATH}/dict.${lang_tag_to}.txt"

    # STAGE 1 Encoding
    # train
    echo "******************** Start encoding for STAGE 1 train set ********************\n"
    python fairseq/scripts/spm_encode.py \
        --model ${PRETRAINED_PATH}/sentence.bpe.model \
        --inputs ${path_to_data1}/train-${lang_tag_from}.txt ${path_to_data1}/train-${lang_tag_to}.txt  \
        --outputs ${stage1_model_save_path}/encode/${domain1}/train.spm.${lang_tag_from} ${stage1_model_save_path}/encode/${domain1}/train.spm.${lang_tag_to}
    echo "******************** Finished encoding for STAGE 1 train set ********************\n"

    # dev
    echo "******************** Start encoding for STAGE 1 dev set ********************\n"
    python fairseq/scripts/spm_encode.py \
        --model ${PRETRAINED_PATH}/sentence.bpe.model \
        --inputs ../Datasets/${lang}/dev/${dev_set}/dev-${lang_tag_from}.txt ../Datasets/${lang}/dev/${dev_set}/dev-${lang_tag_to}.txt  \
        --outputs  ${stage1_model_save_path}/encode/${domain1}/dev.spm.${lang_tag_from} ${stage1_model_save_path}/encode/${domain1}/dev.spm.${lang_tag_to}
    echo "******************** Finished encoding for STAGE 1 dev set ********************\n"

    fairseq-preprocess \
    --source-lang ${lang_tag_from} \
    --target-lang ${lang_tag_to} \
    --trainpref ${stage1_model_save_path}/encode/${domain1}/train.spm \
    --validpref ${stage1_model_save_path}/encode/${domain1}/dev.spm \
    --destdir ${stage1_model_save_path}/preprocess/${domain1} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict $DICT_s \
    --tgtdict $DICT_t \
    --workers 70 \
    --seed 222

    # STAGE 2 Encoding
    # train
    # echo "******************** Start encoding for STAGE 2 train set ********************\n"
    # python fairseq/scripts/spm_encode.py \
    #     --model ${PRETRAINED_PATH}/sentence.bpe.model \
    #     --inputs ${path_to_data2}/train-${lang_tag_from}.txt ${path_to_data2}/train-${lang_tag_to}.txt  \
    #     --outputs ${save_path}/encode/${domain2}/train.spm.${lang_tag_from} ${save_path}/encode/${domain2}/train.spm.${lang_tag_to}
    # echo "******************** Finished encoding for STAGE 2 train set ********************\n"

    # # dev
    # echo "******************** Start encoding for STAGE 2 dev set ********************\n"
    # python fairseq/scripts/spm_encode.py \
    #     --model ${PRETRAINED_PATH}/sentence.bpe.model \
    #     --inputs ../Datasets/${lang}/dev/${dev_set}/dev-${lang_tag_from}.txt ../Datasets/${lang}/dev/${dev_set}/dev-${lang_tag_to}.txt  \
    #     --outputs  ${save_path}/encode/${domain2}/dev.spm.${lang_tag_from} ${save_path}/encode/${domain2}/dev.spm.${lang_tag_to}
    # echo "******************** Finished encoding for STAGE 2 dev set ********************\n"

    # fairseq-preprocess \
    # --source-lang ${lang_tag_from} \
    # --target-lang ${lang_tag_to} \
    # --trainpref ${save_path}/encode/${domain2}/train.spm \
    # --validpref ${save_path}/encode/${domain2}/dev.spm \
    # --destdir ${save_path}/preprocess/${domain2} \
    # --thresholdtgt 0 \
    # --thresholdsrc 0 \
    # --srcdict $DICT_s \
    # --tgtdict $DICT_t \
    # --workers 70 \
    # --seed 222

    for test_set in "${domains[@]}"
    do
        mkdir -p ${stage1_model_save_path}/test/encode/${test_set}
        mkdir -p ${stage1_model_save_path}/test/preprocess/${test_set}
        mkdir -p ${stage1_model_save_path}/test/hypothesis/${domain1}/${test_set}
        # mkdir -p ${save_path}/test/hypothesis/${domain2}/${test_set}

        echo "******************** Start encoding for test set: ${test_set} ********************\n"
        python fairseq/scripts/spm_encode.py \
            --model ${PRETRAINED_PATH}/sentence.bpe.model \
            --inputs ../Datasets/${lang}/test/${test_set}/test-${lang_tag_from}.txt ../Datasets/${lang}/test/${test_set}/test-${lang_tag_to}.txt  \
            --outputs ${stage1_model_save_path}/test/encode/${test_set}/test.spm.${lang_tag_from} ${stage1_model_save_path}/test/encode/${test_set}/test.spm.${lang_tag_to}
        echo "******************** Finished encoding for test set: ${test_set} ********************\n"

        DICT_s="${PRETRAINED_PATH}/dict.${lang_tag_from}.txt"
        DICT_t="${PRETRAINED_PATH}/dict.${lang_tag_to}.txt"

        fairseq-preprocess \
        --source-lang ${lang_tag_from} \
        --target-lang ${lang_tag_to}\
        --testpref ${stage1_model_save_path}/test/encode/${test_set}/test.spm \
        --destdir ${stage1_model_save_path}/test/preprocess/${test_set} \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --srcdict $DICT_s \
        --tgtdict $DICT_t \
        --workers 70 \
        --seed 222

    done

    echo "******************** Finished PREPROCESSING for ${lang_pair} ********************\n"
}

train () {

    domain=$1
    pretrained_model=$2
    save_dir_path=$3

    path_2_data="${stage1_model_save_path}/preprocess/${domain}"
    lang_list="${PRETRAINED_PATH}/ML50_langs.txt" 

    mkdir -p $save_dir_path
    mkdir -p $save_dir_path/checkpoints

    echo "*** domain=${domain} ****\n"
    echo "*** model=${pretrained_model} ****\n"
    echo "*** save_path=${save_dir_path} ****\n"
    echo "*** path to data=${path_2_data} ****\n"

    fairseq-train $path_2_data \
    --finetune-from-model $pretrained_model \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large --layernorm-embedding \
    --task translation_multi_simple_epoch \
    --sampling-method "temperature" \
    --sampling-temperature 1 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "${lang_pair}" \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 40000 \
    --batch-size 32 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 1024 --update-freq 2 \
    --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
    --seed 222 --log-format simple --log-interval 2 \
    --save-dir ${save_dir_path}/checkpoints > ${save_dir_path}/logs 2> ${save_dir_path}/errors

}

generate () {
    lang_list="${PRETRAINED_PATH}/ML50_langs.txt"

    domain=$1
    model_path=$2
    path_2_model="${model_path}/checkpoints/checkpoint_best.pt"

    for test_set in "${domains[@]}"
    do
        path_2_data="./${stage1_model_save_path}/test/preprocess/${test_set}"
        save_dir_path="./${stage1_model_save_path}/test/hypothesis/${domain}/${test_set}"

        echo "*** domain=${domain} ****\n"
        echo "*** model=${model_path} ****\n"
        echo "*** model checkpoint=${path_2_model} ****\n"
        echo "*** path to data=${path_2_data} ****\n"
        echo "*** path to test data=${path_2_data} ****\n"
        echo "*** test save path=${save_dir_path} ****\n"
  

        # Create directory to save predictions
        save_test_path="${save_dir_path}/${lang_tag_from}_${lang_tag_to}_checkpoint_best.pt"

        echo "==============================================================================="
        echo "Testing: ${model_path}, src: ${lang_tag_from}, tgt: ${lang_tag_to}, domain: ${test_set}"
        echo "data: ${path_2_data}"
        echo "model: checkpoint_best.pt"
        fairseq-generate $path_2_data \
                --path $path_2_model \
                --task translation_multi_simple_epoch \
                --gen-subset test \
                --source-lang $lang_tag_from \
                --target-lang $lang_tag_to \
                --sacrebleu --remove-bpe 'sentencepiece' \
                --batch-size 32 \
                --encoder-langtok "src" \
                --decoder-langtok \
                --lang-dict $lang_list \
                --lang-pairs $lang_pair > "${save_test_path}.txt"

        cat "${save_test_path}.txt" | grep -P "^H" |sort -V |cut -f 3-  > "${save_test_path}.hyp"
        
    done

}

main () {

    perl -E "print '=' x 80 "
    printf "\n"

    lang=$1
    domain1=$3
    size1=$4
    domain2=$5
    # size2=$6

    # $language $reverse $mixed "$size+$size" $domain $size

    if [ "$2" -eq "0" ]
    then
        lang_tag_from="en_XX"
        lang_tag_to=${LANG_TAG[${lang}]}
    else
        lang_tag_from=${LANG_TAG[${lang}]}
        lang_tag_to="en_XX"
    fi 
   
    lang_pair="${lang_tag_from}-${lang_tag_to}"
    # save_path="mixed-${domain1}-${size1}_domain-${domain2}-${size2}_${lang_pair}"
    stage1_model_save_path="mixed-s1-${domain1}-${size1}-${domain2}_${lang_pair}"


    # mkdir -p ${save_path}
    # mkdir -p ${save_path}/test
    mkdir -p ${stage1_model_save_path}

    domains=("${PMO_LIST[${lang}]}" "${BIBLE_LIST[${lang}]}" "flores")
    PRETRAINED_PATH="../mbart50.pretrained"

    # Preprocessing
    preprocess

    # STAGE 1 Training
    perl -E "print '=' x 80 "
    printf "\n"
    echo "******************** Start STAGE 1 Training ${lang_pair} ********************\n"
    train $domain1 "${PRETRAINED_PATH}/model.pt" "./${stage1_model_save_path}/stage1_${domain1}_model"
    echo "******************** Finished STAGE 1 Training ${lang_pair} ********************\n"

    # STAGE 1 Testing
    echo "******************** Start STAGE 1 Generation ${lang_pair} ********************\n"
    generate $domain1 "./${stage1_model_save_path}/stage1_${domain1}_model"
    echo "******************** Finished STAGE 1 Generation ${lang_pair} ********************\n"

    # # STAGE 2 Training
    # perl -E "print '=' x 80 "
    # printf "\n"
    # echo "******************** Start STAGE 2 Training ${lang_pair} ********************\n"
    # train $domain2 "./${save_path}/stage1_${domain1}_model/checkpoints/checkpoint_last.pt" "./${save_path}/stage2_${domain1}-${domain2}_model"
    # echo "******************** Finished STAGE 2 Training ${lang_pair} ********************\n"

    # # STAGE 2 Testing
    # echo "******************** Start STAGE 2 Generation ${lang_pair} ********************\n"
    # generate $domain2 "./${save_path}/stage2_${domain1}-${domain2}_model"
    # echo "******************** Finished STAGE 2 Generation ${lang_pair} ********************\n"

}

language=$1
reverse=$2
dom1=$3
dom1_size=$4
dom2=$5

main $language $reverse $dom1 $dom1_size $dom2
