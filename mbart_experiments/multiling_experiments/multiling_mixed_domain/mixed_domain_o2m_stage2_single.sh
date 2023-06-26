#!/bin/bash

# Note: directory structure requirements:
#   run in parent directory: experiment4
#   require: ./fairseq
#   require: ../mbart50.pretrained
#   require: ../Datasets

declare -A LANG_TAG=( ["English"]="en_XX" ["Hindi"]="hi_IN" ["Gujarati"]="gu_IN" ["Kannada"]="te_IN" ["Sinhala"]="si_LK" ["Tamil"]="ta_IN")

declare -A BIBLE_LIST=(["Hindi"]="new_bible_g1" ["Gujarati"]="new_bible_g1" ["Kannada"]="new_bible_g1" ["Sinhala"]="new_bible_g2" ["Tamil"]="new_bible_g1")
declare -A PMO_LIST=(["Hindi"]="PrimeMinisterCorpus" ["Gujarati"]="PrimeMinisterCorpus" ["Kannada"]="PrimeMinisterCorpus" ["Sinhala"]="government" ["Tamil"]="government")

declare -A DEV_LIST=(["cc_aligned"]="flores" ["PrimeMinisterCorpus"]="PrimeMinisterCorpus" ["government"]="government" ["new_bible_g1"]="new_bible_g1" ["new_bible_g2"]="new_bible_g2")

preprocess () {
    perl -E "print '=' x 80 "
    printf "\n"
    echo "******************** Start PREPROCESSING for ${lang_pair_list} ********************\n"

    count=0
    while [ $count -ne $(($language_num)) ]
    do
        lang_tag_from=${lang_tag_from_array[$count]}
        lang=${lang_array[$count]}
        lang_tag_to=${lang_tag_to_array[$count]}
        echo $lang_tag_from
        echo $lang
        echo $lang_tag_to
        echo "******************** Start PREPROCESSING for ${lang_pairs[$count]} ********************\n"
        # train data paths
        # path_to_data1="../Datasets/${lang}/train/${domain1}/${size1}"
        path_to_data2="../Datasets/${lang}/train/${domain2}/${size2}"

        # mkdir -p ${save_path}/encode/${domain1}
        mkdir -p ${save_path}/encode/${domain2}
        # mkdir -p ${save_path}/preprocess/${domain1}
        mkdir -p ${save_path}/preprocess/${domain2}

        DICT_s="${PRETRAINED_PATH}/dict.${lang_tag_from}.txt" 
        DICT_t="${PRETRAINED_PATH}/dict.${lang_tag_to}.txt"

        # # STAGE 1 Encoding
        # # train
        # echo "******************** Start encoding for STAGE 1 train set ********************\n"
        # python fairseq/scripts/spm_encode.py \
        #     --model ${PRETRAINED_PATH}/sentence.bpe.model \
        #     --inputs ${path_to_data1}/train-${lang_tag_from}.txt ${path_to_data1}/train-${lang_tag_to}.txt  \
        #     --outputs ${save_path}/encode/${domain1}/train.spm.${lang_tag_from} ${save_path}/encode/${domain1}/train.spm.${lang_tag_to}
        # echo "******************** Finished encoding for STAGE 1 train set ********************\n"

        # # dev
        # echo "******************** Start encoding for STAGE 1 dev set ********************\n"
        # python fairseq/scripts/spm_encode.py \
        #     --model ${PRETRAINED_PATH}/sentence.bpe.model \
        #     --inputs ../Datasets/${lang}/dev/${DEV_LIST[${domain1}]}/dev-${lang_tag_from}.txt ../Datasets/${lang}/dev/${DEV_LIST[${domain1}]}/dev-${lang_tag_to}.txt  \
        #     --outputs  ${save_path}/encode/${domain1}/dev.spm.${lang_tag_from} ${save_path}/encode/${domain1}/dev.spm.${lang_tag_to}
        # echo "******************** Finished encoding for STAGE 1 dev set ********************\n"

        # fairseq-preprocess \
        # --source-lang ${lang_tag_from} \
        # --target-lang ${lang_tag_to} \
        # --trainpref ${save_path}/encode/${domain1}/train.spm \
        # --validpref ${save_path}/encode/${domain1}/dev.spm \
        # --destdir ${save_path}/preprocess/${domain1} \
        # --thresholdtgt 0 \
        # --thresholdsrc 0 \
        # --srcdict $DICT_s \
        # --tgtdict $DICT_t \
        # --workers 70 \
        # --seed 222

        # domains=("${PMO_LIST[${lang}]}" "${BIBLE_LIST[${lang}]}" "flores")
        # echo "Test domains: ${domains}"
        # for test_set in "${domains[@]}"
        # do
        #     mkdir -p ${save_path}/test/encode/${test_set}
        #     mkdir -p ${save_path}/test/preprocess/${test_set}
        #     mkdir -p ${save_path}/test/hypothesis/${domain1}/${test_set}
        #     mkdir -p ${save_path}/test/hypothesis/${domain2}/${test_set}

        #     echo "******************** Start encoding for test set: ${test_set} ********************\n"
        #     python fairseq/scripts/spm_encode.py \
        #         --model ${PRETRAINED_PATH}/sentence.bpe.model \
        #         --inputs ../Datasets/${lang}/test/${test_set}/test-${lang_tag_from}.txt ../Datasets/${lang}/test/${test_set}/test-${lang_tag_to}.txt  \
        #         --outputs ${save_path}/test/encode/${test_set}/test.spm.${lang_tag_from} ${save_path}/test/encode/${test_set}/test.spm.${lang_tag_to}
        #     echo "******************** Finished encoding for test set: ${test_set} ********************\n"

        #     DICT_s="${PRETRAINED_PATH}/dict.${lang_tag_from}.txt"
        #     DICT_t="${PRETRAINED_PATH}/dict.${lang_tag_to}.txt"

        #     fairseq-preprocess \
        #     --source-lang ${lang_tag_from} \
        #     --target-lang ${lang_tag_to} \
        #     --testpref ${save_path}/test/encode/${test_set}/test.spm \
        #     --destdir ${save_path}/test/preprocess/${test_set} \
        #     --thresholdtgt 0 \
        #     --thresholdsrc 0 \
        #     --srcdict $DICT_s \
        #     --tgtdict $DICT_t \
        #     --workers 70 \
        #     --seed 222
        # done
        # STAGE 2 Encoding
        # train
        echo "******************** Start encoding for STAGE 2 train set ********************\n"
        python fairseq/scripts/spm_encode.py \
            --model ${PRETRAINED_PATH}/sentence.bpe.model \
            --inputs ${path_to_data2}/train-${lang_tag_from}.txt ${path_to_data2}/train-${lang_tag_to}.txt  \
            --outputs ${save_path}/encode/${domain2}/train.spm.${lang_tag_from} ${save_path}/encode/${domain2}/train.spm.${lang_tag_to}
        echo "******************** Finished encoding for STAGE 2 train set ********************\n"

        # dev
        echo "******************** Start encoding for STAGE 2 dev set ********************\n"
        python fairseq/scripts/spm_encode.py \
            --model ${PRETRAINED_PATH}/sentence.bpe.model \
            --inputs ../Datasets/${lang}/dev/${DEV_LIST[${domain2}]}/dev-${lang_tag_from}.txt ../Datasets/${lang}/dev/${DEV_LIST[${domain2}]}/dev-${lang_tag_to}.txt  \
            --outputs  ${save_path}/encode/${domain2}/dev.spm.${lang_tag_from} ${save_path}/encode/${domain2}/dev.spm.${lang_tag_to}
        echo "******************** Finished encoding for STAGE 2 dev set ********************\n"

        fairseq-preprocess \
        --source-lang ${lang_tag_from} \
        --target-lang ${lang_tag_to} \
        --trainpref ${save_path}/encode/${domain2}/train.spm \
        --validpref ${save_path}/encode/${domain2}/dev.spm \
        --destdir ${save_path}/preprocess/${domain2} \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --srcdict $DICT_s \
        --tgtdict $DICT_t \
        --workers 70 \
        --seed 222

        domains=("${PMO_LIST[${lang}]}" "${BIBLE_LIST[${lang}]}" "flores")
        echo "Test domains: ${domains}"
        for test_set in "${domains[@]}"
        do
            mkdir -p ${save_path}/test/encode/${test_set}
            mkdir -p ${save_path}/test/preprocess/${test_set}
            mkdir -p ${save_path}/test/hypothesis/${domain1}/${test_set}
            mkdir -p ${save_path}/test/hypothesis/${domain2}/${test_set}

            echo "******************** Start encoding for test set: ${test_set} ********************\n"
            python fairseq/scripts/spm_encode.py \
                --model ${PRETRAINED_PATH}/sentence.bpe.model \
                --inputs ../Datasets/${lang}/test/${test_set}/test-${lang_tag_from}.txt ../Datasets/${lang}/test/${test_set}/test-${lang_tag_to}.txt  \
                --outputs ${save_path}/test/encode/${test_set}/test.spm.${lang_tag_from} ${save_path}/test/encode/${test_set}/test.spm.${lang_tag_to}
            echo "******************** Finished encoding for test set: ${test_set} ********************\n"

            DICT_s="${PRETRAINED_PATH}/dict.${lang_tag_from}.txt"
            DICT_t="${PRETRAINED_PATH}/dict.${lang_tag_to}.txt"

            fairseq-preprocess \
            --source-lang ${lang_tag_from} \
            --target-lang ${lang_tag_to}\
            --testpref ${save_path}/test/encode/${test_set}/test.spm \
            --destdir ${save_path}/test/preprocess/${test_set} \
            --thresholdtgt 0 \
            --thresholdsrc 0 \
            --srcdict $DICT_s \
            --tgtdict $DICT_t \
            --workers 70 \
            --seed 222
        done
        count=$(($count+1))
    done

    echo "******************** Finished PREPROCESSING for ${lang_pair_list} ********************\n"
}

train () {

    domain=$1
    pretrained_model=$2
    save_dir_path=$3

    path_2_data="${save_path}/preprocess/${domain}"
    lang_list="${PRETRAINED_PATH}/ML50_langs.txt" 

    mkdir -p $save_dir_path
    mkdir -p $save_dir_path/checkpoints

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
    --lang-pairs "$lang_pair_list" \
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

    path_2_model="./${model_path}/checkpoints/checkpoint_best.pt"    
    count=0
    while [ $count -ne $(($language_num)) ]
    do
        lang_tag_from=${lang_tag_from_array[$count]}
        lang=${lang_array[$count]}
        lang_tag_to=${lang_tag_to_array[$count]}
        lang_pair=${lang_pairs[$count]}
        echo $lang_tag_from
        echo $lang
        echo $lang_tag_to
        echo $lang_pair
        domains=("${PMO_LIST[${lang}]}" "${BIBLE_LIST[${lang}]}" "flores")
        echo "Test domains: ${domains}"
        for test_set in "${domains[@]}"
        do
            path_2_data="./${save_path}/test/preprocess/${test_set}"
            save_dir_path="./${save_path}/test/hypothesis/${domain}/${test_set}"

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
            sacrebleu -tok spm -s none "../Datasets/${lang}/test/${test_set}/test-${lang_tag_to}.txt" < "${save_test_path}.hyp" -m chrf bleu > "${save_dir_path}/scores_${lang_pair}.json" 
            
        done
        count=$(($count+1))
    done
}

main () {
    perl -E "print '=' x 80 "
    printf "\n"
    echo "lang_array: ${lang_array[@]}"
    echo "lang_pairs: ${lang_pairs[@]}"
    echo "lang_pair_list: ${lang_pair_list}"
    echo "lang_tag_from_array: ${lang_tag_from_array[@]}"
    echo "lang_tag_to_array: ${lang_tag_to_array[@]}"
    echo "Stage 1 Domain: ${domain1}"
    echo "Stage 1 size: ${size1}"
    echo "Stage 2 Domain: ${domain2}"
    echo "Stage 2 size: ${size2}"

    stage1_save_path="multi-mixed-stage1-${domain1}-${size1}${save_path_postfix}"
    save_path="multi-mixed-stage1-${domain1}-${size1}_stage2-${domain2}-${size2}${save_path_postfix}"
    echo "Save path: ${save_path}"
    mkdir -p ${save_path}
    mkdir -p ${save_path}/test

    PRETRAINED_PATH="../mbart50.pretrained"

    # Preprocessing
    preprocess

    # # STAGE 1 Training
    # perl -E "print '=' x 80 "
    # printf "\n"
    # echo "******************** Start STAGE 1 Training ${lang_pair_list} using ${domain1}-${size1} ********************\n"
    # train $domain1 "${PRETRAINED_PATH}/model.pt" "./${save_path}/stage1_${domain1}_model"
    # echo "******************** Finished STAGE 1 Training ${lang_pair_list} using ${domain1}-${size1} ********************\n"

    # # STAGE 1 Testing
    # echo "******************** Start STAGE 1 Generation ${lang_pair_list} ********************\n"
    # generate $domain1 "./${save_path}/stage1_${domain1}_model"
    # echo "******************** Finished STAGE 1 Generation ${lang_pair_list} ********************\n"
    
    # STAGE 2 Training
    perl -E "print '=' x 80 "
    printf "\n"
    echo "******************** Start STAGE 2 Training ${lang_pair_list} using ${domain2}-${size2} ********************\n"
    train $domain2 "./${stage1_save_path}/stage1_${domain1}_model/checkpoints/checkpoint_best.pt" "./${save_path}/stage2_${domain1}-${domain2}_model"
    echo "******************** Finished STAGE 2 Training ${lang_pair_list} using ${domain2}-${size2} ********************\n"

    # STAGE 2 Testing
    echo "******************** Start STAGE 2 Generation ${lang_pair_list} ********************\n"
    generate $domain2 "./${save_path}/stage2_${domain1}-${domain2}_model"
    echo "******************** Finished STAGE 2 Generation ${lang_pair_list} ********************\n"
}

language_num=$1
reverse=$2
save_path_postfix=""
lang_pair_list=""
lang_tag_from_array=()
lang_array=()
lang_pairs=()
lang_tag_to_array=()
count=0
while [ $count -ne $(($language_num)) ]
do
    i=$((3+$count))
    lang_array+=(${!i})
    if [ "$2" -eq "0" ]
    then
        lang_tag_from_array+=("en_XX")
        lang_tag_to_array+=(${LANG_TAG[${!i}]})
    else
        lang_tag_from_array+=(${LANG_TAG[${!i}]})
        lang_tag_to_array+=("en_XX")
    fi
    lang_pairs+=("${lang_tag_from_array[$count]}-${lang_tag_to_array[$count]}")
    save_path_postfix+="_${lang_tag_from_array[$count]}-${lang_tag_to_array[$count]}"
    if [ "$count" -eq "0" ]
    then
        lang_pair_list="${lang_tag_from_array[$count]}-${lang_tag_to_array[$count]}"
    else
        lang_pair_list+=",${lang_tag_from_array[$count]}-${lang_tag_to_array[$count]}"
    fi
    count=$(($count+1))
done
i=$((3+$count))
domain1=${!i}
i=$((4+$count))
size1=${!i}
i=$((5+$count))
s2lang=${!i}
i=$((6+$count))
domain2=${!i}
i=$((7+$count))
size2=${!i}
main
