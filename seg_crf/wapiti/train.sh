#!/bin/sh
Sel=female
CrfReatureRoot=../Resource/Prosody${Sel}CrfFeature/
OutputModelDir=../Resource/Prosody${Sel}CrfModel/
mkdir -p tmp
L2_params=
L3_params=-1 2 -a sgd-l1
#L2_params="-a rprop- -1 2"
#L3_Params="-a rprop- -1 2"
./wapiti train -p ${OutputModelDir}pat.txt -d $CrfReatureRoot/prosody_1.dev $CrfReatureRoot/prosody_1.train ${OutputModelDir}prosody_1.model
./wapiti label -s -c -m ${OutputModelDir}prosody_1.model $CrfReatureRoot/prosody_1.test tmp/1.out
./wapiti train $L2_params -p ${OutputModelDir}pat.txt -d $CrfReatureRoot/prosody_2.dev $CrfReatureRoot/prosody_2.train ${OutputModelDir}prosody_2.model
./wapiti label -s -c -m ${OutputModelDir}prosody_2.model $CrfReatureRoot/prosody_2.test tmp/2.out
./wapiti train $L3_Params -p ${OutputModelDir}pat.txt -d $CrfReatureRoot/prosody_3.dev $CrfReatureRoot/prosody_3.train ${OutputModelDir}prosody_3.model
echo "./wapiti train $L3_Params -p ${OutputModelDir}pat.txt -d $CrfReatureRoot/prosody_3.dev $CrfReatureRoot/prosody_3.train ${OutputModelDir}prosody_3.model"
./wapiti label -s -c -m ${OutputModelDir}prosody_3.model $CrfReatureRoot/prosody_3.test tmp/3.out
./wapiti train -p ${OutputModelDir}pat.txt -d $CrfReatureRoot/prosody_4.dev $CrfReatureRoot/prosody_4.train ${OutputModelDir}prosody_4.model
./wapiti label -s -c -m ${OutputModelDir}prosody_4.model $CrfReatureRoot/prosody_4.test tmp/4.out
