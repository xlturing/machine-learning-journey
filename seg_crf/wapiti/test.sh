#!/bin/sh
Sel=female_EN
CrfReatureRoot=../Resource/Prosody${Sel}CrfFeature/
OutputModelDir=../Resource/Prosody${Sel}CrfModel/
mkdir -p tmp
#L2_params=
L2_params="-a rprop- -1 2"
L3_Params="-a rprop- -1 2"
./wapiti label -s -c -m ${OutputModelDir}prosody_1.model $CrfReatureRoot/prosody_1.test tmp/1.out
./wapiti label -s -c -m ${OutputModelDir}prosody_2.model $CrfReatureRoot/prosody_2.test tmp/2.out
./wapiti label -s -c -m ${OutputModelDir}prosody_3.model $CrfReatureRoot/prosody_3.test tmp/3.out
./wapiti label -s -c -m ${OutputModelDir}prosody_4.model $CrfReatureRoot/prosody_4.test tmp/4.out
