#!/bin/bash

eps=8
epst=8
python -u bias_variance.py > "tmp/ad_aug_bias_variance/bias_variance_eps="$eps"_epst="$epst"_formal.out" 2>&1 &
