#!/bin/bash

if [ ! -f data.zip ]; then
    wget -O data.zip https://cmu.box.com/shared/static/ebklocte1t2wl4dja61dg1ct285l65md.zip
fi
if [ ! -f images.zip ]; then
    wget -O images.zip https://cmu.box.com/shared/static/147jyd6uewh1fff96yfh6e11s98v11xm.zip
fi
unzip data.zip -d ../data/
unzip images.zip -d ../images/
rm images.zip
rm data.zip
