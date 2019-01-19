#!/bin/bash

# 1LPx1aVraBPpogRZXuttSb-1B4IkGuJ9m

echo "Downloading prepared enfr data..."
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1LPx1aVraBPpogRZXuttSb-1B4IkGuJ9m' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LPx1aVraBPpogRZXuttSb-1B4IkGuJ9m" -O data.tar.gz && rm -f cookies.txt

echo "Extracting parallel data..."
tar -xvzf data.tar.gz

