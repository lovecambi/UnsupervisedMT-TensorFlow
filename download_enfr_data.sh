#!/bin/bash

echo "Downloading prepared enfr data..."
wget --load-cookies cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1E2olcmeA77MWgQt4j6j9ubB29LjIBKlU' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1E2olcmeA77MWgQt4j6j9ubB29LjIBKlU" -O data.tar.gz && rm -f cookies.txt

echo "Extracting parallel data..."
tar -xvzf data.tar.gz

