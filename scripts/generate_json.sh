#!/bin/bash
PATTERN=$1
CURR=/home/gustavoe/gitlab/recsys_input_obfuscation/scripts
cd /share/hel/datasets/cf_obfuscation/ 
find $PATTERN -type d  > /home/gustavoe/gitlab/recsys_input_obfuscation/scripts/$2 
cd /home/gustavoe/gitlab/recsys_input_obfuscation/scripts
python transformlist.py --file $2

# generate_json.sh ml-1m-1000_*0.3 ml-1m-1000-all
# generate_json.sh lfm-100k-1000_*0.3 lfm-100k-1000-all