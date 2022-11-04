#!/bin/zsh
#search using the criteria
search=$(cat ~/Downloads/search_opengl.csv)
find ../Data/ -type f -name "*.jsonl" | xargs grep -E $search > opengl_files.jsonl
awk -F 'jsonl:' '{print $2}' opengl_files.jsonl > opengl_files1.jsonl

# find repo details from output