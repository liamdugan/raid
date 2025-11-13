#!/bin/bash
# hacky bash script to copy all results files over and then index them

copy_split() {
  split=$1
  mkdir -p web/dist/data/$split
  outfiles=()
  for name in leaderboard/$split/*/results.json
  do
    subname="$(basename -- "$(dirname -- "$name")").json"
    echo "copying $name to web/dist/data/$split/$subname"
    jq -c . "$name" > "web/dist/data/$split/$subname"
    outfiles+=("/data/$split/$subname")
  done
  echo "writing web/dist/data/$split/_index.json"
  printf '%s\n' "${outfiles[@]}" | jq -R . | jq -s . > web/dist/data/$split/_index.json
}

copy_split "submissions"
# copy_split "shared-task"  # this is static in the web/public/data/shared-task dir
