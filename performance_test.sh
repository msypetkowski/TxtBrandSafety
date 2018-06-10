#!/bin/env bash

# before running experiments, there should be index.html
# for example:
# wget "https://www.thedrinkshop.com/"

function benchmark {
    n_procs=$(seq 1 100)

    # run processes and store pids in array
    for i in $n_procs; do
        curl -X POST localhost:5000/compat -F "data=@index.html" &
        pids[${i}]=$!
    done

    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
}

time benchmark
