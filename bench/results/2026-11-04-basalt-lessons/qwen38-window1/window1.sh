#!/bin/bash
cd /tmp/basalt-prof
./batch9.sh > batch9.out 2>&1
./batch8.sh > batch8.out 2>&1
./batch6.sh > batch6.out 2>&1
./batch7.sh > batch7.out 2>&1
./batch10.sh > batch10.out 2>&1
echo WINDOW DONE
