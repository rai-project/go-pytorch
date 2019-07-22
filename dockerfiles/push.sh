#! /bin/bash

make $1

while [ $? -ne 0 ]; do
  make $1
done
