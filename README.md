# Code for Classification of Ependymomas

## Patching

This directory contains all files necessary for extracting patches (sub-images) from *whole-slide images* in NDPI-format.

## Custom SimSiam

This is a custom implementation of the SimSiam algorithm that is specialized to the file structure created by the modules in `Patching`.
Of note, this code depends on a implemetation of the LARS algorithm in a module `LARC`, which can be obtained from NVIDIA's github repositories.

## Custom CLAM

This is a custom implementation of the CLAM algorithm that is specialized to the data structures created by the code above. This code additionally depends on a python-implementation of the smooth-topk loss function.

# Contact

Please report any issues in the GitHub forum or contact [the author](mailto:schumany@hsu-hh.de) directly.