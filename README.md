# safetensors.h

This is a public domain single-header C library for reading .safetensors files,
which are sometimes used to store machine learning models. 
It is a single-header library in the style of Sean Barrett (https://github.com/nothings/):
The file `safetensors.h` contains both the header and the library implementation.
To use this library, copy `safetensors.h` into your project, and define 
`SAFETENSORS_IMPLEMENTATION` in exactly one `.c` file, before you include it.


