ZML test: NVFP4 dequantizer in zig.

### Rules
- .safetensor only
- dequant on the fly:? chunk by chunk
- SIMD: chunk
- under the shape `std.Io.reader`: no mmap
- multithreaded: threadpool over chunks


### Questions
- rack.cpu >= 2 -> NUMA workload split?
- dequant to disk or RAM ? (translate or usage)
    - if translation, do i need output file or RAM? 
        - i could extend file end and translate from the end to top. With a "solver"/WAL to retreive partially done dequant. (eww)



### Data reconstruction
w: reconstructed weights
s: scale
q: quantized weights
z: `zero point` correspond to the median of the rang (eg [-0.1;0.5]: z=0.3)

- standard affine: w = s(q - z)
- Round To Nearest: w = q*s

