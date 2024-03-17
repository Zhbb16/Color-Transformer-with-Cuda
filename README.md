Object Oriented example of Cuda programming in C++ for color transforming video frame from BGR to BT.709


The program utilizes C++ and CUDA to process video files via the OpenCV library, converting each frame from the BGR color space to Rect. 709. This conversion is performed on both the CPU and GPU, with CUDA kernels launched for the latter. The results from both computations are then displayed simultaneously. Asynchronous operations are employed for CPU and GPU computations, utilizing two separate threads managed through std::future. Additionally, the efficiency of the CUDA kernel was evaluated using Nvidia Nsight Compute for memory bandwidth optimization. CMake serves as the build system tool for the program's development.

Main Function for displaying the GPU & CPU results:
As shown in the image, I have defined two classes: one handling the video file opening and retrieving camera images, and another class for handling the color space conversion on either GPU or CPU.
Moreover, the computation on CPU & GPU are async operations using std::future.

![](imgs/mainFunction.png)

CUDA Kernel:
Block dimension is (32,32, 1) to be sure that it fits to all GPU hardwares
Grid Dimension is grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1) so that each cuda thread is mapped to each image pixel.
Note that the input image has 3 dimensional which means that each cuda thread is responsible for handling the 3 dimensional.
The mapping between cuda thread indices and pixel indices are shown below

![](imgs/CudaKernel.png)


CUDA Kernel Benchmark Test using Nsight
To make sure that the model is not suffering from uncoalesced memory access, 
NCU benchmark test is done and showing before which shows that memory bandwidth has low utilization compared to compute utilization.
![](imgs/ProfilingWithNsight.png)

Latency time for each line of Cuda Kernel is also showed below showing that math power functions has the highest latency in computations!
![](imgs/ProfilingWithNsight1.png)


