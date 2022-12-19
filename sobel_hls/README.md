# Sobel Edge Detector

Optimization of Sobel algorithm using Xilinx Vitis HLS flow.

Gx is:

| -1 | 0 | +1 |
| -  | - | -  |
| -2 | 0 | +2 |
| -1 | 0 | +1 |

And Gy is: 

| +1 | +2 | +1 |
| -  | -  | -  |
| 0  |  0 | 0  |
| -1 | -1 | -1 |

The formula for compute edge is : 
$\lvert G\rvert$ = $\sqrt{Gx^2 + Gy^2}$