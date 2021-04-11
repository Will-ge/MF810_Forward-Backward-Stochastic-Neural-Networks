# Forward-Backward-Stochastic-Neural-Networks
This is a forward-backward Stochastic Neural Networks used to solve PDEs with higher dimensionality in Tensorflow 2.0.
The deep neural networks is built by 6 layers, with 101, 256, 256, 256, 256, 1 nodes each. Using gradient descent optimization method while training.

The application model is Black-Scholes-Barenblatt model which have an analytical solution. This helps monitor the difference between deep neural networks' outcome with the exact solution.
The result below is obtained after 2k, 3k, 3k, 2k consecutive iterations with learning rates of 10-3, 10-4, 10-5 and 10-6 respectively.
It is recommended to run the codes on Colab to use GPU acceleration and huge memory if your PC configuration is not that good.

![image](https://user-images.githubusercontent.com/71861810/114291681-568f5b80-9a57-11eb-8675-3a24738455fd.png)
![image](https://user-images.githubusercontent.com/71861810/114291851-a4589380-9a58-11eb-83cb-9ba8ad25dd34.png)


