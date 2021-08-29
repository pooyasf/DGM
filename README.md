# DGM

This library allows you to experiment with the Deep Galerkin algorithm. For finding a PDE or ODE solution you simply define a loss function. Then, by calling train(), the neural network learns the solution. It outputs several useful information:
<br>
<br>
1- Loss function value (for the differential operator, boundary condition, etc.) <br>
2- Neural Network solution for the given equation  <br>
3- Layer by Layer mean activation value (during training) for the neural network <br>
<br>
You can also find implementation code for Free Boundry PDE (American Option) up to 7 assets (9 dimensions) with the method discussed at https://arxiv.org/abs/1708.07469. There is also a finite-difference Matlab code that is useful for measuring the accuracy of your result.
<br>
In this repository, there are two low-dimensional examples: the heat equation and the advection equation. The following animation illustrates the two equations as they are trained:
<br>
<p align="center">
<img src="https://github.com/pooyasf/DGM/blob/main/Advection/anim/advection_anim.gif?raw=true" width="320">


<img src="https://github.com/pooyasf/DGM/blob/main/Heat/anim/heat_anim.gif?raw=true" width="320">
</p>

<br>

Mean activation value for different layers of the neural net (during training):
<br><br>

<p align="center">
<img src="https://github.com/pooyasf/DGM/blob/main/Docs/heat_layers_activ_value.png?raw=true" width="400" >
</p>

<br><br>


Here are the building blocks of this code:
<br><br>

<p align="center">
<img src="https://github.com/pooyasf/DGM/blob/main/Docs/LibraryDiagram.png?raw=true" width="400" >
</p>

<br><br>

### Requirements

Python 3.7.7 <br>
Pytorch 1.6

Any questions? contact me at: pooya[dot]saffarieh[at]student[dot]sharif[dot]ir
