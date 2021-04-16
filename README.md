# DGM

This is a library that could facilitate experimentation with Deep Galerkin algorithm. for learning the solution you could define new PDEs/ODEs and just call the train function. The knowledge of designing an appropriate loss function for your application is required. The library outputs several useful stuff:

1- Loss function value (for differential operator, boundary condition and etc. )
2- Neural Network solution for the given equation
3- Layer by Layer mean activation value (during training) for the given neural network ( like the method dicussed in Xavier's initialization paper )

You can also find implementation code for Free Boundry PDE (American Option) up to 7 assets (9 dimensions) with the method discussed at https://arxiv.org/abs/1708.07469 .

There is also a finite difference matlab code that is useful for measuring the accuracy of your result.

There are two other examples in this repository. Heat equation and advection equation. You can see the animation of these two equations during training of neural net:
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


This is the schematic of the building blocks of this code:
<br><br>

<p align="center">
<img src="https://github.com/pooyasf/DGM/blob/main/Docs/LibraryDiagram.png?raw=true" width="400" >
</p>

<br><br>

### Requirements

Python 3.7.7 <br>
Pytorch 1.6

For collaboration , suggestion or question feel free to contact me at: pooya[dot]saffarieh[at]student[dot]sharif[dot]ir
