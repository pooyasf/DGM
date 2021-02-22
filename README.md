# DGM

Implementation code for Free Boundry PDE (American Option) up to 7 assets (9 dimensions) with the method discussed at https://arxiv.org/abs/1708.07469

You can also find a finite difference matlab code that is useful when you are going to see how accurate are your results.

There are two other examples in this repository. Heat equation and advection equation. You can see the animation of these two equations during training of neural net:
<br>
<p align="center">
<img src="https://github.com/pooyasf/DGM/blob/main/Advection/anim/advection_anim.gif?raw=true" width="320">


<img src="https://github.com/pooyasf/DGM/blob/main/Heat/anim/heat_anim.gif?raw=true" width="320">
</p>

<br>
This is the schematic of the building blocks of this code:
<br>

<p align="center">
<img src="https://github.com/pooyasf/DGM/blob/main/Docs/LibraryDiagram.png?raw=true" width="400" >
</p>

<br>

### Requirements

Python 3.7.7 <br>
Pytorch 1.6
