# FNO-Stress Strain

Using a neural operator based framework- Fourier Neural Operator [FNO](https://arxiv.org/abs/2010.08895)  to learn the stress strain fields for 2D Composites. 

### Stress-Strain Prediction
We use the FNO framework to predcit components of Stress and Strain Tensors
$$\hspace{10cm}  Stress := \sigma_{xx} ,\sigma_{yy},\sigma_{xy}$$
$$\hspace{10cm}  Strain := \varepsilon_{xx} ,\varepsilon_{yy},\varepsilon_{xy}$$

The predicted tensor components are used to measure the equivalent properties like Von-mises stress
and Equilvalent Strains.


### Benchmarking

We compare the results of the FNO with ResNet and UNet architectures.



#### Dataset

The Dataset used are available at FNO-DATASET. It contains:
$$E := [Sample, H, W]$$
$$Strains := [Sample,H,W, component]$$
$$Stresses := [Sample, H,W, component]$$






Shield: [![CC BY NC SA][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by-nc-sa].

[![CC BY NC SA][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
