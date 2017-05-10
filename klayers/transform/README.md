## Transformation layers in keras

* **Spatial Transformer**: implementation of the spatial transformer module, as per "Spatial Transformer Networks", by Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
	* Features:
		* pluggable localization network, transformation function, interpolation function
		* affine transformation function
		* attention transformation function
		* nearest neighbour interpolation
		* bilinear interpolation
		* gaussian interpolation (with adjustable kernel size (cut-off) and kernel step (resolution of the sampling)
	* TODO:
		* thin plate spline transformation function

* TODO: augmentation layers
