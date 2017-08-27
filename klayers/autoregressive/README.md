## Autoregressive recurrent layers

* **Autoregressive GRU**: implementation of an autoregressive GRU layer, that feeds each output `y_{t-1}` from the previous time step as an addition input when forming the current hidden memory state `h_t`. The implementation is based
on the existing GRU layer in keras.
	* Features:
		* pluggable `output_fn`, a callable that forms `y_t` given `h_t` at every time step: the behavior of that callable is left for the user to specify
		* the autoregressive GRU returns the `y_{1:T}` as output, and not the `h_{1:T}` (as the normal GRU does in keras)
		* pluggable `initializer`, `regularizer` and `constraint` keras objects for the auto-regressive kernel matrix (as is normally done in keras layers)
