# Simple model inversion

These notebooks demonstrate the original method proposed by Fredrikson et al in 2015 (see Paper section below).

## How to run

Open the `simplemodel.ipynb` Jupyter notebook and run the cells in order. The first code cell will install packages, but the requirements.txt file could be used instead (`pip install -r requirements.txt`).

There is also a torch version (`simplemodel_torch.ipynb`), less documented but similarly functional. Produces slightly different results.

The notebooks were tested on Python 3.11.9. A virtual environment is recommended.

## Dataset

The training and testing data is available in the faces directory. It consists of greyscale images in `pgm` format. It is small enough to run on CPU, adequate for demonstration purposes.

You can check out `simplemodel_torch_celeba.ipynb` which uses the bigger and colorful CelebA dataset (downloaded automatically), but due to technical constraints, I couldn't manage to make it work.

## References

The following GitHub repositories were used to help the implementation:

* [sarahsimionescu/simple-model-inversion](https://github.com/sarahsimionescu/simple-model-inversion)
* [Djiffit/face-decoding-with-model-inversion](https://github.com/Djiffit/face-decoding-with-model-inversion/blob/054bc93fbe405381564dc0fac50d94783c6b385e/inversion.ipynb)

## Paper

Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. 2015. Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security (CCS '15). Association for Computing Machinery, New York, NY, USA, 1322–1333. https://doi.org/10.1145/2810103.2813677
