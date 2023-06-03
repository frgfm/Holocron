Compact Convolutional Transformer
=================================

.. currentmodule:: holocron.models

The Compact Convolutional Transformer is based on the `"Escaping the Big Data Paradigm with
Compact Transformers" <https://arxiv.org/pdf/2104.05704.pdf>`_ paper.

Architecture overview
---------------------

This paper provides new ViT based architectures that are more compact and efficient than the original ViT architecture.

The key takeaways from the paper are the following:

* new sequence pooling strategy (replaces normally used CLS token), which pools over output tokens and improves performance.
* lite ViT style architectures with convolutional patch embeddings.
* increased performance and flexibility for input image sizes while also demonstrating that these variants do not depend as much on Positional Embedding.


Model builders
--------------

The following model builders can be used to instantiate a CCT model, with or
without pre-trained weights. All the model builders internally rely on the
``holocron.models.classification.cct.CCT`` base class. Please refer to the `source
code
<https://github.com/frgfm/Holocron/blob/main/holocron/models/classification/cct.py>`_ for
more details about this class.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    cct_2
    cct_4
    cct_6
    cct_7
    cct_14
