# BMS-Molecular-Translation
Kaggle competition of chemical image-to-text translation: A build-from-scratch cnn-transformer model for translating 2d chemical structures to InChI notations
## Project Description
The 2D skeletal formula is the structural notation of organic molecules used by organic chemists for centuries. International Chemical Identifier (InChI) is the machine-readable chemical descriptions. There exist decades of scanned image documents that can not be searched for InChI directly. The goal of this project is to build up a deep learning model that could link the molecular structure images with the InChI notations, i.e., translation from images to texts. Once successfully trained, the model could connect images with multidimensional chemical property spaces directly for chem/bioinformatics research since InChI or SMILES were usually used as input.

My first consideration is the employment of vision transformer model (ViT), which is implemented based on [this website](https://nlp.seas.harvard.edu/2018/04/03/attention.html). The validation metric used includes loss value and BLEU score on the obs randomly selected from validation dataset (CV).

## Reference
@misc{bms-molecular-translation,
    author = {Addison Howard, inversion, Jacob Albrecht, Yvette},
    title = {Bristol-Myers Squibb – Molecular Translation},
    publisher = {Kaggle},
    year = {2021},
    url = {https://kaggle.com/competitions/bms-molecular-translation}
}
@misc{2010.11929,
Author = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
Title = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
Year = {2020},
Eprint = {arXiv:2010.11929},
}
