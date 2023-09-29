# BMS-Molecular-Translation
Kaggle competition of chemical image-to-text translation: A build-from-scratch cnn-transformer model for translating 2d chemical structures to InChI notations
## Project Description
The 2D skeletal formula is the structural notation of organic molecules used by organic chemists for centuries. International Chemical Identifier (InChI) is the machine-readable chemical description. There exist decades of scanned image documents that can not be searched for InChI directly. The goal of this project is to build up a deep learning model that could link the molecular structure images with the InChI notations, i.e., translation from images to texts. Once successfully trained, the model could connect images with multidimensional chemical property spaces directly for chem/bioinformatics research since InChI or SMILES were usually used as input.

My first consideration is the employment of vision transformer model (ViT), which is implemented based on [this website](https://nlp.seas.harvard.edu/2018/04/03/attention.html) using pytorch. The ViT model was first applied on the MNIST dataset aiming to translate digit images to spanish and 98% accuracy of prediction was observed after training on colab & GCP. Then the model was applied on the InChI translation task, the validation metric used includes loss value and BLEU score on the obs randomly selected from validation dataset (CV). However, the decrease of loss was slow and BLEU score was oscillated between 20 and 30. ViT divids images into patches and encode those patches. Such treatment looks simple and does not take the convolution of pixels into consideration, which may cause underfitting.

Then I would like to use CNN as the encoder. Thanks to torchvision, I can use the ResNet18 architecture with pretrained parameters. The last two layers of ResNet18 were ignored and the decoder part of ViT was kept the same. The data employed, NO. of epochs, and batch size was restricted due to the limited source of computation. However, the BLEU score was able to reach 50+ for the CNN-ViT model.

Further hypertuning and extensive training using sufficient GPU/TPU sources could lead to >90 BLEU score and I will keep working on that and updating here. Another consideration is to try to change the cnn architecture or connect cnn to LSTM instead of transformer.
## Repository Structure
* model/:the architecture of transformer and resnet18/transformer.
* helper-files/:utils that help to train, bachify, summarize training state, load MNIST ds, tokenization, and greedy decode.
* InChI/:preprocess InChI train and extra ds, including image transformation, extra chemicals images generation using RDKit, tokenization of InChI text, etc.
* Run/check MNIST_work_notebook.ipynb to apply transformer on MNIST. The trained model is saved as mnistfinalmodel.pt.
* Run/check InChI_work_notebook.ipynb to apply transformer on InChI train and extra.
## Notes
Download the train and extra data from [kaggle](https://www.kaggle.com/competitions/bms-molecular-translation) to the created folder bms-molecular-translation. You have to go into the .py code to change if you want to name the folder for kaggle ds differently.

Move all files, including work notebook, model, helper, and InChI under one directory before running work_notebook.ipynb unless sys.path.append is used.

Environment.yml will be updated.
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

For InChI text and image preprocessing:

https://github.com/kozodoi/BMS_Molecular_Translation/tree/main

https://www.kaggle.com/code/yasufuminakama/inchi-preprocess-2/notebook

https://www.kaggle.com/code/tuckerarrants/inchi-allowed-external-data/notebook
