import rdkit.Chem as Chem
import cv2
import numpy as np

def add_noise(image):
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        depth = image.shape[2]
        if depth == 3:  # RGB
            black = np.array([0, 0, 0], dtype = 'uint8')
            white = np.array([255, 255, 255], dtype = 'uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype = 'uint8')
            white = np.array([255, 255, 255, 255], dtype = 'uint8')
    probs = np.random.random(image.shape[:2])
    output[probs < .00015] = black
    output[probs > .85] = white
    return output

def extra_inchi_image(extra_inchi, extra_inchi_image_path):
    mol = Chem.MolFromInchi(extra_inchi)
    d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(300, 300)
    Chem.rdDepictor.SetPreferCoordGen(True)
    d.drawOptions().maxFontSize = 14
    d.drawOptions().multipleBondOffset = np.random.uniform(0.05, 0.2)
    d.drawOptions().useBWAtomPalette()
    d.drawOptions().bondLineWidth = 1
    d.drawOptions().additionalAtomLabelPadding = np.random.uniform(0, 0.2)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    d.WriteDrawingText(extra_inchi_image_path)
    # if crop_and_pad:
    #     img = cv2.imread(extra_inchi_image_path, cv2.IMREAD_GRAYSCALE)
    #     # crop_rows = img[~np.all(img==255, axis = 1), :]
    #     # img = crop_rows[:, ~np.all(crop_rows==255, axis = 0)]
    #     # img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value = 255)
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # else:
    #     img = cv2.imread(extra_inchi_image_path)
    # if a_n:
    #     img = add_noise(img)
    # cv2.imwrite(extra_inchi_image_path, img)
    # return img
