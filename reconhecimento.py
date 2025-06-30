import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import sys

model = load_model('modelo/modelo_digitos.h5')

def carregar_imagem(caminho):
    img = Image.open(caminho).convert('L') 
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array 
    img_array = img_array / 255.0
    return img_array.reshape(1, 28, 28)

if len(sys.argv) < 2:
    print("Uso: python reconhecimento.py imagens_teste/exemplo_1.png")
    sys.exit()

imagem = carregar_imagem(sys.argv[1])
resultado = model.predict(imagem)
print("Número previsto:", np.argmax(resultado))
