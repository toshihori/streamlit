import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image


if __name__ == '__main__':

    st.title('MNISTのラベル予測')
    model = keras.models.load_model("my_model.h5")
    load_data = st.file_uploader("ラベルを予測する画像ファイルをアップロード", type='jpg')
    if load_data is not None:
        image = Image.open(load_data)
        img_array = np.array(image)
        x_pred = img_array/255
        y_pred = model.predict(x_pred[np.newaxis, :,:,np.newaxis])
        st.write('推論結果')
        st.write(f"画像のラベルは{np.argmax(y_pred)}です")