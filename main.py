from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np

app = FastAPI()

fresh_rotten_model_path = "./models/rotten_fresh.h5"
fresh_rotten_model = tf.keras.models.load_model(fresh_rotten_model_path)

fruit_vegi_model_path = "./models/fruit_vegi.h5"
fruit_vegi_model = tf.keras.models.load_model(fruit_vegi_model_path)


index_fruit_mapping = {0: 'Apple',
                       9: 'Banana',
                       8: 'Bellpepper',
                       3: 'Carrot',
                       7: 'Cucumber',
                       6: 'Mango',
                       2: 'Orange',
                       1: 'Potato',
                       4: 'Strawberry',
                       5: 'Tomato'}


@app.get('/')
def home():
    return "Upload your image to the /image endpoint"


@app.post('/predict_image')
async def predict_image(image: UploadFile = File(...)):
    image_bytes = image.file.read()
    image_tensor = tf.io.decode_image(image_bytes)

    resized_img = tf.image.resize(image_tensor, (120, 120))

    fresh_prediction = fresh_rotten_model.predict(
        np.expand_dims(resized_img/255, 0))

    fruit_vegi_prediction = fruit_vegi_model.predict(
        np.expand_dims(resized_img/255, 0))

    # check if the item isn't a fruit or not
    confidence = np.max(fruit_vegi_prediction)
    if confidence < 0.5:
        fresh = False
        item_name = 'No fruit/vegitable detected!'
    else:
        fresh = True if fresh_prediction >= 0.5 else False
        item_name = index_fruit_mapping[np.argmax(fruit_vegi_prediction)]

    print(item_name)
    return {
        "fresh": fresh,
        "item_name": item_name,
        "confidence": float(confidence)
    }
