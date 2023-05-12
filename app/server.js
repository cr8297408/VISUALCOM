const express = require('express');
const sharp = require('sharp');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require("path");
const multer  = require('multer')
const app = express();

// cargar  imagen
const upload = multer({ dest: 'uploads/' })

// Ruta para procesar la imagen y hacer la predicción
app.post('/predict', upload.single('image') ,async (req, res) => {
  try {
    // Cargar el modelo de Keras
    const modelPath = path.resolve(__dirname, '../model_json/model.json');
    const model = await tf.loadGraphModel(`file://${modelPath}`);
    // Cargar la imagen y preprocesarla
    const imageBuffer = path.resolve(__dirname, `../${req.file.path}`);
    const tensor = await sharp(imageBuffer)
      .resize(28, 28)
      .normalise()
      .toFormat('png')
      .toBuffer()
      .then((buffer) => tf.node.decodePng(buffer))
      const tensorFormat = tf.expandDims(tensor, 0).cast('float32');
      console.log(tensorFormat)
    // Hacer la predicción con el modelo
    const predictions = model.predict(tensorFormat);
    const predictedClass = predictions.argMax(1).dataSync()[0];

    // Devolver la predicción al usuario
    res.json({ predictedClass });
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: 'Error al procesar la imagen.' });
  }
});

// Iniciar el servidor
app.listen(3000, () => console.log('Servidor iniciado en el puerto 3000'));
