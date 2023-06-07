const express = require('express');
const sharp = require('sharp');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const multer = require('multer');
const app = express();
const cors = require('cors');

// cargar  imagen
const upload = multer({ dest: 'uploads/' });
app.use(express.json());
// app.use('/api/v1', router)
app.use(cors());
// Ruta para procesar la imagen y hacer la predicción
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    // Cargar el modelo de Keras
    // const modelPath = path.resolve(
    //   __dirname,
    //   '../model_json/model_github/model.json'
    // );
    const modelPath = path.resolve(__dirname, '../model_json/vgg19/model.json');
    const model = await tf.loadGraphModel(`file://${modelPath}`);
    console.log(req.files);
    // Cargar la imagen y preprocesarla
    const imageBuffer = path.resolve(__dirname, `../${req.file.path}`);
    const tensor = await sharp(imageBuffer)
      .resize(56, 56)
      .normalise()
      .toFormat('png')
      .toBuffer()
      .then((buffer) => tf.node.decodePng(buffer));
    const tensorFormat = tf.expandDims(tensor, 0).cast('float32');
    console.log(tensorFormat);
    // Hacer la predicción con el modelo
    const predictions = model.predict(tensorFormat);
    const predictedClass = predictions.argMax(1).dataSync();

    // Devolver la predicción al usuario
    const parsePredict = {
      0: 'A',
      1: 'B',
      2: 'C',
      3: 'D',
      4: 'E',
      5: 'F',
      6: 'G',
      7: 'H',
      8: 'I',
      9: 'J',
      10: 'L',
      11: 'K',
      12: 'M',
      13: 'N',
      14: 'O',
      15: 'P',
      16: 'Q',
      17: 'R',
      18: 'S',
      19: 'T',
      20: 'U',
      21: 'V',
      22: 'W',
      23: 'X',
      24: 'Y',
      25: 'Z',
      26: 'del',
      27: 'nothing',
      28: 'space',
    };
    res.json({ predict: parsePredict[predictedClass], value: predictedClass });
  } catch (error) {
    console.log(error);
    res.status(500).json({ message: 'Error al procesar la imagen.' });
  }
});

// Iniciar el servidor
app.listen(3000, () => console.log('Servidor iniciado en el puerto 3000'));
