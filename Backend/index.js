// index.js

// 1) Imports
const express   = require('express');
const cors      = require('cors');
const multer    = require('multer');
const admin     = require('firebase-admin');
const canvas    = require('canvas');
const faceapi   = require('face-api.js');
const tf        = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-cpu');
const path      = require('path');

// — Inicializa Firebase —
admin.initializeApp({
  credential: admin.credential.cert(require('./serviceAccountKey.json')),
  storageBucket: 'crud-fdmv-2025.firebasestorage.app'
});
const db     = admin.firestore();
const bucket = admin.storage().bucket();

// — Monkey-patch canvas para face-api en Node —
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// 2) Express + middleware
const app    = express();
const upload = multer({ storage: multer.memoryStorage() });
app.use(cors());

// Array global para descriptores precargados
let knownFaces = [];

// 3) Ruta de reconocimiento facial
app.post('/upload', upload.single('image'), async (req, res) => {
  console.log('--- NUEVA PETICIÓN /upload ---');
  console.log('▷ Headers recibidos:', JSON.stringify(req.headers, null, 2));
  console.log('▷ Query params:', req.query);

  if (!req.file) {
    console.log('🚫 No se recibió ningún archivo en la petición');
    console.log('🔙 Respondiendo matches vacíos');
    return res.json({ matches: [] });
  }

  console.log(`📩 Archivo recibido:
    fieldname    = ${req.file.fieldname}
    originalname = ${req.file.originalname}
    mimetype     = ${req.file.mimetype}
    size         = ${req.file.size} bytes
  `);

  try {
    console.log('🔎 Cargando imagen a canvas…');
    const uploadImg = await canvas.loadImage(req.file.buffer);

    console.log('🧠 Detectando rostro en la imagen subida…');
    const uploadDet = await faceapi
      .detectSingleFace(uploadImg)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!uploadDet) {
      console.log('⚠️ No se detectó rostro en la imagen');
      console.log('🔙 Respondiendo matches vacíos');
      return res.json({ matches: [] });
    }
    console.log('✔️ Rostro detectado (descriptor length =', uploadDet.descriptor.length, ')');

    console.log(`🔍 Comparando con ${knownFaces.length} caras precargadas…`);
    let best = { name: null, distance: 1 };
    for (let face of knownFaces) {
      const d = faceapi.euclideanDistance(uploadDet.descriptor, face.descriptor);
      console.log(`    • ${face.name} → distance = ${d.toFixed(3)}`);
      if (d < best.distance) best = { name: face.name, distance: d };
    }
    console.log(`🏆 Mejor match: ${best.name} (distance = ${best.distance.toFixed(3)})`);

    if (!best.name || best.distance > 0.6) {
      console.log('❌ Ninguna coincidencia fiable (threshold = 0.6)');
      console.log('🔙 Respondiendo matches vacíos');
      return res.json({ matches: [] });
    }

    console.log(`📚 Consultando Firestore doc "${best.name}"…`);
    const doc = await db.collection('tbl_face1').doc(best.name).get();
    if (!doc.exists) {
      console.log(`❌ Documento no encontrado para "${best.name}"`);
      console.log('🔙 Respondiendo matches vacíos');
      return res.json({ matches: [] });
    }

    const data = doc.data();
    console.log('📄 Datos recuperados de Firestore:', data);

    const match = {
      label:           data.label,
      fechaNacimiento: data.fechaNacimiento,
      tlfEmergencia:   data.tlfEmergencia,
      cedula:          data.cedula,
      imgUrl:          data.imgUrl,
      distance:        parseFloat(best.distance.toFixed(3))
    };

    const payload = { matches: [ match ] };
    console.log('📦 Payload de respuesta (JSON):\n', JSON.stringify(payload, null, 2));
    console.log('➡️ Enviando respuesta 200 OK');

    return res.json(payload);
  } catch (err) {
    console.error('💥 Error en /upload:', err);
    console.log('🔙 Respondiendo 500 con matches vacíos');
    return res.status(500).json({ matches: [] });
  }
});

// 4) IIFE para cargar modelos y cara conocidas, luego arranca el servidor
;(async () => {
  console.log('🔄 Inicializando TensorFlow backend CPU...');
  await tf.setBackend('cpu');

  const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights';
  console.log('🔄 Cargando modelos desde', MODEL_URL);
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
  console.log('🧠 Modelos cargados');

  console.log('🔍 Precargando caras conocidas de Firebase Storage...');
  const [files] = await bucket.getFiles({ prefix: 'img/' });

  for (let file of files) {
    const fullName = file.name.split('/').pop();
    const baseName = fullName.split('.').shift();
    console.log(`  • Procesando ${fullName}`);

    try {
      const [metadata] = await file.getMetadata();
      const contentType = metadata.contentType || '';
      if (!contentType.startsWith('image/')) {
        console.log(`    ⏭ Saltado (no es imagen): ${fullName} — ${contentType}`);
        continue;
      }

      const [buffer] = await file.download();
      console.log(`    📥 Descargado ${fullName} (${buffer.length} bytes)`);

      const img = await canvas.loadImage(buffer);
      const det = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (det) {
        knownFaces.push({ name: baseName, descriptor: det.descriptor });
        console.log(`    ✔️ Cara detectada y precargada: ${baseName}`);
      } else {
        console.log(`    ⚠️ No se detectó rostro en ${fullName}`);
      }
    } catch (e) {
      console.warn(`    ❌ Error procesando ${baseName}: ${e.message}`);
    }
  }

  console.log(`✅ Precarga finalizada. Total knownFaces: ${knownFaces.length}`);

  app.listen(5000, () => console.log('🚀 Server corriendo en http://localhost:5000'));
})();
