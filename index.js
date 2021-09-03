// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;
const csvWriter = createCsvWriter({
    path: './appello.csv',
    fieldDelimiter: ';',
    header: [
        { id: 'name', title: 'Nome' },
        { id: 'present', title: 'Presente' },
        { id: 'confidence', title: 'Confidenza' }
    ]
});

// implements nodejs wrappers for HTMLCanvasElement, HTMLImageElement, ImageData
const canvas = require('canvas');

const faceapi = require('@vladmandic/face-api');

// patch nodejs environment, we need to provide an implementation of
// HTMLCanvasElement and HTMLImageElement
const { Canvas, Image, ImageData } = canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData, fetch : require('node-fetch') })

const screenshot = require('screenshot-desktop')
const fs = require('fs');
const studentsData = process.argv[2] || "output/list.json"

console.log("Loading descriptors");
let students 
let studentsBackup 
try {
    students = JSON.parse(fs.readFileSync(studentsData))
    console.log("loaded",students.total);
    students = students.students;
    
} catch (error) {
    console.log("Can't load student lists");
    console.log(error)
    process.exit();
}

try {
    studentsBackup = JSON.parse(fs.readFileSync("./.backup.json"))

} catch (error) {
    console.log("Not using backup");
    studentsBackup = []
}

screenshot().then( async (img) => {
    // img: Buffer filled with jpg goodness
    // write image to file
    const model =  path.join(__dirname, './model/ssd_model.manifest.json').substring(3);
    const appello = students.map( s => ({ name: s.name , present: false }))
   // await faceapi.nets.ssdMobilenetv1.loadFromDisk(model);
    await faceapi.nets.ssdMobilenetv1.loadFromDisk('./model/ssd_model.manifest.json')
    await faceapi.nets.faceLandmark68Net.loadFromDisk('./model/face_landmark_68..manifest.json')
    await faceapi.nets.faceRecognitionNet.loadFromDisk('./model/face_recognition.manifest.json')

    const input = await tf.node.decodeImage(new Uint8Array(img));
    console.log("input ok");
    const detections = await faceapi
        .detectAllFaces(input)
        .withFaceLandmarks()
        .withFaceDescriptors()
    console.log("Detected ",detections.length,"faces");
    if (!detections.length) {
        console.log("No face detected; there's nobody here");
        await csvWriter.writeRecords(appello)
        return
    }

    const faceMatcher = new faceapi.FaceMatcher(detections)
    console.log("processing students");
    let found = 0;
    students.forEach(s => {
        if (s.descriptor){
            const bestMatch = faceMatcher.findBestMatch(s.descriptor)
            const confidence = 1 - bestMatch.distance;
            if(confidence > 0.45){
                console.log(s.name,"presente!")
                const out = appello.find(s2 => s2.name === s.name)
                out.present = true
                out.confidence =  confidence
                found++
            }else{
                console.log(s.name, "forse assente :(",confidence)
                const backup = studentsBackup.find(s2 => s2.name === s.name)
                if (backup && backup.present){
                    console.log("never mind student was already here")
                    const out = appello.find(s2 => s2.name === s.name)
                    out.present = true
                    out.confidence = backup.confidence
                    found++
                    return; 
                }
                const out = appello.find(s2 => s2.name === s.name)
                out.present = false
                out.confidence = confidence
            }
            
        }
    });
    console.log("fatto! Presenti:",found);
    await csvWriter.writeRecords(appello)
    fs.writeFileSync("./.backup.json",JSON.stringify(appello,null,4))
    fs.writeFile('screenshot.jpg', img, 'base64', (err) => {
        if (err) {
            console.log(err)
        }
    });
}).catch((err) => {
    console.log(err);
})