const jsdom = require("jsdom");
const { JSDOM } = jsdom;
const fs = require('fs')
// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

// implements nodejs wrappers for HTMLCanvasElement, HTMLImageElement, ImageData
const canvas = require('canvas');

const faceapi = require('@vladmandic/face-api');

// patch nodejs environment, we need to provide an implementation of
// HTMLCanvasElement and HTMLImageElement
const { Canvas, Image, ImageData } = canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData, fetch: require('node-fetch') })

function studentsFromPage(file, output){
    fs.mkdirSync(output, { recursive: true });
    const pageText = fs.readFileSync(file,'utf-8');
    const dom = new JSDOM(pageText);

    let students = []
    dom.window.document.querySelectorAll("div.myImage").forEach( element => {
        let name = element.getAttribute("nome")
        console.log(name)
        let data = element.querySelector("img").getAttribute("src")
        data = data.replace(/^data:image\/png;base64,/, "")
        fs.writeFileSync(`${output}/${name}.png`,data,'base64')
        students.push(name)
    })
    
    return students;
}

async function buildLandmarks(students,folder) {
    const studentsWithDescriptors = []
    await faceapi.nets.ssdMobilenetv1.loadFromDisk('./model/ssd_model.manifest.json')
    await faceapi.nets.faceLandmark68Net.loadFromDisk('./model/face_landmark_68..manifest.json')
    await faceapi.nets.faceRecognitionNet.loadFromDisk('./model/face_recognition.manifest.json')
    for (const student of students) {
        const file = `${folder}/${student}.png`
        const image = fs.readFileSync(file);
        
        
        try {
            const input = await tf.node.decodeImage(new Uint8Array(image));
            const result = await faceapi.detectSingleFace(input)
                .withFaceLandmarks()
                .withFaceDescriptor()
            
            studentsWithDescriptors.push({
                name: student,
                descriptor: Array.from(result.descriptor)
            })
            console.log(student, "saved");
        } catch (error) {
            console.log(student, "saved but without descriptor");
            studentsWithDescriptors.push({
                name: student
            })
        }
        
    }

    fs.writeFileSync(`${output}/list.json`, JSON.stringify({
        total: students.length,
        students: studentsWithDescriptors
    }))
    console.log("Done");
}



let file = process.argv[2] 
let output = process.argv[3] || "output"

if(!file){
    console.error('provide a file')
    process.exit(1)
}

let students = studentsFromPage(file,output)
console.log("Found: ",students.length,"building descriptors...");
buildLandmarks(students,output)
