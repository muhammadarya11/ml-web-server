import tfjs from '@tensorflow/tfjs-node';

function loadModel() {
    const modelUrl = "file://models/model.json";
    return tfjs.loadLayersModel(modelUrl);
}

function predict(model, imageBuffer) {
    const tensor = tfjs.node
        .decodeJpeg(imageBuffer)
        .resizeNearestNeighbor([150, 150])
        .expandDims()
        .toFloat();

    return model.predict(tensor).data();
}

export { loadModel, predict }