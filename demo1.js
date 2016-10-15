/*
demo1.js
This script should create a Keras model using JavaScript.
Ref:
https://github.com/transcranial/keras-js#usage
*/

const model = new KerasJS.Model({
  filepaths: {
    model: 'model.json',
    weights: 'model_weights.buf',
    metadata: 'model_metadata.json'
  }
  ,gpu: true
})

// last row in iris.csv:
// 5.9,3.0,5.1,1.8,virginica
var mydata = [5.9,3.0,5.1,1.8]

model.ready().then(() => {

  // input data object keyed by names of the input layers
  // or `input` for Sequential models
  // values are the flattened Float32Array data
  // (input tensor shapes are specified in the model config)
  const inputData = {
    'input': new Float32Array(mydata)
  }

  // make predictions
  // outputData is an object keyed by names of the output layers
  // or `output` for Sequential models
  model.predict(inputData).then(outputData => {
      var setosa_prob     = outputData["output"][0]
      var virginica_prob  = outputData["output"][1]
      var versicolor_prob = outputData["output"][2]
      console.log(['setosa_prob', 'virginica_prob', 'versicolor_prob'])
      console.log([setosa_prob, virginica_prob, versicolor_prob])
  })
})

'bye'
