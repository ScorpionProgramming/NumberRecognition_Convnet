
let fs = require('fs');
const convnetjs = require("./convnet.js");
  //test
let score = 0;

// 1. Rohdaten besorgen
let digits = [];
 
for (let i = 0; i < 10; i++) {
    digits.push(JSON.parse(fs.readFileSync(`./digits_as_json/${i}.json`)).data);
}
 
for (let i = 0; i < 10; i++) {
    console.log(`loaded digit ${i}: ${digits[i].length / 784}`);
}
 
// 2. Trainingsdaten vorbereiten
let trainingData = [];
for (let i = 0; i < 10; i++) {
    for (let count = 0; count < 400; count++) {
        let singleTrainingData = {
            // Problem: Das ist eindimensional, evtl. ändern auf 2D-Array mit 28*28
            input: new convnetjs.Vol(digits[i].slice(count * 784, (count + 1) * 784)),
            output: i
        };
        trainingData.push(singleTrainingData);
    }
}
 
// 3. Netz bauen
layer_defs = [];
layer_defs.push({ type: 'input', out_sx: 784, out_sy: 1, out_depth: 1 });
layer_defs.push({ type: 'fc', num_neurons: 80, activation: 'sigmoid' });
 
layer_defs.push({ type: 'softmax', num_classes: 10 });
 
net = new convnetjs.Net();
net.makeLayers(layer_defs);
 
trainer = new convnetjs.SGDTrainer(net, { method: 'adadelta', batch_size: 20, l2_decay: 0.001 });
 
// 4. Training
for (let i = 0; i < 100; i++) {
    trainingData.forEach(data => {
    trainer.train(data.input, data.output); // train the network, specifying that x is class zero
});
benchmark();
} 

// 5. Validieren
function benchmark() {
    for(let imagecount; imagecount < 200; imagecount++){
        for(let number; number < 10; number++){

            let obj = net.forward(new convnetjs.Vol(digits[number].slice(700+imagecount * 784, 70+imagecount+1 * 784)));
            
            console.log(`Laenge des DigitsArrays: ${digits.length}`);
            
            //console.log("erg: " + JSON.stringify(net.forward(new convnetjs.Vol(digits[5].slice(700 * 784, 701 * 784)))));
            
            console.log(`${number} =  ${obj.w[number]}`);
            let highest = 0;
            
            //get the hightest percent 
            //check every percent where what number peaks out
            for(let i = 0; i < 10; i++){
                if(obj.w[i] > obj.w[highest]){
                    highest = i;
                }
            }
            
            //if number who has the highest percent is equal to training number
            // add one to score
            if(highest == number){
                score = score + 1;
            }
            
            console.log(`Highest: ${highest} Score: ${score}`);
        }
    }
}