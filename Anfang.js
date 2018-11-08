
let fs = require('fs');
const convnetjs = require("./convnet.js");

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
    for (let count = 0; count < 600; count++) {
        let singleTrainingData = {
            // Problem: Das ist eindimensional, evtl. ändern auf 2D-Array mit 28*28
            input: new convnetjs.Vol(digits[i].slice(count * 784, (count + 1) * 784)),
            output: i
        };
        trainingData.push(singleTrainingData);
    }
}

//einmal die Trainingsdaten Shufflen
shuffle(trainingData);
 
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
//console.log("arbeitet in Runde " + i + " ......");
console.log(`${i+1}. Durchgang: `);
benchmark();
} 

// 5. Validieren
function benchmark() {
    let imageMax = 200;
    for(let number = 0; number < 10; number++){
        //Score - how good the network is trained.
        let score = 0; 
        for(let imagecount = 0; imagecount < imageMax; imagecount++){

            //console.log(`Imagecount: ${imagecount} | Number: ${number}` );
            let obj = net.forward(new convnetjs.Vol(digits[number].slice((600+imagecount) * 784, (600+imagecount+1) * 784)));
            
            //console.log("erg: " + JSON.stringify(net.forward(new convnetjs.Vol(digits[5].slice(700 * 784, 701 * 784)))));

            //console.log(`${number} = ${obj.w[number]}`);
            
            //get the hightest percent 
            //check every percent where what number peaks out
            let highest = 0;
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
            //console.log(`Highest: ${highest} Score: ${score}`);
        }
        //Auswertung
        console.log(`Auswertung für ${number}: ${score}/${imageMax} = ${Math.round(((score/imageMax)*100)*100)/100} % richtig`);
    }
    console.log("-------------------------------------------------------------------------------------------");
}


function shuffle(array){
    //bist du ein array? nein dann brech ab sonst weiter
    for(let i = 0; i < array.length; i++){
        let randompos = Math.floor(Math.random() * array.length);
        swap(array, i, randompos);
    }
}

function swap(array, pos1, pos2){
    let temp = array[pos1];
    array[pos1] = array[pos2];
    array[pos2] = temp;
}