// I'd recommend you read the code starting from "function mouseMoving(event)" to get the main idea

document.addEventListener("mousemove", mouseMoving); // detect mouse position anywhere on page

// TODO: setting samplePeriod doesn't seem to affect setInterval()
var samplePeriod = 1000*2; // 1 per 2 seconds if samplePeriod = 1000*2 ; 10 per second if samplePeriod = 1000/10
var sampleTimer;
var snapshots = 50;
var confidenceThreshold = 80;
var rows = 3;
var columns = 3;
var neuralNet = create3DMatrix(snapshots,rows,columns);
var xNN = columns;
var yNN = rows;
var zNN = snapshots;
var wts_clockwise;
//var wts = create3DMatrix(snapshots,rows,columns); // so guarantee same size
var wts_updown;
var wts_leftright;
loadPretrainedWts();
var testInputMatrix = create3DMatrix(snapshots,rows,columns); // so guarantee same size

var w = window.innerWidth;
var h = window.innerHeight;

// set "access points" to HTML GUI elements:  CW = Clockwise ; UD = Up/Down ; LR = Left/Right
var meter_clockwise = document.getElementById("meter_clockwise");
var meter_updown = document.getElementById("meter_updown");
var meter_leftright = document.getElementById("meter_leftright");
var signal_clockwise = document.getElementById("signal_clockwise");
var signal_updown = document.getElementById("signal_updown");
var signal_leftright = document.getElementById("signal_leftright");
var player = document.getElementById("player");
player.style.top = h/2 + "px"; // initialize so that top can be changed
player.style.left = w/2 + "px";

function loadPretrainedWts() {
    //var matrix = [
    wts_clockwise = [
        [[0.86,0.86,0],[0.86,0.91,0.86],[0,0.86,0]],
        [[0.91,0.86,0],[0,0.86,0],[0.86,0,0.86]],
        [[0.91,0.86,0],[0,0.86,0],[0.86,0.86,0.86]],
        [[0.86,0.86,0],[0,0,0.86],[0.86,0.86,0.91]],
        [[0.91,0.86,0],[0,0,0.86],[0,0.86,0.86]],
        [[0.86,0.86,0],[0,0,0.86],[0.86,0.86,0.91]],
        [[0.86,0.91,0],[0,0,0],[0.86,0.86,0.86]],
        [[0.86,0.91,0],[0.86,0,0],[0.86,0.86,0.86]],
        [[0.86,0.86,0.86],[0,0,0],[0,0.86,0.91]],
        [[0.86,0.91,0.86],[0.86,0,0],[0.86,0.86,0.86]],
        [[0,0.91,0.86],[0,0,0],[0.86,0.86,0.86]],
        [[0,0.91,0.86],[0.86,0,0],[0.86,0,0.86]],
        [[0,0.86,0.86],[0.86,0,0.86],[0.86,0.86,0.91]],
        [[0.86,0.86,0.91],[0,0,0.86],[0.86,0.86,0.86]],
        [[0.86,0.86,0.86],[0.86,0,0.86],[0,0.86,0.91]],
        [[0.86,0,0.91],[0.86,0,0.86],[0,0.86,0.86]],
        [[0.86,0,0.86],[0.86,0,0.86],[0.86,0.86,0.91]],
        [[0.86,0,0.91],[0,0,0.86],[0,0.86,0.86]],
        [[0.86,0,0.86],[0,0,0.86],[0.86,0.86,0.91]],
        [[0.86,0.86,0],[0,0,0.91],[0,0.86,0.86]],
        [[0.86,0,0],[0,0,0.86],[0.86,0.86,0.91]],
        [[0.86,0.86,0],[0,0,0.91],[0,0.86,0.86]],
        [[0,0.86,0],[0.86,0,0.86],[0,0.91,0.86]],
        [[0,0.86,0],[0,0,0.91],[0,0.86,0.86]],
        [[0,0.86,0],[0.86,0,0.86],[0,0.91,0.86]],
        [[0,0.86,0.86],[0,0,0.86],[0,0.86,0.91]],
        [[0,0.86,0.86],[0.86,0,0.86],[0.86,0.91,0.86]],
        [[0.86,0.86,0.86],[0,0,0],[0.86,0.86,0.91]],
        [[0.86,0,0.86],[0,0,0.86],[0.86,0.91,0.86]],
        [[0,0,0.86],[0,0,0],[0.86,0.86,0.91]],
        [[0.86,0,0],[0.86,0,0.86],[0.86,0.91,0.86]],
        [[0,0,0.86],[0.86,0,0],[0.86,0.91,0.86]],
        [[0,0.86,0],[0.86,0,0.86],[0.86,0.91,0.86]],
        [[0,0,0],[0.86,0,0.86],[0.86,0.91,0.86]],
        [[0,0.86,0],[0.86,0,0.86],[0.91,0.86,0.86]],
        [[0,0.86,0],[0.86,0,0.86],[0.86,0.91,0.86]],
        [[0,0.86,0],[0.86,0,0.86],[0.91,0.86,0.86]],
        [[0.86,0.86,0],[0.86,0.86,0],[0.86,0.91,0.86]],
        [[0.86,0,0.86],[0.91,0.86,0.86],[0,0.86,0.86]],
        [[0.86,0.86,0],[0.86,0.86,0],[0.86,0.86,0.91]],
        [[0.86,0.86,0.86],[0.91,0.86,0],[0,0.86,0.86]],
        [[0.86,0.86,0],[0.86,0.86,0],[0.86,0.86,0.91]],
        [[0.86,0.86,0],[0.86,0.86,0.91],[0,0.86,0.86]],
        [[0,0.86,0],[0.86,0,0.86],[0.86,0.86,0.91]],
        [[0.86,0.86,0],[0.86,0.86,0],[0.86,0.91,0.86]],
        [[0.86,0.86,0],[0.86,0,0.86],[0,0.86,0.91]],
        [[0.86,0.86,0.86],[0.86,0.86,0],[0.86,0,0.91]],
        [[0.86,0.86,0.86],[0,0.86,0.86],[0.86,0,0.91]],
        [[0.86,0.86,0.86],[0.86,0.86,0],[0.86,0.91,0.86]],
        [[0.86,0.86,0.86],[0.86,0.86,0.86],[0,0.91,0.86]]
    ];
    wts_updown = [
        [[0,0,0],[0.86,0.91,0.86],[0,0,0]],
        [[0,0,0],[0.86,0.91,0.86],[0,0.86,0]],
        [[0,0,0],[0.86,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0,0.86],[0,0,0.91]],
        [[0,0,0],[0.86,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.86],[0,0.86,0.91]],
        [[0,0,0],[0.86,0.86,0.91],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0.86,0,0.91],[0,0.86,0.86]],
        [[0,0,0],[0.86,0,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.91],[0,0.86,0]],
        [[0,0,0],[0.86,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0.86],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.91,0.86],[0,0.86,0.86]],
        [[0,0,0],[0.91,0.86,0.86],[0,0,0.86]],
        [[0,0,0],[0.91,0.86,0.86],[0,0,0]],
        [[0,0,0],[0.91,0,0.86],[0,0,0.86]],
        [[0,0,0],[0.91,0,0.86],[0,0,0.86]],
        [[0,0,0],[0.91,0.86,0.86],[0,0,0.86]],
        [[0,0,0],[0.91,0.86,0.86],[0,0,0]],
        [[0,0,0],[0.91,0.86,0.86],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0.91,0.86,0.86],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0.91,0.86,0.86],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0.86,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0.86,0,0.91],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.86,0.86],[0,0.86,0.91]],
        [[0,0,0],[0.86,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.86],[0,0.86,0.91]],
        [[0,0,0],[0.86,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0.86,0,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0.86,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.86],[0,0.91,0.86]],
        [[0,0,0],[0.86,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0.91],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.86,0.91],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0.86],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0.91,0.86,0.86],[0,0,0.86]],
        [[0,0,0],[0.91,0.86,0.86],[0,0,0]],
        [[0,0,0],[0.91,0,0.86],[0,0,0.86]]
    ];
    wts_leftright=[
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.91,0],[0,0.86,0]],
        [[0,0.86,0],[0,0.91,0],[0,0.86,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.86,0.91]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0,0],[0,0.86,0.91]],
        [[0,0.86,0],[0,0,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.91,0],[0,0.86,0]],
        [[0,0.86,0],[0,0.91,0],[0,0.86,0]],
        [[0,0.91,0],[0,0,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.86,0],[0,0.91,0],[0,0.86,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.86,0.91]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.86,0],[0,0.91,0]],
        [[0,0.86,0],[0,0.91,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0.86,0]]
    ];
    //return matrix;
}

function create3DMatrix(snapshots,rows,columns) {
    var x = snapshots;
    var y = rows;
    var z = columns;
    var matrix = new Array(x);
    for (i = 0; i < x; i++) {
        matrix[i] = new Array(y);
        for (j = 0; j < y; j++) {
            matrix[i][j] = new Array(z);
            for(k = 0; k < z; k++) {
                matrix[i][j][k] = 0;
            }
        }
    }
    return matrix;
}

function create2DMatrix(rows,columns) {
    var y = rows;
    var z = columns;
    var matrix = new Array(y);
    for (j = 0; j < y; j++) {
        matrix[j] = new Array(z);
        for(k = 0; k < z; k++) {
            matrix[j][k] = 0;
        }
    }
    return matrix;
}

function mouseMoving(event) { // I'd recommend you read the code starting from here
    // start detecting
    var gesture = detectGesture(event);
    showGesture(gesture);
    playerAction(gesture);
}

function getDirectionVector(event) {
    var x = event.movementX;
    var y = event.movementY;
    return [x,y];
}

function updateNeuralNetwork(event, matrix) {
    shiftSamples(event, matrix); // show ML algorithm a gesture
}

function shiftSamples(event, matrix) {
    var inputSectionMatrix = getVelocityDirection(event); // getPadSection(event);
    var numberOfSnapshots = matrix.length;
    for (i = numberOfSnapshots-1; i > 0; i--) {
        matrix[i] = matrix[i-1];
    }
    matrix[0] = inputSectionMatrix; // example:  [[0,0,0],[0,0,0],[1,0,0]]
}

function getVelocityDirection(event) {
    var directionMatrix = create2DMatrix(3,3);
    var directionx, directiony;
    var vector = getDirectionVector(event); // [x,y]
    var dx = vector[0];
    var dy = vector[1];
    var slope = dy/dx;
    var thresholdMovementSize = 5;
    // get which section of the matrix to set to 1
    if (Math.abs(dx) < thresholdMovementSize && Math.abs(dy) < thresholdMovementSize) {
        directionx = 1;
        directiony = 1;
        directionMatrix[directionx][directiony] = 1;
    } else {
        if (dx < 0) {
            if (dy < 0) {
                if (slope > 2.414) {
                    directionx = 1;
                    directiony = 0;
                } else if (slope <= 2.414 && slope > 0.414) {
                    directionx = 0;
                    directiony = 0;
                } else if (slope <= 0.414) {
                    directionx = 0;
                    directiony = 1;
                }
            } else if (dy === 0) {
                directionx = 0;
                directiony = 1;
            } else if (dy > 0) {
                if (slope > -0.414) {
                    directionx = 0;
                    directiony = 1;
                } else if (slope <= -0.414 && slope > -2.414) {
                    directionx = 0;
                    directiony = 2;
                } else if (slope <= -2.414) {
                    directionx = 1;
                    directiony = 2;
                }
            }
        } else if (dx === 0) {
            if (dy < 0) {
                directionx = 1;
                directiony = 0;
            } else if (dy === 0) {
                directionx = 1;
                directiony = 1;
            } else if (dy > 0) {
                directionx = 1;
                directiony = 2;
            }
        } else if (dx > 0) {
            if (dy > 0) {
                if (slope > 2.414) {
                    directionx = 1;
                    directiony = 2;
                } else if (slope <= 2.414 && slope > 0.414) {
                    directionx = 2;
                    directiony = 2;
                } else if (slope <= 0.414) {
                    directionx = 2;
                    directiony = 1;
                }
            } else if (dy === 0) {
                directionx = 2;
                directiony = 1;
            } else if (dy < 0) {
                if (slope > -0.414) {
                    directionx = 2;
                    directiony = 1;
                } else if (slope <= -0.414 && slope > -2.414) {
                    directionx = 2;
                    directiony = 0;
                } else if (slope <= -2.414) {
                    directionx = 1;
                    directiony = 0;
                }
            }
        }
        // detect the section of the matrix to set to 1
        directionMatrix[directionx][directiony] = 1;
        //// make the rest have weights of -1
        //for (j = 0; j < rows; j++) {
        //    for(k = 0; k < columns; k++) {
        //        if (j !== directionx || k !== directiony) {
        //            directionMatrix[j][k] = -1;
        //        }
        //    }
        //}
    }
    return directionMatrix;
}

function round(x,digits) {
    return Math.round(x * Math.pow(10,digits)) / Math.pow(10,digits);
}

function detectGesture(event) {
    var gesture = "";
    var confidence_clockwise = 0;
    var confidence_updown = 0;
    var confidence_leftright = 0;
    var x = zNN;
    var y = yNN;
    var z = xNN;
    // track the input gesture
    trackGesture(event);
    // have ML algorithm try to categorize as gesture or not
    for (i = 0; i < x; i++) {
        for (j = 0; j < y; j++) {
            for(k = 0; k < z; k++) {
                weight_clockwise = wts_clockwise[i][j][k];
                weight_updown = wts_updown[i][j][k];
                weight_leftright = wts_leftright[i][j][k];
                input = Math.abs(testInputMatrix[i][j][k]);
                confidence_clockwise += weight_clockwise * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
                confidence_updown += weight_updown * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
                confidence_leftright += weight_leftright * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
            }
        }
    }
    // get final, rounded percent output value
    confidence_clockwise = round(confidence_clockwise,2); // round to 2 decimal places
    confidence_clockwise = confidence_clockwise*100; // get percentage
    confidence_updown = round(confidence_updown,2); // round to 2 decimal places
    confidence_updown = confidence_updown*100; // get percentage
    confidence_leftright = round(confidence_leftright,2); // round to 2 decimal places
    confidence_leftright = confidence_leftright*100; // get percentage
    // debug output
    meter_clockwise.value = confidence_clockwise/100;
    meter_updown.value = confidence_updown/100;
    meter_leftright.value = confidence_leftright/100;
    // set gesture and signal colour
    gesture = "?";
    if (confidence_clockwise > confidenceThreshold) {
        gesture = "CLOCKWISE CIRCLES";
        signal_clockwise.style.backgroundColor = "yellow";
        signal_clockwise.style.opacity = 1;
    } else {
        signal_clockwise.style.backgroundColor = "blue";
        signal_clockwise.style.opacity = 0.5;
    }
    if (confidence_updown > confidenceThreshold) {
        gesture = "UP/DOWN";
        signal_updown.style.backgroundColor = "yellow";
        signal_updown.style.opacity = 1;
    } else {
        signal_updown.style.backgroundColor = "blue";
        signal_updown.style.opacity = 0.5;
    }
    if (confidence_leftright > confidenceThreshold) {
        gesture = "LEFT/RIGHT";
        signal_leftright.style.backgroundColor = "yellow";
        signal_leftright.style.opacity = 1;
    } else {
        signal_leftright.style.backgroundColor = "blue";
        signal_leftright.style.opacity = 0.5;
    }
    return gesture;
}

function trackGesture(event) {
    sampleTimer = setInterval(updateNeuralNetwork(event, testInputMatrix), samplePeriod); // 1 per 2 seconds if samplePeriod = 1000*2 ; 10 per second if samplePeriod = 1000/10
}

function showGesture(gesture) {
    gesture.innerHTML = "Gesture:  " + gesture + ".";
}

function playerAction(gesture) {
    if (gesture === "UP/DOWN") {
        specialAction_UpDown();
    } else if (gesture === "LEFT/RIGHT") {
        specialAction_LeftRight();
    } else if (gesture === "CLOCKWISE CIRCLES") {
        specialAction_ClockWise();
    }
}

function specialAction_UpDown() {
    moveUp(player, 5);
}

function specialAction_LeftRight() {
    moveDown(player, 5);
}

function specialAction_ClockWise() {
    setTimeout( waitAndUp, 0 );
    setTimeout( waitAndRight, 100 );
    setTimeout( waitAndDown, 200 );
    setTimeout( waitAndLeft, 300 );
}

function moveRight(player, speed) {
    //player.style.left = parseInt(player.getBoundingClientRect().left) + speed + "px";
    player.style.left = parseInt(player.style.left) + speed + 'px'; // this alternate requires .left initialized
}

function moveLeft(player, speed) {
    //player.style.left = parseInt(player.getBoundingClientRect().left) - speed + "px";
    player.style.left = parseInt(player.style.left) - speed + 'px'; // this alternate requires .left initialized
}

function moveUp(player, speed) {
    // need this intialized beforehand:  player.style.top = '0px'; // initialize so that top can be changed
    player.style.top = parseInt(player.style.top) - speed + 'px';
}

function moveDown(player, speed) {
    // need this intialized beforehand:  player.style.top = '0px'; // initialize so that top can be changed
    player.style.top = parseInt(player.style.top) + speed + 'px';
}

function waitAndUp() {
    moveUp(player, 10);
}
function waitAndRight() {
    moveRight(player, 10);
}
function waitAndDown() {
    moveDown(player, 10);
}
function waitAndLeft() {
    moveLeft(player, 10);
}