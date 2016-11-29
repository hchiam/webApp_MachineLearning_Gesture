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
var wts;
//var wts = create3DMatrix(snapshots,rows,columns); // so guarantee same size
var wts2_updown;
var wts3_leftright;
var wts4_right;
var wts5_up;
var wts6_down;
var wts7_left;
loadPretrainedWts();
var testInputMatrix = create3DMatrix(snapshots,rows,columns); // so guarantee same size

var w = window.innerWidth;
var h = window.innerHeight;

// set "access points" to HTML GUI elements:
var meter = document.getElementById("meter");
var meter2 = document.getElementById("meter2");
var meter3 = document.getElementById("meter3");
var meter4 = document.getElementById("meter4");
var meter5 = document.getElementById("meter5");
var meter6 = document.getElementById("meter6");
var signal = document.getElementById("signal");
var signal2 = document.getElementById("signal2");
var signal3 = document.getElementById("signal3");
var signal4 = document.getElementById("signal4");
var signal5 = document.getElementById("signal5");
var signal6 = document.getElementById("signal6");
var gesture = document.getElementById("gesture");
var player = document.getElementById("player");
player.style.top = h/2 + "px"; // initialize so that top can be changed
player.style.left = w/2 + "px";

function loadPretrainedWts() {
    //var matrix = [
    wts = [
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
    wts2_updown = [
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
    wts3_leftright=[
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
    wts4_right = [
        [[0,0,0],[0,0.86,0],[0.86,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.86,0.91]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.91,0],[0,0.86,0]],
        [[0,0,0],[0,0.91,0],[0,0.86,0.86]],
        [[0,0,0],[0,0.91,0],[0,0.86,0]],
        [[0,0,0],[0,0.91,0],[0,0.86,0.86]],
        [[0,0,0],[0,0.91,0],[0,0.86,0.86]],
        [[0,0,0],[0,0.86,0],[0.86,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]],
        [[0,0,0],[0,0.86,0],[0,0.86,0.91]],
        [[0,0,0],[0,0.86,0],[0.91,0.86,0]],
        [[0,0,0],[0,0.86,0.86],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0],[0,0.91,0]]
    ];
    wts5_up = [
        [[0,0,0],[0.86,0.91,0],[0,0.86,0]],
        [[0,0,0],[0.86,0.91,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0.86,0]],
        [[0,0,0],[0.86,0.91,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0.86,0.86]],
        [[0,0,0],[0.91,0.86,0],[0,0.86,0]],
        [[0,0,0],[0.91,0.86,0],[0,0,0.86]],
        [[0,0,0],[0.91,0.86,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0.91,0.86,0],[0,0.86,0.86]],
        [[0,0,0],[0.91,0.86,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.86,0],[0,0.91,0.86]],
        [[0,0,0],[0.91,0.86,0],[0,0,0.86]],
        [[0,0,0],[0.91,0.86,0],[0,0.86,0.86]],
        [[0,0,0],[0.91,0.86,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.86,0],[0,0,0.91]],
        [[0,0,0],[0.91,0,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0,0],[0,0,0.91]],
        [[0,0,0],[0.91,0,0],[0,0,0.86]],
        [[0,0,0],[0.86,0,0],[0,0,0.91]],
        [[0,0,0],[0.91,0,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0,0],[0,0,0.91]],
        [[0,0,0],[0.91,0,0],[0,0,0.86]],
        [[0,0,0],[0.86,0,0],[0,0,0.91]],
        [[0,0,0],[0.91,0,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0,0],[0,0,0.91]],
        [[0,0,0],[0.91,0,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.86,0],[0,0,0.91]],
        [[0,0,0],[0.91,0.86,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0],[0,0,0.91]],
        [[0,0,0],[0.91,0.86,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0],[0,0,0.91]],
        [[0,0,0],[0.91,0.86,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0],[0,0,0.91]],
        [[0,0,0],[0.91,0.86,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.86,0],[0,0.86,0.91]],
        [[0,0,0],[0.91,0.86,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0.86,0.86]],
        [[0,0,0],[0.86,0.91,0],[0,0.86,0.86]]
    ];
    wts6_down = [
        [[0,0,0],[0,0.91,0.86],[0,0,0]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0.86,0.91]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0.86,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0.86,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0.86,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0.86,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.91,0.86],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0.86,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0.86,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0,0.86,0.91],[0,0.86,0]],
        [[0,0,0],[0,0.86,0.86],[0,0,0.91]],
        [[0,0,0],[0,0.86,0.91],[0,0.86,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0.86,0.91]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0.91,0.86]],
        [[0,0,0],[0,0.86,0.91],[0,0,0.86]],
        [[0,0,0],[0,0.86,0.86],[0,0.91,0.86]]
    ];
    wts7_left = [
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.86,0],[0,0.86,0],[0,0,0.91]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.86,0],[0,0,0],[0,0,0.91]],
        [[0,0.91,0],[0,0,0],[0,0,0]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0,0],[0,0,0]],
        [[0,0.91,0],[0,0,0],[0,0,0]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.86,0],[0,0.86,0],[0,0,0.91]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0,0],[0,0.86,0.86]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.86,0],[0,0.86,0],[0,0,0.91]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.91,0],[0,0.86,0],[0,0,0]],
        [[0,0.86,0],[0,0.91,0],[0,0,0]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]],
        [[0,0.91,0],[0,0.86,0],[0,0,0.86]]
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
    updateSynapsesWeights(); // have ML algorithm set neuron synapse weights
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

function updateSynapsesWeights() {
    // numOfWts = number of weights, already defined at top
    // wts = matrix of weights, already defined at top
    var x = zNN;
    var y = yNN;
    var z = xNN;
    var sensitivity = 0.1;
    for (i = 0; i < x; i++) {
        for (j = 0; j < y; j++) {
            for(k = 0; k < z; k++) {
                wts[i][j][k] += neuralNet[i][j][k] * sensitivity;
                wts[i][j][k] = round( sigmoid( wts[i][j][k] ), 2 );
            }
        }
    }
}

function round(x,digits) {
    return Math.round(x * Math.pow(10,digits)) / Math.pow(10,digits);
}

function sigmoid(x) { // to keep number range within 0 to 1
    // 0 to 1
    //return 1 / (1 + Math.exp(-x*10+5));
    // -1 to 1
    return (1 / (1 + Math.exp(-x*3)) -0.5)*2; // "-x*3" instead of "-x*6" so wts can be decimal values (not so quick to get to 1)
    // "-0.5)*2" because want input=0 to give output=0
    // "-x*6" because want to compress plot to have input ranging from 0 to 1 (and not 0 to 6)
}

function detectGesture(event) {
    var gesture = "";
    var confidence = 0;
    var confidence2 = 0;
    var confidence3 = 0;
    var confidence4 = 0;
    var confidence5 = 0;
    var confidence6 = 0;
    var confidence7 = 0;
    var x = zNN;
    var y = yNN;
    var z = xNN;
    // track the input gesture
    trackGesture(event);
    // have ML algorithm try to categorize as gesture or not
    for (i = 0; i < x; i++) {
        for (j = 0; j < y; j++) {
            for(k = 0; k < z; k++) {
                weight = wts[i][j][k];
                weight2 = wts2_updown[i][j][k];
                weight3 = wts3_leftright[i][j][k];
                weight4 = wts4_right[i][j][k];
                weight5 = wts5_up[i][j][k];
                weight6 = wts6_down[i][j][k];
                weight7 = wts7_left[i][j][k];
                input = Math.abs(testInputMatrix[i][j][k]);
                confidence += weight * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
                confidence2 += weight2 * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
                confidence3 += weight3 * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
                confidence4 += weight4 * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
                confidence5 += weight5 * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
                confidence6 += weight6 * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
                confidence7 += weight7 * input /snapshots; // "/snapshots" to divide by the number of matching snapshots
            }
        }
    }
    // get final, rounded percent output value
    confidence = round(confidence,2); // round to 2 decimal places
    confidence = confidence*100; // get percentage
    confidence2 = round(confidence2,2); // round to 2 decimal places
    confidence2 = confidence2*100; // get percentage
    confidence3 = round(confidence3,2); // round to 2 decimal places
    confidence3 = confidence3*100; // get percentage
    confidence4 = round(confidence4,2); // round to 2 decimal places
    confidence4 = confidence4*100; // get percentage
    confidence5 = round(confidence5,2); // round to 2 decimal places
    confidence5 = confidence5*100; // get percentage
    confidence6 = round(confidence6,2); // round to 2 decimal places
    confidence6 = confidence6*100; // get percentage
    confidence7 = round(confidence7,2); // round to 2 decimal places
    confidence7 = confidence7*100; // get percentage
    // debug output
    meter.value = confidence/100;
    meter2.value = confidence2/100;
    meter3.value = confidence3/100;
    meter4.value = confidence4/100;
    meter5.value = confidence5/100;
    meter6.value = confidence6/100;
    meter7.value = confidence7/100;
    // set gesture and signal colour
    gesture = "?";
    if (confidence > confidenceThreshold) {
        gesture = "CLOCKWISE CIRCLES";
        signal.style.backgroundColor = "yellow";
        signal.style.opacity = 1;
    } else {
        signal.style.backgroundColor = "blue";
        signal.style.opacity = 0.5;
    }
    if (confidence2 > confidenceThreshold) {
        gesture = "UP/DOWN";
        signal2.style.backgroundColor = "yellow";
        signal2.style.opacity = 1;
    } else {
        signal2.style.backgroundColor = "blue";
        signal2.style.opacity = 0.5;
    }
    if (confidence3 > confidenceThreshold) {
        gesture = "LEFT/RIGHT";
        signal3.style.backgroundColor = "yellow";
        signal3.style.opacity = 1;
    } else {
        signal3.style.backgroundColor = "blue";
        signal3.style.opacity = 0.5;
    }
    if (confidence4 > confidenceThreshold) {
        gesture = "RIGHT";
        signal4.style.backgroundColor = "yellow";
        signal4.style.opacity = 1;
    } else {
        signal4.style.backgroundColor = "blue";
        signal4.style.opacity = 0.5;
    }
    if (confidence5 > confidenceThreshold) {
        gesture = "UP";
        signal5.style.backgroundColor = "yellow";
        signal5.style.opacity = 1;
    } else {
        signal5.style.backgroundColor = "blue";
        signal5.style.opacity = 0.5;
    }
    if (confidence6 > confidenceThreshold) {
        gesture = "DOWN";
        signal6.style.backgroundColor = "yellow";
        signal6.style.opacity = 1;
    } else {
        signal6.style.backgroundColor = "blue";
        signal6.style.opacity = 0.5;
    }
    if (confidence7 > confidenceThreshold) {
        gesture = "LEFT";
        signal7.style.backgroundColor = "yellow";
        signal7.style.opacity = 1;
    } else {
        signal7.style.backgroundColor = "blue";
        signal7.style.opacity = 0.5;
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
    if (gesture === "UP") {
        moveUp(player, 5);
    } else if (gesture === "DOWN") {
        moveDown(player, 5);
    } else if (gesture === "LEFT") {
        moveLeft(player, 5);
    } else if (gesture === "RIGHT") {
        moveRight(player, 5);
    } else if (gesture === "CLOCKWISE CIRCLES") {
        setTimeout( waitAndUp, 0 );
        setTimeout( waitAndRight, 100 );
        setTimeout( waitAndDown, 200 );
        setTimeout( waitAndLeft, 300 );
    }
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