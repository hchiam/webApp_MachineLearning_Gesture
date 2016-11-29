# Machine Learning Web App:

Web app that uses machine learning to learn or detect mouse/cursor gestures.

Main files:
* `gestures.html`:  the "structure" of the presentation of the web page.
* `gestures.js`:  the "brains" of the web page. Tries to detect a mouse gesture when the mouse runs over the "pad".
* `gestures.css`:  the "looks/styling" of the presentation of the web page.

Extra files under "extras" folder:
* `gestures2.html`:  the version of `gestures.html` that detects more than one gesture.
* `gestures2.js`:  the version of `gestures.js` that detects more than one gesture.
* `split.py`:  convenience python file to format wts string (1 matrix per timestamp = 9 values per row) for copying-and-pasting or for creating visualizations of the neural network weights.

Uses a simple time delay neural network [https://en.wikipedia.org/wiki/Time_delay_neural_network](https://en.wikipedia.org/wiki/Time_delay_neural_network).

Instead of running the html file locally, you can try out the web app live here: [https://codepen.io/hchiam/pen/rrwQRa](https://codepen.io/hchiam/pen/rrwQRa).

![webApp](https://github.com/hchiam/webApp_MachineLearning_Gesture/blob/master/extras/LearnGesture.png "a web app that tries to detect a gesture made by the mouse anywhere on the page")

# Example Potential Use:

* For future mobile web apps/games using mouse path gestures.
* For interaction capabilities in web apps.

# Main Data Flow Steps:

* gestures.html

    1) onmouseover="mouseMoving(event);"

* gestures.js

    2) mouseMoving(event)

    3) learnGesture(event)

    4) updateSynapsesWeights()

    5) detectGesture(event)

# Example Gesture:

Making two quick clockwise circles with the mouse.  The following synapse weights and parameters make this happen.  The neural net can distinguish the motion from simple mouse cursor swipes, and even discriminate clockwise from counterclockise.  Small motions are automatically "filtered out" because of the thresholdMovementSize being > 0.  Training takes a while with confidenceThreshold = 90, but also helps "filter out" most false positives.

## Parameters that worked for "two quick clockwise circles with the mouse":

For gestures.js, with "confidence > 90, movement dx and dy both > 5 then 0 detection":

```
var snapshots = 50;
var confidenceThreshold = 90;
var thresholdMovementSize = 5;
if (Math.abs(dx) < thresholdMovementSize && Math.abs(dy) < thresholdMovementSize) {
    directionMatrix[directionx][directiony] = 0;
}
```

## Example Synapse Weights for "two quick clockwise circles with the mouse":

From this un-formatted string of numbers...

<sub>wts=1,1,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,0,0,1,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,0,1,0,0,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,1,1,1,0,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,0,1,0,0,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1,0.99,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,1,1</sub>

...we get this visualization:

Each "row" below = one timestamp or snapshot.  
Each "column" below = direction of mouse cursor motion at each timestamp.  
Notice any patterns?

wts = 

![LearnGesture_ExampleWts01.png](https://github.com/hchiam/webApp_MachineLearning_Gesture/blob/master/extras/LearnGesture_ExampleWts01.png)

## Parameter CHANGES that make it "ONE quick clockwise circle with the mouse":

For gestures.js:

```
var confidenceThreshold = 80;
```

And to not ignore the center of the matrix:


```
directionx = 1;
directiony = 1;
directionMatrix[directionx][directiony] = 1;
```

## Example Synapse Weights for "ONE quick clockwise circle with the mouse":

From this un-formatted string of numbers...

<sub>wts=0.86,0.86,0,0.86,0.91,0.86,0,0.86,0,0.91,0.86,0,0,0.86,0,0.86,0,0.86,0.91,0.86,0,0,0.86,0,0.86,0.86,0.86,0.86,0.86,0,0,0,0.86,0.86,0.86,0.91,0.91,0.86,0,0,0,0.86,0,0.86,0.86,0.86,0.86,0,0,0,0.86,0.86,0.86,0.91,0.86,0.91,0,0,0,0,0.86,0.86,0.86,0.86,0.91,0,0.86,0,0,0.86,0.86,0.86,0.86,0.86,0.86,0,0,0,0,0.86,0.91,0.86,0.91,0.86,0.86,0,0,0.86,0.86,0.86,0,0.91,0.86,0,0,0,0.86,0.86,0.86,0,0.91,0.86,0.86,0,0,0.86,0,0.86,0,0.86,0.86,0.86,0,0.86,0.86,0.86,0.91,0.86,0.86,0.91,0,0,0.86,0.86,0.86,0.86,0.86,0.86,0.86,0.86,0,0.86,0,0.86,0.91,0.86,0,0.91,0.86,0,0.86,0,0.86,0.86,0.86,0,0.86,0.86,0,0.86,0.86,0.86,0.91,0.86,0,0.91,0,0,0.86,0,0.86,0.86,0.86,0,0.86,0,0,0.86,0.86,0.86,0.91,0.86,0.86,0,0,0,0.91,0,0.86,0.86,0.86,0,0,0,0,0.86,0.86,0.86,0.91,0.86,0.86,0,0,0,0.91,0,0.86,0.86,0,0.86,0,0.86,0,0.86,0,0.91,0.86,0,0.86,0,0,0,0.91,0,0.86,0.86,0,0.86,0,0.86,0,0.86,0,0.91,0.86,0,0.86,0.86,0,0,0.86,0,0.86,0.91,0,0.86,0.86,0.86,0,0.86,0.86,0.91,0.86,0.86,0.86,0.86,0,0,0,0.86,0.86,0.91,0.86,0,0.86,0,0,0.86,0.86,0.91,0.86,0,0,0.86,0,0,0,0.86,0.86,0.91,0.86,0,0,0.86,0,0.86,0.86,0.91,0.86,0,0,0.86,0.86,0,0,0.86,0.91,0.86,0,0.86,0,0.86,0,0.86,0.86,0.91,0.86,0,0,0,0.86,0,0.86,0.86,0.91,0.86,0,0.86,0,0.86,0,0.86,0.91,0.86,0.86,0,0.86,0,0.86,0,0.86,0.86,0.91,0.86,0,0.86,0,0.86,0,0.86,0.91,0.86,0.86,0.86,0.86,0,0.86,0.86,0,0.86,0.91,0.86,0.86,0,0.86,0.91,0.86,0.86,0,0.86,0.86,0.86,0.86,0,0.86,0.86,0,0.86,0.86,0.91,0.86,0.86,0.86,0.91,0.86,0,0,0.86,0.86,0.86,0.86,0,0.86,0.86,0,0.86,0.86,0.91,0.86,0.86,0,0.86,0.86,0.91,0,0.86,0.86,0,0.86,0,0.86,0,0.86,0.86,0.86,0.91,0.86,0.86,0,0.86,0.86,0,0.86,0.91,0.86,0.86,0.86,0,0.86,0,0.86,0,0.86,0.91,0.86,0.86,0.86,0.86,0.86,0,0.86,0,0.91,0.86,0.86,0.86,0,0.86,0.86,0.86,0,0.91,0.86,0.86,0.86,0.86,0.86,0,0.86,0.91,0.86,0.86,0.86,0.86,0.86,0.86,0.86,0,0.91,0.86</sub>

...we get this visualization:

Each "row" below = one timestamp or snapshot.  
Each "column" below = direction of mouse cursor motion at each timestamp.  
Notice any patterns?

wts = 

![LearnGesture_ExampleWts.png](https://github.com/hchiam/webApp_MachineLearning_Gesture/blob/master/extras/LearnGesture_ExampleWts.png)

Here's that same info, animated:

![circle.gif](https://github.com/hchiam/webApp_MachineLearning_Gesture/blob/master/extras/circle.gif)