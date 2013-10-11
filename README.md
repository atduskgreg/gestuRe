## GestuRe: A mixed-initiative interactive machine learning system for recognizing hand gestures

GestuRe is a mixed-initiative interactive machine learning system for recognizing hand gestures. It attempts to give the user visibility into the classifier's confidence for each class and control of the conditions under which it actively requests labeled gestures when its predictions are uncertain.

GestuRe is built with [OpenCV for Processing](http://github.com/atduskgreg/opencv-pro) and [PSVM](http://github.com/atduskgreg/processing-svm). It uses a Support Vector Machines classifier with a feature vector based on Histogram of Oriented Gradients calculated by OpenCV. It uses Libsvm for classification rather than OpenCV's native SVM implementation for improved performance.

See the system in action here: <a href="http://vimeo.com/76664145">GestuRe: A mixed-initiative interactive machine learning system for recognizing hand gestures</a> from <a href="http://vimeo.com/user1249829">Greg Borenstein</a> on <a href="https://vimeo.com">Vimeo</a>.