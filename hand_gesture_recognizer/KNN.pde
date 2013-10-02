import org.opencv.ml.CvKNearest;

class KNN extends Classifier {
    CvKNearest classifier;

  KNN(PApplet parent){
    super(parent);
  }
  
  void train() {  
    Mat trainingMat = new Mat(trainingSamples.size(), trainingSamples.get(0).featureVector.length, CvType.CV_32FC1);
    Mat labelMat = new Mat( trainingSamples.size(), 1, CvType.CV_32FC1);

    // load samples into training and label mats. 
    for (int i = 0; i < trainingSamples.size(); i++) {
      Sample trainingSample = trainingSamples.get(i);
      trainingMat.put(0, i, trainingSample.featureVector);
      labelMat.put(i, 0, trainingSample.label);
    }

    classifier = new CvKNearest();
    classifier.train(trainingMat, labelMat);
  }

  // Use this function to get a prediction, after having trained the algorithm.
  double predict(Sample sample) {
    Mat predictionTraits = new Mat(1, sample.featureVector.length, CvType.CV_32FC1);
    predictionTraits.put(0, 0, sample.featureVector);

    return classifier.find_nearest(predictionTraits, 4, new Mat(), new Mat(), new Mat());
  }
}
