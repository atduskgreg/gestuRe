import org.opencv.ml.CvBoost;
import org.opencv.ml.CvBoostParams;
import org.opencv.core.Range;

class AdaBoost extends Classifier {  
  CvBoost classifier;

  AdaBoost(PApplet parent) {
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

    Mat varType = new Mat(trainingMat.width()+1, 1, CvType.CV_8U );
    varType.setTo(new Scalar(0)); // 0 = CV_VAR_NUMERICAL.
    varType.put(trainingMat.width(), 0, 1); // 1 = CV_VAR_CATEGORICAL;

    CvBoostParams params = new CvBoostParams();
    params.set_boost_type(CvBoost.DISCRETE);
    params.set_weight_trim_rate(0);
//    params.set_weak_count(50000);
    params.set_cv_folds(3);
   

    classifier = new CvBoost();
    classifier.train(trainingMat, 1, labelMat, new Mat(), new Mat(), varType, new Mat(), params, false);
//    classifier.prune(new Range(0, (int)(params.get_weak_count() * 0.2)));
  }

  // Use this function to get a prediction, after having trained the algorithm.
  double predict(Sample sample) {
    Mat predictionTraits = new Mat(1, sample.featureVector.length, CvType.CV_32FC1);
    predictionTraits.put(0, 0, sample.featureVector);
    return classifier.predict(predictionTraits);
  }
}
