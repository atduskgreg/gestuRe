import org.opencv.ml.CvRTParams;
import org.opencv.ml.CvRTrees;

class RandomForest extends Classifier {  
  CvRTrees forest;

   RandomForest(PApplet parent) {
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

    // Begin magic numbers...
    // TODO: make this setable.
    CvRTParams params = new CvRTParams();
    params.set_max_depth(1000);
    params.set_min_sample_count(5);
    params.set_regression_accuracy(1);
    params.set_use_surrogates(false);
    params.set_max_categories(7);
    params.set_cv_folds(5);
    //params.set_truncate_pruned_tree(true);
    //params.set_use_1se_rule(true);
    // priors?????
    params.set_calc_var_importance(true);
    params.set_nactive_vars(numFeatures);
    params.set_term_crit(new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 100, 0.00f));

    forest = new CvRTrees();
    forest.train(trainingMat, 1, labelMat, new Mat(), new Mat(), varType, new Mat(), params);

  }

  // Use this function to get a prediction, after having trained the algorithm.

  double predict(Sample sample) {
    Mat predictionTraits = new Mat(1, sample.featureVector.length, CvType.CV_32FC1);
    predictionTraits.put(0, 0, sample.featureVector);
    return forest.predict(predictionTraits);
  }
}
