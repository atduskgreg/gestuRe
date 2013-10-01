import psvm.*;

class Libsvm extends Classifier {  
  SVM classifier;

  Libsvm(PApplet parent) {
    super( parent );
  }

  float[] doubleToFloat(double[] input){
    float[] result = new float[input.length];
    for(int i = 0; i < input.length; i++){
      result[i] = (float)input[i];
    }
    
    return result;
  }
  
  void train() {  

    float[][] trainingVectors = new float[trainingSamples.size()][trainingSamples.get(0).featureVector.length];
    int[] labels = new int[trainingSamples.size()];
    
    for(int i = 0; i < trainingSamples.size(); i++){
      trainingVectors[i] = doubleToFloat(trainingSamples.get(i).featureVector);
      labels[i] = trainingSamples.get(i).label;
    }
    
    classifier = new SVM(parent);

    classifier.params.kernel_type = SVM.RBF_KERNEL;

    SVMProblem problem = new SVMProblem();
    problem.setNumFeatures(numFeatures);
    problem.setSampleData(labels, trainingVectors);
    classifier.train(problem);
  }

  // Use this function to get a prediction, after having trained the algorithm.
  double predict(Sample sample) {
    return classifier.test(doubleToFloat(sample.featureVector));
  }
}
